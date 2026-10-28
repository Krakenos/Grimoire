import json
from unittest import TestCase, mock

from grimoire.core.settings import ApiSettings, SecondaryDatabaseSettings, SummarizationSettings, TokenizationSettings


def _settings(context_length=1000, max_tokens=100, thinking_budget=0, api_mode="text"):
    """Runtime settings stub with a scaffolding-free prompt, so budgets are easy to reason about."""
    runtime = mock.Mock()
    runtime.summarization_api = ApiSettings(
        backend="GenericOAI",
        url="http://sample.com",
        auth_key="secret",
        model="some-model",
        context_length=context_length,
        api_mode=api_mode,
        thinking_budget=thinking_budget,
        input_sequence="",
        input_suffix="",
        output_sequence="",
        output_suffix="",
    )
    runtime.summarization = SummarizationSettings(
        max_tokens=max_tokens,
        prompt="{previous_summary}{messages}",
        chat_system_prompt="{previous_summary}{messages}",
        chat_user_prompt="",
    )
    runtime.tokenization = TokenizationSettings()
    runtime.secondary_database = SecondaryDatabaseSettings()
    return runtime


def _token_count(batch, *args, **kwargs):
    """One token per character - lets the tests size inputs exactly."""
    return [len(text) for text in batch]


class LorebookHarness:
    """Runs generate_lorebook_entry with the API, tokenizer and Redis mocked out."""

    def __init__(self, testcase, runtime_settings, summary="S" * 10, summaries=None):
        self.prompts = []
        self.summary = summary
        # Per-call summaries, in order; once exhausted, every further call returns `summary`.
        self.summaries = list(summaries or [])
        patches = [
            mock.patch("grimoire.core.tasks.get_effective_settings", return_value=runtime_settings),
            mock.patch("grimoire.core.tasks.token_count", side_effect=_token_count),
            mock.patch("grimoire.core.tasks.generate_text", side_effect=self._generate),
            mock.patch("grimoire.core.tasks.redis_manager"),
        ]
        for patcher in patches:
            testcase.addCleanup(patcher.stop)
        self.mocks = [patcher.start() for patcher in patches]

    def _generate(self, prompt, *args, **kwargs):
        self.prompts.append(prompt)
        text = self.summaries.pop(0) if self.summaries else self.summary
        return mock.Mock(text=text, reasoning="")

    @property
    def call_count(self):
        return len(self.prompts)

    @property
    def stored_result(self):
        redis_manager = self.mocks[3]
        return json.loads(redis_manager.get_client.return_value.set.call_args.args[1])


class TestGenerateLorebookEntryBudget(TestCase):
    def _run(self, texts, runtime_settings, **kwargs):
        from grimoire.core.tasks import generate_lorebook_entry

        harness = LorebookHarness(self, runtime_settings, **kwargs)
        generate_lorebook_entry(ent_name="Kessler", texts=texts, request_id="req-1")
        return harness

    def test_response_tokens_are_reserved(self):
        # Texts total 400 chars/tokens (+1 newline each). Context 500 with a 200 token response
        # leaves ~300, so this cannot be done in a single generation.
        texts = ["A" * 100 for _ in range(4)]
        harness = self._run(texts, _settings(context_length=500, max_tokens=200))
        self.assertGreater(harness.call_count, 1)

    def test_fits_in_one_call_when_context_is_ample(self):
        texts = ["A" * 100 for _ in range(4)]
        harness = self._run(texts, _settings(context_length=5000, max_tokens=200))
        self.assertEqual(harness.call_count, 1)

    def test_thinking_budget_shrinks_the_text_budget(self):
        texts = ["A" * 100 for _ in range(4)]
        without = self._run(texts, _settings(context_length=900, max_tokens=100))
        self.assertEqual(without.call_count, 1)

        # Same context, but thinking_budget is added on top of the response reservation.
        for api_mode in ("text", "chat"):
            with self.subTest(api_mode=api_mode):
                with_budget = self._run(
                    texts, _settings(context_length=900, max_tokens=100, thinking_budget=400, api_mode=api_mode)
                )
                self.assertGreater(with_budget.call_count, 1)

    def test_oversized_text_is_skipped_not_silently_dropped(self):
        texts = ["A" * 10, "B" * 5000, "C" * 10]
        with self.assertLogs("general", level="WARNING") as logs:
            harness = self._run(texts, _settings(context_length=1000, max_tokens=100))
        self.assertTrue(any("does not fit" in line for line in logs.output))
        # The two small texts still make it into a generation.
        self.assertEqual(harness.call_count, 1)
        self.assertIn("A" * 10, harness.prompts[0])
        self.assertIn("C" * 10, harness.prompts[0])

    def test_no_text_is_lost_across_a_flush(self):
        # Each text is a distinct marker so every one can be traced into some prompt.
        texts = [chr(ord("a") + i) * 60 for i in range(8)]
        harness = self._run(texts, _settings(context_length=400, max_tokens=100))
        self.assertGreater(harness.call_count, 1)
        seen = "".join(harness.prompts)
        for text in texts:
            self.assertIn(text, seen, f"text {text[0]!r} was dropped")

    def test_aborts_when_context_cannot_fit_a_response(self):
        texts = ["A" * 50]
        with self.assertLogs("general", level="ERROR") as logs:
            harness = self._run(texts, _settings(context_length=100, max_tokens=300))
        self.assertTrue(any("too small" in line for line in logs.output))
        self.assertEqual(harness.call_count, 0)


class TestGenerateLorebookEntryEmptyOutput(TestCase):
    def _run(self, texts, runtime_settings, **kwargs):
        from grimoire.core.tasks import generate_lorebook_entry

        harness = LorebookHarness(self, runtime_settings, **kwargs)
        generate_lorebook_entry(ent_name="Kessler", texts=texts, request_id="req-1")
        return harness

    def test_empty_definition_reports_failure(self):
        with self.assertLogs("general", level="WARNING"):
            harness = self._run(["A" * 100], _settings(context_length=5000), summary="")
        self.assertEqual(harness.stored_result["status"], "failed")

    def test_empty_later_chunk_keeps_earlier_definition(self):
        texts = [chr(ord("a") + i) * 60 for i in range(8)]
        with self.assertLogs("general", level="WARNING"):
            harness = self._run(
                texts, _settings(context_length=400, max_tokens=100), summaries=["first definition"], summary=""
            )
        self.assertGreater(harness.call_count, 1)
        self.assertEqual(harness.stored_result, {"status": "success", "result": "first definition"})


class TaskTestCase(TestCase):
    """Runs a summarization task with settings, DB session, API and embeddings mocked out."""

    def _patch(self, target, **kwargs):
        patcher = mock.patch(target, **kwargs)
        self.addCleanup(patcher.stop)
        return patcher.start()

    def _patch_task(self, generated_text):
        self._patch("grimoire.core.tasks.get_effective_settings", return_value=_settings(context_length=5000))
        self._patch("grimoire.core.tasks.token_count", side_effect=_token_count)
        self._patch(
            "grimoire.core.tasks.generate_text",
            return_value=mock.Mock(text=generated_text, reasoning="ran out mid-thou", request_body={}),
        )
        self._patch("grimoire.core.vector_embeddings.get_text_embeddings", return_value=[[0.0]])
        session_cls = self._patch("grimoire.core.tasks.Session")
        self.session = session_cls.return_value.__enter__.return_value


class TestDescribeEntityEmptyOutput(TaskTestCase):
    def _run(self, generated_text):
        from grimoire.core.tasks import describe_entity

        self._patch_task(generated_text)
        self._patch("grimoire.core.tasks.make_summary_prompt", return_value="a prompt")
        entry = mock.Mock(entity="Kessler", summary="old summary", update_count=5, frozen=False, enabled=True)
        self._patch("grimoire.core.tasks.get_knowledge_entity", return_value=entry)
        describe_entity("Kessler", 1, [])
        return entry

    def test_empty_summary_keeps_previous_one(self):
        for generated_text in ("", "  \n"):
            with self.subTest(generated_text=generated_text):
                with self.assertLogs("general", level="WARNING"):
                    entry = self._run(generated_text)
                self.assertEqual(entry.summary, "old summary")
                self.session.commit.assert_not_called()

    def test_summary_is_saved(self):
        entry = self._run("new summary")
        self.assertEqual(entry.summary, "new summary")
        self.assertEqual(entry.summary_entry, "[ Kessler: new summary ]")
        self.session.commit.assert_called_once()


class TestGenerateSegmentedMemoryEmptyOutput(TaskTestCase):
    def _run(self, generated_text):
        from grimoire.core.tasks import generate_segmented_memory

        self._patch_task(generated_text)
        character = mock.Mock()
        character.name = "Alex"
        messages = [mock.Mock(message=f"message {i}", character=character) for i in range(3)]
        self._patch("grimoire.core.tasks.get_messages_by_index", return_value=messages)
        self.memory_cls = self._patch("grimoire.core.tasks.SegmentedMemory")
        generate_segmented_memory(chat_id=1, start_index=0, end_index=2)

    def test_empty_memory_is_not_saved(self):
        with self.assertLogs("general", level="WARNING"):
            self._run("")
        self.memory_cls.assert_not_called()
        self.session.add.assert_not_called()
        self.session.commit.assert_not_called()

    def test_memory_is_saved(self):
        self._run("what happened")
        self.assertEqual(self.memory_cls.call_args.kwargs["summary"], "what happened")
        self.session.add.assert_called_once()
        self.session.commit.assert_called_once()
