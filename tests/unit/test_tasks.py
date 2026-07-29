from unittest import TestCase, mock

from grimoire.core.settings import ApiSettings, SummarizationSettings, TokenizationSettings


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
    return runtime


def _token_count(batch, *args, **kwargs):
    """One token per character - lets the tests size inputs exactly."""
    return [len(text) for text in batch]


class LorebookHarness:
    """Runs generate_lorebook_entry with the API, tokenizer and Redis mocked out."""

    def __init__(self, testcase, runtime_settings, summary="S" * 10):
        self.prompts = []
        self.summary = summary
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
        return mock.Mock(text=self.summary, reasoning="")

    @property
    def call_count(self):
        return len(self.prompts)


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

        # Same context, but chat mode adds thinking_budget on top of the response reservation.
        with_budget = self._run(
            texts, _settings(context_length=900, max_tokens=100, thinking_budget=400, api_mode="chat")
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
