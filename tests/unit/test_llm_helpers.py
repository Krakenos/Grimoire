from unittest import TestCase, mock

from grimoire.common.llm_helpers import generate_text, is_chat_mode, split_reasoning
from grimoire.core.settings import ApiSettings


class MockResponse:
    def __init__(self, json_data, status_code=200):
        self.json_data = json_data
        self.status_code = status_code

    def json(self):
        return self.json_data


def _api_settings(**overrides) -> ApiSettings:
    defaults = {"backend": "GenericOAI", "url": "http://sample.com", "auth_key": "secret", "model": "some-model"}
    defaults.update(overrides)
    return ApiSettings(**defaults)


class TestIsChatMode(TestCase):
    def test_text_mode_is_not_chat(self):
        self.assertFalse(is_chat_mode(_api_settings(api_mode="text")))

    def test_chat_mode_on_genericoai(self):
        self.assertTrue(is_chat_mode(_api_settings(api_mode="chat")))

    def test_chat_mode_falls_back_for_kobold_backends(self):
        self.assertFalse(is_chat_mode(_api_settings(api_mode="chat", backend="koboldcpp")))
        self.assertFalse(is_chat_mode(_api_settings(api_mode="chat", backend="koboldai")))


class TestSplitReasoning(TestCase):
    def test_full_think_pair(self):
        content, reasoning = split_reasoning("<think>pondering things</think>The actual answer.")
        self.assertEqual(content, "The actual answer.")
        self.assertEqual(reasoning, "pondering things")

    def test_orphan_closing_tag(self):
        # Happens when the prompt prefills the opening <think> tag itself.
        content, reasoning = split_reasoning("pondering things</think>The actual answer.")
        self.assertEqual(content, "The actual answer.")
        self.assertEqual(reasoning, "pondering things")

    def test_no_tags(self):
        content, reasoning = split_reasoning("Just a plain answer.")
        self.assertEqual(content, "Just a plain answer.")
        self.assertEqual(reasoning, "")

    def test_multiple_pairs(self):
        text = "<think>first</think>mid<think>second</think>final answer"
        content, reasoning = split_reasoning(text)
        self.assertEqual(content, "final answer")
        self.assertEqual(reasoning, "first</think>midsecond")

    def test_strip_reasoning_disabled_returns_unchanged(self):
        text = "<think>pondering</think>answer"
        content, reasoning = split_reasoning(text, strip_reasoning=False)
        self.assertEqual(content, text)
        self.assertEqual(reasoning, "")

    def test_empty_text(self):
        self.assertEqual(split_reasoning(""), ("", ""))

    def test_custom_tokens(self):
        text = "<reasoning>pondering things</reasoning>The actual answer."
        content, reasoning = split_reasoning(text, start_token="<reasoning>", end_token="</reasoning>")
        self.assertEqual(content, "The actual answer.")
        self.assertEqual(reasoning, "pondering things")

    def test_custom_tokens_ignore_default_tags(self):
        text = "<think>pondering things</think>The actual answer."
        content, reasoning = split_reasoning(text, start_token="<reasoning>", end_token="</reasoning>")
        self.assertEqual(content, text)
        self.assertEqual(reasoning, "")

    def test_custom_orphan_end_token(self):
        content, reasoning = split_reasoning(
            "pondering things<|end|>The actual answer.",
            start_token="<|start|>",
            end_token="<|end|>",
        )
        self.assertEqual(content, "The actual answer.")
        self.assertEqual(reasoning, "pondering things")

    def test_empty_end_token_disables_splitting(self):
        text = "<think>pondering</think>answer"
        content, reasoning = split_reasoning(text, start_token="", end_token="")
        self.assertEqual(content, text)
        self.assertEqual(reasoning, "")


class TestGenerateTextTextMode(TestCase):
    @mock.patch("requests.post")
    def test_generic_oai_posts_to_completions(self, mock_post):
        mock_post.return_value = MockResponse({"choices": [{"text": "a summary"}]})
        api_settings = _api_settings(api_mode="text")

        result = generate_text("a flat prompt", {"max_tokens": 300}, api_settings)

        called_url = mock_post.call_args.args[0]
        called_json = mock_post.call_args.kwargs["json"]
        self.assertEqual(called_url, "http://sample.com/v1/completions")
        self.assertEqual(called_json["prompt"], "a flat prompt")
        self.assertEqual(called_json["model"], "some-model")
        self.assertEqual(result.text, "a summary")
        self.assertEqual(result.reasoning, "")

    @mock.patch("requests.post")
    def test_kobold_posts_to_generate(self, mock_post):
        mock_post.return_value = MockResponse({"results": [{"text": "a summary"}]})
        api_settings = _api_settings(backend="koboldcpp", api_mode="text")

        result = generate_text("a flat prompt", {"max_length": 300}, api_settings)

        called_url = mock_post.call_args.args[0]
        called_json = mock_post.call_args.kwargs["json"]
        self.assertEqual(called_url, "http://sample.com/api/v1/generate")
        self.assertEqual(called_json["prompt"], "a flat prompt")
        self.assertNotIn("model", called_json)
        self.assertEqual(result.text, "a summary")

    @mock.patch("requests.post")
    def test_kobold_falls_back_to_text_when_chat_requested(self, mock_post):
        mock_post.return_value = MockResponse({"results": [{"text": "a summary"}]})
        api_settings = _api_settings(backend="koboldcpp", api_mode="chat")

        result = generate_text("a flat prompt", {"max_length": 300}, api_settings)

        called_url = mock_post.call_args.args[0]
        self.assertEqual(called_url, "http://sample.com/api/v1/generate")
        self.assertEqual(result.text, "a summary")

    @mock.patch("requests.post")
    def test_model_override_reaches_request_body(self, mock_post):
        # Regression test: generate_text used to read settings.summarization_api.model from the
        # import-time global singleton instead of the api_settings argument, so panel overrides
        # of `model` never reached the request body.
        mock_post.return_value = MockResponse({"choices": [{"text": "a summary"}]})
        api_settings = _api_settings(model="overridden-model")

        generate_text("a flat prompt", {"max_tokens": 300}, api_settings)

        called_json = mock_post.call_args.kwargs["json"]
        self.assertEqual(called_json["model"], "overridden-model")

    @mock.patch("requests.post")
    def test_configured_reasoning_tokens_are_split_out(self, mock_post):
        mock_post.return_value = MockResponse({"choices": [{"text": "<reason>pondering</reason>a summary"}]})
        api_settings = _api_settings(reasoning_start_token="<reason>", reasoning_end_token="</reason>")

        result = generate_text("a flat prompt", {"max_tokens": 300}, api_settings)

        self.assertEqual(result.text, "a summary")
        self.assertEqual(result.reasoning, "pondering")

    @mock.patch("requests.post")
    def test_default_think_tags_kept_when_other_tokens_configured(self, mock_post):
        mock_post.return_value = MockResponse({"choices": [{"text": "<think>pondering</think>a summary"}]})
        api_settings = _api_settings(reasoning_start_token="<reason>", reasoning_end_token="</reason>")

        result = generate_text("a flat prompt", {"max_tokens": 300}, api_settings)

        self.assertEqual(result.text, "<think>pondering</think>a summary")
        self.assertEqual(result.reasoning, "")


class TestGenerateTextChatMode(TestCase):
    @mock.patch("requests.post")
    def test_posts_messages_to_chat_completions(self, mock_post):
        mock_post.return_value = MockResponse({"choices": [{"message": {"content": "a summary"}}]})
        api_settings = _api_settings(api_mode="chat")
        messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": "usr"}]
        params = {
            "max_length": 300,
            "max_tokens": 300,
            "truncation_length": 4096,
            "max_context_length": 4096,
            "stop": ["</s>"],
            "stop_sequence": ["</s>"],
            "temperature": 0.6,
        }

        result = generate_text(messages, params, api_settings)

        called_url = mock_post.call_args.args[0]
        called_json = mock_post.call_args.kwargs["json"]
        self.assertEqual(called_url, "http://sample.com/v1/chat/completions")
        self.assertEqual(called_json["messages"], messages)
        self.assertEqual(called_json["model"], "some-model")
        self.assertNotIn("max_length", called_json)
        self.assertNotIn("truncation_length", called_json)
        self.assertNotIn("max_context_length", called_json)
        self.assertNotIn("stop_sequence", called_json)
        self.assertEqual(called_json["stop"], ["</s>"])
        self.assertNotIn("reasoning_effort", called_json)
        self.assertNotIn("chat_template_kwargs", called_json)
        self.assertEqual(result.text, "a summary")

    @mock.patch("requests.post")
    def test_reasoning_effort_and_chat_template_kwargs_sent_when_configured(self, mock_post):
        mock_post.return_value = MockResponse({"choices": [{"message": {"content": "a summary"}}]})
        api_settings = _api_settings(
            api_mode="chat", reasoning_effort="low", chat_template_kwargs={"enable_thinking": False}
        )

        generate_text([{"role": "user", "content": "hi"}], {"max_tokens": 300}, api_settings)

        called_json = mock_post.call_args.kwargs["json"]
        self.assertEqual(called_json["reasoning_effort"], "low")
        self.assertEqual(called_json["chat_template_kwargs"], {"enable_thinking": False})

    @mock.patch("requests.post")
    def test_thinking_budget_added_on_top_of_max_tokens(self, mock_post):
        mock_post.return_value = MockResponse({"choices": [{"message": {"content": "a summary"}}]})
        api_settings = _api_settings(api_mode="chat", thinking_budget=1024)

        generate_text([{"role": "user", "content": "hi"}], {"max_tokens": 300}, api_settings)

        called_json = mock_post.call_args.kwargs["json"]
        self.assertEqual(called_json["max_tokens"], 1324)

    @mock.patch("requests.post")
    def test_reasoning_content_captured_and_kept_out_of_text(self, mock_post):
        mock_post.return_value = MockResponse(
            {"choices": [{"message": {"content": "clean answer", "reasoning_content": "internal musing"}}]}
        )
        api_settings = _api_settings(api_mode="chat")

        result = generate_text([{"role": "user", "content": "hi"}], {"max_tokens": 300}, api_settings)

        self.assertEqual(result.text, "clean answer")
        self.assertEqual(result.reasoning, "internal musing")

    @mock.patch("requests.post")
    def test_openrouter_style_reasoning_field(self, mock_post):
        mock_post.return_value = MockResponse(
            {"choices": [{"message": {"content": "clean answer", "reasoning": "internal musing"}}]}
        )
        api_settings = _api_settings(api_mode="chat")

        result = generate_text([{"role": "user", "content": "hi"}], {"max_tokens": 300}, api_settings)

        self.assertEqual(result.reasoning, "internal musing")

    @mock.patch("requests.post")
    def test_unstructured_think_tags_are_stripped_from_content(self, mock_post):
        mock_post.return_value = MockResponse(
            {"choices": [{"message": {"content": "<think>pondering</think>clean answer"}}]}
        )
        api_settings = _api_settings(api_mode="chat")

        result = generate_text([{"role": "user", "content": "hi"}], {"max_tokens": 300}, api_settings)

        self.assertEqual(result.text, "clean answer")
        self.assertEqual(result.reasoning, "pondering")

    @mock.patch("requests.post")
    def test_strip_reasoning_disabled_keeps_think_tags(self, mock_post):
        mock_post.return_value = MockResponse(
            {"choices": [{"message": {"content": "<think>pondering</think>clean answer"}}]}
        )
        api_settings = _api_settings(api_mode="chat", strip_reasoning=False)

        result = generate_text([{"role": "user", "content": "hi"}], {"max_tokens": 300}, api_settings)

        self.assertEqual(result.text, "<think>pondering</think>clean answer")
        self.assertEqual(result.reasoning, "")
