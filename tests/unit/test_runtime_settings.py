import json
from types import SimpleNamespace
from unittest import TestCase, mock

from grimoire.core.runtime_settings import get_effective_settings
from grimoire.core.settings import settings as file_settings


def _fake_session(rows):
    """A minimal stand-in for a SQLAlchemy Session that only needs to answer
    `session.scalars(query).all()` with the given SettingOverride-like rows.
    """
    session = mock.MagicMock()
    session.scalars.return_value.all.return_value = rows
    return session


def _override_row(section: str, value: dict) -> SimpleNamespace:
    return SimpleNamespace(section=section, value=json.dumps(value))


class TestGetEffectiveSettings(TestCase):
    def test_no_overrides_returns_file_defaults(self):
        result = get_effective_settings(_fake_session([]))

        self.assertEqual(file_settings.summarization_api.url, result.summarization_api.url)
        self.assertEqual(file_settings.summarization_api.backend, result.summarization_api.backend)
        self.assertEqual(file_settings.summarization.limit_rate, result.summarization.limit_rate)
        self.assertEqual(file_settings.tokenization.local_tokenizer, result.tokenization.local_tokenizer)

    def test_override_merges_over_file_section(self):
        session = _fake_session([_override_row("summarization_api", {"model": "custom-model"})])
        result = get_effective_settings(session)

        self.assertEqual("custom-model", result.summarization_api.model)
        # Fields not touched by the override still come from the file default.
        self.assertEqual(file_settings.summarization_api.backend, result.summarization_api.backend)
        self.assertEqual(file_settings.summarization_api.url, result.summarization_api.url)

    def test_multiple_section_overrides_are_all_applied(self):
        session = _fake_session(
            [
                _override_row("summarization_api", {"context_length": 8192}),
                _override_row("tokenization", {"prefer_local_tokenizer": False}),
            ]
        )
        result = get_effective_settings(session)

        self.assertEqual(8192, result.summarization_api.context_length)
        self.assertFalse(result.tokenization.prefer_local_tokenizer)

    def test_existing_validators_still_run_on_overridden_values(self):
        # A literal two-character "\n" (as would come from an env-style raw string) should still
        # be normalized to a real newline by ApiSettings.replace_newline.
        session = _fake_session([_override_row("summarization_api", {"input_sequence": "### Instruction:\\n"})])
        result = get_effective_settings(session)

        self.assertEqual("### Instruction:\n", result.summarization_api.input_sequence)

    def test_params_override_still_gets_stop_keys_defaulted(self):
        # The "summarization" section override replaces the whole `params` dict (the router
        # always writes a complete section, not a deep patch) — so a params dict missing
        # stop/stop_sequence should still get them defaulted by SummarizationSettings.add_stop.
        session = _fake_session([_override_row("summarization", {"params": {"temperature": 0.9}})])
        result = get_effective_settings(session)

        self.assertEqual(0.9, result.summarization.params["temperature"])
        self.assertIn("stop", result.summarization.params)
        self.assertIn("stop_sequence", result.summarization.params)

    def test_malformed_override_value_is_ignored(self):
        session = _fake_session([SimpleNamespace(section="summarization_api", value="not valid json")])
        result = get_effective_settings(session)

        self.assertEqual(file_settings.summarization_api.backend, result.summarization_api.backend)
