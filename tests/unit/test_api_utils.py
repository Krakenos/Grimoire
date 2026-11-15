from unittest import TestCase, mock

from grimoire.common.api_utils import get_autolorebook


def _terminal(result: str, keys: list[str] | None = None) -> dict:
    """A finished entry as surfaced by lorebook_status (status is a dict payload)."""
    return {"status": {"status": "success", "result": result}, "keys": keys or []}


def _pending(keys: list[str] | None = None) -> dict:
    """An entry whose generation task has not written its result yet."""
    return {"status": "pending", "keys": keys or []}


class TestGetAutolorebook(TestCase):
    @mock.patch("grimoire.common.api_utils.lorebook_status", return_value={})
    def test_error_when_request_unknown_or_expired(self, _mock_status):
        response = get_autolorebook("missing-id")
        self.assertEqual("error", response.status)
        self.assertIsNone(response.lorebook)

    @mock.patch("grimoire.common.api_utils.lorebook_status")
    def test_processing_when_any_entry_pending(self, mock_status):
        mock_status.return_value = {
            "Alice": _terminal("Alice is a knight."),
            "Bob": _pending(),
        }
        response = get_autolorebook("req-1")
        self.assertEqual("processing", response.status)
        # Only the finished entry is included so far.
        self.assertEqual(1, len(response.lorebook.entries))

    @mock.patch("grimoire.common.api_utils.lorebook_status")
    def test_done_when_all_entries_terminal(self, mock_status):
        mock_status.return_value = {
            "Alice": _terminal("Alice is a knight.", ["Alice"]),
            "Bob": _terminal("Bob is a baker.", ["Bob"]),
        }
        response = get_autolorebook("req-2")
        self.assertEqual("done", response.status)
        self.assertEqual(2, len(response.lorebook.entries))
        self.assertEqual("Alice is a knight.", response.lorebook.entries["0"].content)
