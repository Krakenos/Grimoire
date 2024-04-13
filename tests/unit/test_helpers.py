from unittest import TestCase, mock

from grimoire.common.llm_helpers import get_context_length

responses = {
    'http://sample.com/v1/config/max_context_length': {"value": 10}
}


def mocked_requests_get(*args, **kwargs):
    class MockResponse:
        def __init__(self, json_data, status_code):
            self.json_data = json_data
            self.status_code = status_code

        def json(self):
            return self.json_data

    response_value = responses.get(args[0], None)

    if response_value:
        return MockResponse(response_value, 200)

    return MockResponse(None, 404)


class TestGetContextLength(TestCase):
    @mock.patch('requests.get', side_effect=mocked_requests_get)
    def test_returns_value(self, mock_requests_get):
        result = get_context_length('http://sample.com')
        assert result == 10
