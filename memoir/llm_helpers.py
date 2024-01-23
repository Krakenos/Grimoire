import requests
from transformers import AutoTokenizer

from loggers import general_logger


def count_context(text: str, api_type: str, api_url: str, api_auth=None) -> int:
    """
    Counts the token length of the text either through endpoint or with local tokenizer
    :param text:
    :param api_type:
    :param api_url:
    :param api_auth:
    :return:
    """
    if api_type == 'Aphrodite':
        models_endpoint = api_url + '/v1/models'
        response = requests.get(models_endpoint, headers={'Authorization': f'Bearer {api_auth}'})
        model_name = response.json()['data'][0]['id']
        # Default to llama tokenizer if model tokenizer is not on huggingface
        try:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        except OSError as s:
            general_logger.warning('Could not load model tokenizer, defaulting to llama-tokenizer')
            general_logger.warning(s)
            tokenizer = AutoTokenizer.from_pretrained('oobabooga/llama-tokenizer')
        encoded = tokenizer(text)
        token_amount = len(encoded['input_ids'])
        return token_amount

    else:
        token_count_endpoint = api_url + '/api/extra/tokencount'
        request_body = {'prompt': text}
        kobold_response = requests.post(token_count_endpoint, json=request_body)
        value = int(kobold_response.json()['value'])
        return value
