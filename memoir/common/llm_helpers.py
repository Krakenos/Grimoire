import requests
from transformers import AutoTokenizer

from memoir.common.loggers import general_logger


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
        tokenize_endpoint = f'{api_url}/v1/tokenize'
        tokenize_json = {
            'prompt': text
        }
        tokenize_response = requests.post(tokenize_endpoint,
                                          json=tokenize_json,
                                          headers={'Authorization': f'Bearer {api_auth}'})
        if tokenize_response.status_code == 200:
            tokenized = tokenize_response.json()
            return tokenized['value']
        else:
            general_logger.warning(
                f'Tokenize endpoint not found for {api_type}, proceeding to count based on tokenizer')
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


def get_context_length(api_url: str) -> int:
    length_endpoint = api_url + '/v1/config/max_context_length'
    kobold_response = requests.get(length_endpoint)
    value = int(kobold_response.json()['value'])
    return value


def get_model_name(api_url: str, api_key):
    model_endpoint = api_url + '/v1/models'
    response = requests.get(model_endpoint, headers={'Authorization': f'Bearer {api_key}'})
    model_name = response.json()['data'][0]['root']
    return model_name


def generate_text(text: str, params: dict, api_type: str, api_url: str, api_key: str = None):
    if api_type == 'KoboldAI':
        request_body = {'prompt': text}
        request_body.update(params)
        response = requests.post(api_url + '/api/v1/generate', json=request_body)
        generated_text = response.json()['results'][0]['text']
    else:
        request_body = {'prompt': text}
        model_name = get_model_name(api_url, api_key)
        request_body['model'] = model_name
        request_body.update(params)
        endpoint = api_url + '/v1/completions'
        response = requests.post(endpoint, json=request_body, headers={'Authorization': f'Bearer {api_key}'})
        generated_text = response.json()['choices'][0]['text']
    return generated_text
