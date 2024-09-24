# Grimoire

Grimoire is a server that implements long term memory for AI chatbots through processing messages and generating description of concepts such as locations, people, items. It's main purpose is for conversational AI chatbots.

### How it works
Grimoire collects the messages that are meant to be sent to LLM and analyzes them with Natural Language Processing (NLP), it sends collected messages to LLM in order to generate descriptions of concepts found in messages, and then the entries can be retrieved via API. It's essentially Retrieval Augmented Generation (RAG) system that collects and generates it's database throughout the conversation with AI.

### Setup
#### Prerequisites
To run Grimoire you need to have installed:
- Python 3.10 or above
- Docker
- Linux (for Windows run under WSL)

Copy the default settings file:
```bash
cp config/settings.default.yaml config/settings.yaml
```

Edit settings.yaml file with your values:
```yaml
DEBUG: True # Enables debug logs
LOG_PROMPTS: True # Enables prompt logging
summarization_api: # Api used for summarization
  backend: GenericOAI # Accepted values: GenericOAI, Kobold, KoboldCPP, Aphrodite, Tabby
  url: http://127.0.0.1:5002 # Url to side api that will summarize entries
  auth_key: "your-api-authkey" # Api key to summarization api, leave empty or delete entry if there is none
  input_sequence: "### Instruction:\n" # Instruct sequence for summarization api
  input_suffix: "\n"
  output_sequence: "### Response:\n" # Instruct sequence for summarization api
  output_suffix: "\n"
```

### Running from source

Create python virtual environment and enter it
```bash
python -m venv venv
source venv/bin/activate
```

Run containers for Grimoire dependencies (redis and postgres):
```bash
docker compose -f /docker/docker-compose-dev.yaml up -d 
```

Installing Grimoire requirements:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```

Setup database:

```bash
alembic upgrade head
```

To start a process that will make summarization prompts use the following command:

```bash
celery -A grimoire.core.tasks worker --loglevel=INFO --concurrency=1 -Q summarization_queue --pool=threads
```
Note: -concurrency=1 refers to how many prompts will be directed to side api at the same time. Leave it at 1 unless you know the backend supports proper queueing or batching.

And to run Grimoire API use:
```bash
python run.py
```
### Usage
Following backends are supported:

- Aphrodite
- Tabby
- KoboldCPP

Grimoire API starts by default on port 5005, you can view available endpoints at http://127.0.0.1:5005/docs. In order to run pipeline send POST request to `/grimoire/get_data` in following format.

```json
{
  "external_chat_id": "some_uid",
  "external_user_id": "some_uid",
  "max_tokens": 2000,
  "messages": [
    {
      "sender_name": "Some user",
      "text": "Hi"
    },
    {
      "sender_name": "Some Character",
      "text": "Hello How are you?"
    }
  ]
}
```

This will run messages through the pipeline, save them, queue new summaries, and return entries that you can insert to the prompt in following format. 
```json
[
  {
    "text": "Some summary 1",
    "relevance": 1
  },
  {
    "text": "Some summary 2",
    "relevance": 2
  }
]
```