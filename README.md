# Grimoire

Grimoire is a server that implements long term memory for AI chatbots through processing prompts and generating description of concepts such as locations, people, items. It's main purpose is for conversational AI chatbots. The server is meant to be slotted between frontend that interacts with LLM(such as SillyTavern) and backend that host the LLM(Aphrodite, KoboldAI etc).

### How it works
Grimoire collects the prompts that are meant to be sent to LLM and analyzes them with Natural Language Processing (NLP), it sends collected prompts to generate descriptions of concepts found in messages, and in the end, it inserts them into current context. It's essentially Retrieval Augmented Generation (RAG) system that collects and generates it's database throughout the conversation with AI. It can run with 2 language models. One for the actual chatbot, and other for summarization tasks, or with one that does both of those things.

### IMPORTANT NOTE
Grimoire is still VERY EARLY in development. There is bound to be a lot of bugs, there will be breaking changes in codebase and it's not stable. You will most likely lose the data that Grimoire has collected along the way. It's a prototype that's not production ready.
### Setup
#### Prerequisites
To run Grimoire you need to have installed:
- Python 3.10 or above
- Docker

#### Quick start
Edit settings.yaml file with your values:
```yaml
DEBUG: True # Enables debug logs
LOG_PROMPTS: True # Enables prompt logging
single_api_mode: False # Single API mode, change to True if you want to use main api for summarization.
context_percentage: 0.25 # How much of prompt context Grimoire entries will take
main_api:
  backend: GenericOAI # Accepted values: GenericOAI, Kobold, KoboldCPP, Aphrodite, Tabby
  url: http://127.0.0.1:5001 # url to your main api that will generate responses
  auth_key: 'your-api-authkey' # api key to your main api, leave empty or delete entry if there is none
side_api: # The whole section below is ignored if single api mode is set to True
  backend: GenericOAI # Accepted values: GenericOAI, Kobold, KoboldCPP, Aphrodite, Tabby
  url: http://127.0.0.1:5002 # Url to side api that will summarize entries
  auth_key: 'your-api-authkey' # Api key to side api, leave empty or delete entry if there is none
  input_sequence: '### Instruction:\n' # Instruct sequence for side api
  output_sequence: '\n### Response:\n' # Instruct sequence for side api

```

Depending on your system, run `Start.bat` for windows, or `Start.sh` on linux. These scripts will automatically:
- download and run docker image for redis and postgres
- create and enter python virtual environment
- install required dependencies
- run Grimoire

In the end script will open 2 terminal windows, both of them are required for Grimoire to run properly

### Usage
Following backends are supported:

- Aphrodite
- Tabby
- KoboldCPP

Currently, the only frontend that works with Grimoire is SillyTavern, however you have to install  [Grimoire extension](https://github.com/Krakenos/Grimoire-ST-Extension/) in order to make it work. To use Grimoire, set the settings to whatever your main api is (so for example: Text completion Aphrodite), and then set api url to `http://127.0.0.1:5005/`


Grimoire API starts by default on port 5005, and you interact with it pretty much the same way as you would with your main api(using the same endpoints, you can also view available endpoints at http://127.0.0.1:5005/docs. The only change that is required is including additional field in json called `grimoire` that includes id of the current chat, example json for generic OAI endpoints:

```json
{
  "prompt": "Example prompt",
  "max_tokens": 300,
  "truncation_length": 4096,
  "grimoire": {
    "chat_id": "id-of-current-chat"
  }
}
```

### Running from source
If instead of using `Start` scripts you want to run Grimoire manually, here is how you do it.

To install Redis through docker use the following command:
```bash
docker run -d --name redis -p 6379:6379 redis
```

Installing postgres in docker container:
```bash
docker run --name grimoire-postgres -p 5432:5432 -e POSTGRES_PASSWORD=secretpassword -e POSTGRES_USER=grimoire -e POSTGRES_DB=grimoire -d postgres
```

Installing Grimoire requirements:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```
For Windows, you also have to install:
```bash
pip install eventlet
```

Setup database:

```bash
alembic upgrade head
```

To start a process that will make summarization prompts use the following command:

Linux
```bash
celery -A grimoire.core.tasks worker --loglevel=INFO --concurrency=1
```
Windows
```bash
celery -A grimoire.core.tasks worker --loglevel=INFO --concurrency=1 -P eventlet
```
Note: -concurrency=1 refers to how many prompts will be directed to side api at the same time. Leave it at 1 unless you know the backend supports proper queueing on it's part.

And to run Grimoire API use:
```bash
python run.py
```
