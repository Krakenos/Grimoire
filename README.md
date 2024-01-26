# Memoir

Memoir is a server that implements long term memory for AI chatbots through processing prompts and generating description of concepts such as locations, people, items. It's main purpose is for conversational AI chatbots. The server is meant to be slotted between frontend that interacts with LLM(such as SillyTavern) and backend that host the LLM(Aphrodite, KoboldAI etc).

### How it works
Memoir collects the prompts that are meant to be sent to LLM and analyzes them with Natural Language Processing (NLP), it sends collected prompts to generate descriptions of concepts found in messages, and in the end, it inserts them into current context. It's essentially Retrieval Augmented Generation (RAG) system that collects and generates it's database throughout the conversation with AI. It requires 2 language models to run. One for the actual chatbot, and other for summarization tasks.

### IMPORTANT NOTE
Memoir is still VERY EARLY in development. There is bound to be a lot of bugs, there will be breaking changes in codebase and it's not stable. You will most likely lose the data that Memoir has collected along the way. It's a prototype that's not production ready.
### Setup
#### Running from source

To run from source following is required:
- Python >3.10
- RabbitMQ Server (Either installed natively or in docker)

To install RabbitMQ Server through docker use the following command:
```bash
docker run -d -p 15672:15672 -p 5672:5672 --hostname my-rabbitmq --name my-rabbitmq-container rabbitmq:3-management
```

Installing Memoir requirements:
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_trf
```
For Windows, you also have to install:
```bash
pip install eventlet
```

Copy .env_template file and name it .env. Fill the following variables
```
MAIN_API_URL = 'http://127.0.0.1:5001'          # Url of main api used to generate bot responses
MAIN_API_AUTH = 'MyApiKey'                      # API key to main api, remove entry if there is none
SIDE_API_URL = 'http://127.0.0.1:5002'          # Url of side api used to generate descriptions of items etc
CONTEXT_PERCENTAGE = 0.25                       # Maximum amount of tokens that Memoir database entries will take, when filling the context
DB_ENGINE = 'sqlite:///db.sqlite3'              # Database url, defaults to sqlite file
CELERY_BROKER_URL = 'amqp://guest@localhost//'  # Url to message broker, in our case RabbitMQ server, leave it as it is if you installed through docker
```

Setup database:

```bash
alembic upgrade head
```

### Usage
Following backends are supported:

Main API:
- Aphrodite
- KoboldCPP

Side API:
- KoboldCPP

Following instruct presets for LLM's are supported:

Main API:
- Alpaca

Side API:
- Mistral

To start a process that will direct summarization prompts to side api use the following command:

Linux
```bash
celery -A memoir.core.tasks worker --loglevel=INFO --concurrency=1
```
Windows
```bash
celery -A memoir.core.tasks worker --loglevel=INFO --concurrency=1 -P eventlet
```
Note: -concurrency=1 refers to how many prompts will be directed to side api at the same time. Leave it at 1 unless you know the backend supports proper queueing on it's part.

And to run Memoir API use:
```bash
python run.py
```

Memoir API starts by default on port 5005, and you interact with it pretty much the same way as you would with your main api (using the same endpoints, you can also view available endpoints at `http://127.0.0.1/docs`). The only change that is required is including additional field in json called `memoir` that includes id of the current chat, example json for Aphrodite:

```json
{
  "prompt": "Example prompt",
  "max_tokens": 300,
  "truncation_length": 4096,
  "memoir": {
    "chat_id": "id-of-current-chat"
  }
}
```