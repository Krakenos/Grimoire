# Grimoire

Grimoire is a server that implements long term memory for AI chatbots through processing messages and generating description of concepts such as locations, people, items. It's main purpose is for conversational AI chatbots.

### How it works
Grimoire collects the messages that are meant to be sent to LLM and analyzes them with Natural Language Processing (NLP), it sends collected messages to LLM in order to generate descriptions of concepts found in messages, and then the entries can be retrieved via API. It's essentially Retrieval Augmented Generation (RAG) system that collects and generates it's database throughout the conversation with AI.

### Setup
#### Prerequisites
To run Grimoire you need to have installed:
- [uv](https://docs.astral.sh/uv/) (manages Python and dependencies)
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

#### Chat completions and reasoning/thinking models

By default Grimoire talks to the summarization backend with plain text completions
(`/v1/completions`, or Kobold's native `/api/v1/generate`). Set `api_mode: chat` to instead use
chat completions (`/v1/chat/completions`), which most modern OpenAI-compatible servers (vLLM,
Aphrodite, Tabby, GenericOAI) support alongside text completions. KoboldAI/KoboldCPP have no chat
endpoint, so they always use text completions regardless of `api_mode`.

```yaml
summarization_api:
  backend: GenericOAI
  api_mode: chat # "text" (default) or "chat"
  reasoning_effort: "" # OpenAI/vLLM reasoning effort knob, e.g. "low"/"medium"/"high"; empty omits it
  chat_template_kwargs: {} # extra chat kwargs, e.g. {enable_thinking: false} for Qwen/vLLM/Aphrodite
  thinking_budget: 0 # extra tokens added on top of summarization.max_tokens to leave room for reasoning
  strip_reasoning: true # strip <think>...</think> (or an orphaned closing tag) from generated text

summarization:
  # Used only in chat mode, in place of `prompt` / `segmented_memory_prompt`. Same placeholders,
  # but no instruct sequences - the server applies its own chat template.
  chat_system_prompt: "{previous_summary}{additional_info}{messages}"
  chat_user_prompt: "Describe {term}."
  segmented_memory_chat_system_prompt: "Below is conversation snippet.\n{messages}"
  segmented_memory_chat_user_prompt: "Summarize the most important facts and events in the story so far. Limit the summary to one paragraph. Your response should include nothing but the summary."
```

For reasoning models that emit `<think>...</think>` blocks, Grimoire captures that separately
(from a server's `reasoning_content`/`reasoning` field in chat mode, or by parsing it out of the
generated text otherwise) and never stores it in the summary - it's only logged for debugging.
Set `strip_reasoning: false` to disable this and keep the raw output as-is.

### Running from source

Install dependencies (creates `.venv` automatically):
```bash
uv sync
```

If you have a CUDA GPU and want GPU acceleration:
```bash
uv sync --extra cuda
```

Run containers for Grimoire dependencies (redis and postgres):
```bash
docker compose -f docker/docker-compose-dev.yaml up -d
```

Download the spacy model:
```bash
uv run python -m spacy download en_core_web_trf
```

Setup database:
```bash
uv run alembic upgrade head
```

To start a process that will make summarization prompts use the following command:

```bash
uv run celery -A grimoire.core.tasks worker --loglevel=INFO --concurrency=1 -Q summarization_queue --pool=threads
```
Note: -concurrency=1 refers to how many prompts will be directed to side api at the same time. Leave it at 1 unless you know the backend supports proper queueing or batching.

And to run Grimoire API use:
```bash
uv run python run.py
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

### Management panel

Grimoire ships an optional web panel at http://127.0.0.1:5005/panel for inspecting and editing what
it has stored: users and their chats, message history, knowledge entries and segmented memories
(view, edit, delete), a memory graph, and the summarization settings.

It is off by default. To enable it:

```yaml
enable_management_panel: True
```

**The panel is a local administration tool, and the correct production deployment is to leave it
off.** It has no per-user access control by design — anyone who reaches it reads and edits *every*
user's chats, and can repoint the summarization backend at another server. That is fine for a
single-operator instance on your own machine, and unacceptable on a host that serves other people.
Turning it off is the control; a key is not a substitute for it.

If the panel is enabled on a host that anyone else can reach, set a key:

```yaml
PANEL_KEY: "a-secret-only-you-have"
```

The panel prompts for it in the browser on the first request and remembers it. `PANEL_KEY` must not
be the same value as `AUTH_KEY` — `AUTH_KEY` is the key you give to chat clients, so reusing it
would make every client key a key over every user's data. Grimoire refuses to start if they match,
and logs a warning when the panel is enabled with no key at all.

With the panel disabled its endpoints do not exist at all — the sub-application is never mounted, so
`/panel` and everything under it returns 404 rather than being merely access-controlled.

#### Settings edited in the panel outlive it

Settings changed through the panel are saved to the `setting_override` database table and layered
over `settings.yaml` every time they are read, including by the Celery worker. **Disabling the panel
stops further edits but does not revert edits already made.** If you change the summarization backend
in the panel and later set `enable_management_panel: False`, the worker keeps using the overridden
value and your `settings.yaml` will appear to be ignored.

Only `summarization_api`, `summarization` and `tokenization` can be overridden this way. To go back
to what is in the file, either use the panel's reset control while it is still enabled, or clear the
rows directly:

```bash
docker compose -f docker/docker-compose-dev.yaml exec postgres \
  psql -U grimoire -d grimoire -c "DELETE FROM setting_override;"
```

Overrides are read from the database on each use rather than cached at startup, so this takes effect
on the next summarization with no restart needed.
