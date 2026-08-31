# axio-repl

`axio-repl` is a terminal coding assistant that runs an Axio agent with file and
shell tools. It auto-detects your LLM backend from environment variables and
streams every token and tool call directly to the terminal.

## Install

```bash
uv tool install axio-repl
```

Add optional transports:

```bash
# Anthropic Claude
uv tool install axio-repl --with axio-transport-anthropic

# Google Gemini
uv tool install axio-repl --with axio-transport-google
```

Within the monorepo workspace:

```bash
uv run axio-repl
```

## Start the REPL

```bash
axio-repl
```

The REPL picks the first transport whose environment variable is set:

| Transport | Env Variable | Package |
|---|---|---|
| `google` | `GEMINI_API_KEY` | `axio-transport-google` |
| `google-vertex` | `GOOGLE_GENAI_USE_VERTEXAI` | `axio-transport-google` |
| `anthropic` | `ANTHROPIC_API_KEY` | `axio-transport-anthropic` |
| `openai` | `OPENAI_API_KEY` | `axio-transport-openai` |
| `nebius` | `NEBIUS_API_KEY` | `axio-transport-openai` |
| `openrouter` | `OPENROUTER_API_KEY` | `axio-transport-openai` |

Override with `--transport <name>`. `--model` is looked up in that transport's registry by
exact id, so an id it does not carry raises `KeyError` rather than being passed through:

```bash
axio-repl --transport anthropic --model claude-sonnet-4-6
axio-repl --transport google --model gemini-3.1-pro-preview
```

## Single-prompt mode

Pass a prompt as an argument for non-interactive use:

```bash
axio-repl "list the files in this project"
axio-repl --transport openai "write tests for src/auth.py"
```

## CLI flags

| Flag | Default | Description |
|---|---|---|
| `--transport` | auto | Transport name (see table above) |
| `--model` | transport default | Model name |
| `--temperature` | transport default | Sampling temperature |
| `--thinking` | off | Thinking level: `LOW`, `MEDIUM`, `HIGH`, or a token budget |
| `--max-tokens` | transport default | Max output tokens |
| `--max-iterations` | 30 | Max agent iterations |
| `--debug` | off | Log raw request/response bodies |

## REPL commands

| Command | Description |
|---|---|
| `/model` | Show current model and list available models |
| `/model <query>` | Switch to a model matching the query |
| `/thinking [level]` | Show or set thinking level/budget |
| `/temperature [val]` | Show or set sampling temperature |
| `/max-tokens [val]` | Show or set max output tokens |
| `/iterations [val]` | Show or set max agent iterations |
| `/debug [on\|off]` | Toggle request/response debug logging |
| `/help` | List all tools and commands |
| `/quit`, `/exit`, `/q` | Exit the REPL |

## Tools

| Tool | Description |
|---|---|
| `read_file` | Read file contents; images and videos are returned as vision blocks |
| `write_file` | Create or overwrite files |
| `patch_file` | Replace line ranges (1-indexed, inclusive) |
| `list_files` | List directory contents |
| `search_files` | Text or regex search across files |
| `shell` | Run shell commands with streaming stdout/stderr |
| `generate_image` | Generate images via Gemini (Google transport only) |
| `generate_video` | Generate videos via Veo (Google transport only) |

## AGENTS.md

Place an `AGENTS.md` file in the working directory to inject workspace-specific
instructions into the system prompt:

```markdown
# My project

- Always run `make test` after editing Python files
- The main entry point is `src/app.py`
- Use the `dev` branch for all changes
```

The file is loaded at startup and on every `/model` switch. It is optional -
if absent, the default system prompt is used unchanged.

## Capability-aware system prompt

The system prompt adapts to the selected model's capabilities. Switching models
with `/model` recalculates capabilities and rewrites the prompt automatically:

- **Vision** - `read_file` on images (PNG, JPG, GIF, WebP) returns pixel data.
  Screenshot-based UI review loops are unlocked.
- **Reasoning** - Extended thinking (chain-of-thought) is available.
- **Image generation** - Inline image generation via `generate_image`.
- **Video** - `read_file` on video files returns vision blocks.

## Multiline input

Paste multi-line text directly into the prompt. The REPL detects continuation
lines and joins them before sending:

```
You> Refactor this function:
...   def old(x):
...     return x+1
```

## Interrupting the agent

Press **Ctrl-C** to cancel the running agent loop. Partial tool output (stdout
already captured, files already written) is preserved in conversation context
so the model sees what happened and can resume cleanly.
