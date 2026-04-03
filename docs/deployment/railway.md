# Deploying RD-Agent to Railway with Supabase

This repository now includes a Railway-ready `nixpacks.toml` and `railway.json` for deploying the Flask-backed Web UI:

- backend entrypoint: `rdagent server_ui`
- frontend build: `web/` via `npm run build:flask`
- persistent artifacts: optional Supabase-backed upload/trace/stdout sync

## What this deployment supports today

This deployment path runs the existing `server_ui` service on Railway.

- The Vue frontend is built into `git_ignore_folder/static`
- Flask serves the built frontend and real-time APIs
- RD-Agent jobs still run as subprocesses launched by the Flask service
- If Supabase persistence is enabled, uploaded files, trace artifacts, and stdout logs are mirrored to Supabase Storage and can be hydrated back after container replacement

This is the shortest path to getting the current project online on Railway without rewriting the execution model.

## 1. Create the Railway service

1. Create a new Railway project from this repository.
2. Keep the service root at the repository root.
3. Railway will pick up `nixpacks.toml` automatically.

The service starts with:

```bash
rdagent server_ui --port $PORT
```

## 2. Create Supabase resources

Create one Supabase project and at least one Storage bucket for RD-Agent artifacts.

Recommended setup:

- **Postgres**: optional for future task metadata / queue state
- **Storage bucket**: required for persisted artifacts

Suggested bucket name:

```text
rdagent
```

## 3. Configure Railway environment variables

### Required for the web service

```bash
PORT                         # Provided by Railway
UI_STATIC_PATH=./git_ignore_folder/static
UI_TRACE_FOLDER=./git_ignore_folder/traces
```

### Enable Supabase-backed artifact persistence

```bash
UI_SUPABASE_ENABLED=true
UI_SUPABASE_URL=https://<your-project-ref>.supabase.co
UI_SUPABASE_SERVICE_ROLE_KEY=<supabase-service-role-key>
UI_SUPABASE_BUCKET=rdagent
```

Optional path prefixes inside the bucket:

```bash
UI_SUPABASE_TRACE_PREFIX=traces
UI_SUPABASE_STDOUT_PREFIX=stdout
UI_SUPABASE_UPLOAD_PREFIX=uploads
```

### LLM / provider configuration

Set the same provider credentials you already use locally. Common examples:

```bash
LLM_BACKEND=rdagent.oai.backend.LiteLLMAPIBackend
OPENAI_API_KEY=<your-openai-key>
OPENAI_API_BASE=<optional-base-url>
CHAT_MODEL=<your-model-name>
EMBEDDING_MODEL=<your-embedding-model>
```

If you use Azure or other providers, set the corresponding `LLM_SETTINGS` environment variables used by this repo.

## 4. Deploy

Once the environment variables are set, trigger a deploy in Railway.

The build pipeline will:

1. install Python dependencies
2. install frontend dependencies in `web/`
3. build the Vue frontend into `git_ignore_folder/static`
4. start `rdagent server_ui`

## 5. Validate the deployment

After Railway finishes deploying:

1. open the Railway public URL
2. confirm the frontend loads
3. start a task from the UI
4. confirm traces stream normally
5. after the task produces output, verify that artifacts appear in Supabase Storage under:
   - `uploads/...`
   - `traces/...`
   - `stdout/...`

## Current architecture notes

- The current codebase still executes RD-Agent jobs as subprocesses inside the same `server_ui` service.
- Supabase persistence now protects uploaded files, trace logs, and stdout artifacts against container replacement.
- A fully split Web/API + Worker deployment would require an additional queue / orchestration layer; that is not wired in this repository yet.
