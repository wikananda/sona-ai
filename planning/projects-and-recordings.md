# Projects & Recordings UI Revamp

## Goal

Replace the current single-shot upload page with a project-scoped workflow: users create projects, upload one or more recordings into each project, and review the stored transcript for each recording.

This remains single-user and local-only for now.

## Verdict

This is a good next step. The existing speech pipeline, transcription service, and summarization route should stay intact. The revamp should add a persistence layer around them, then rebuild the frontend around projects and recordings.

The most important design choice: keep the backend recording upload endpoint focused on one file at a time, but let the frontend accept multiple files and submit them one by one. That keeps backend status/error handling simple while still supporting batch upload from the UI.

## Scope

Included:

- Persist projects, recordings, transcript segments, and pipeline metadata.
- Create, list, open, and delete projects.
- Upload one or more audio files into a project. One uploaded audio file equals one recording.
- Process recordings asynchronously so upload returns immediately.
- Show recording status: `pending`, `processing`, `done`, or `failed`.
- Show a project detail screen with a recordings sidebar and selected transcript view.
- Delete project data permanently with cascade behavior: project -> recordings -> transcripts -> audio files.
- Keep the legacy `POST /transcribe` route during the transition.

Deferred:

- Multi-user accounts and auth.
- Real queue system such as Celery/RQ.
- Re-transcribe existing recordings with different params.
- Auto-generated summaries per recording. Summaries stay on-demand for now.
- Search/RAG/vector database over transcripts.
- WebSocket/SSE status updates. Polling is enough locally.

## Product Flow

### Home: `/`

Purpose: project list and project creation.

Expected UI:

- Compact app shell, not a landing page.
- Project list ordered by newest first.
- Inline "New project" form with name and optional description.
- Delete action per project with confirmation.
- Empty state that lets the user create the first project.

### Project Detail: `/projects/[id]`

Purpose: manage recordings inside one project.

Recommended layout:

```text
project header
upload controls

left sidebar: recordings list + status badges
main panel: selected recording transcript
```

The sidebar should show:

- Original filename.
- Created time.
- Status badge.
- Short error message if failed.
- Active/selected state.

The main transcript panel should show:

- Selected recording metadata.
- Transcript segments when `done`.
- Progress/waiting state when `pending` or `processing`.
- Failure state when `failed`.
- Optional summarize action using the existing `/summarize` route. Do not auto-generate summaries yet.

### Upload Controls

The UI should expose only the options a user needs to choose:

- Language: `auto`, `en`, `id`, etc.
- Transcription model: e.g. `whisperx` or `parakeet`. Default to `parakeet`.
- Speaker count controls: keep `min_speakers` and `max_speakers` visible.

The frontend should not expose low-level config choices such as device, compute type, align model, diarization model, cache path, or batch size. It should send the selected `model` and `language`; the backend maps that to the correct config and pipeline behavior.

## Data Model

Use SQLite through SQLAlchemy. Store the DB at:

```text
backend/data/sona.db
```

Add this to `.gitignore`:

```text
backend/data/*
data/projects/*
```

### `projects`

| col         | type       | notes       |
|-------------|------------|-------------|
| id          | str (uuid) | PK          |
| name        | str        | required    |
| description | str?       | optional    |
| created_at  | datetime   | default now |
| updated_at  | datetime   | on update   |

### `recordings`

| col             | type       | notes                                           |
|-----------------|------------|-------------------------------------------------|
| id              | str (uuid) | PK                                              |
| project_id      | str        | FK -> projects.id, ON DELETE CASCADE           |
| original_name   | str        | uploaded filename                               |
| stored_path     | str        | relative to `PROJECT_ROOT`                      |
| mime_type       | str?       | from upload                                     |
| file_size_bytes | int?       | useful for UI/debugging                         |
| language_hint   | str?       | request-time language hint                      |
| model           | str        | request-time transcription model, e.g. `parakeet` |
| min_speakers    | int?       | request-time hint                               |
| max_speakers    | int?       | request-time hint                               |
| status          | enum       | `pending`, `processing`, `done`, `failed`       |
| error           | str?       | populated on failure                            |
| created_at      | datetime   | default now                                     |
| updated_at      | datetime   | on update                                       |

### `transcripts`

| col                  | type       | notes                                                   |
|----------------------|------------|---------------------------------------------------------|
| id                   | str (uuid) | PK                                                      |
| recording_id         | str        | FK -> recordings.id, ON DELETE CASCADE, unique          |
| segments_json        | text       | JSON-encoded sanitized `SpeakerSegment[]`               |
| language             | str?       | detected or resolved language                           |
| transcription_engine | str        | e.g. `whisperx`, `parakeet`                             |
| diarization_engine   | str?       | e.g. `pyannote`                                         |
| model_config_json    | text?      | JSON metadata for engine/model/params used              |
| created_at           | datetime   | default now                                             |
| updated_at           | datetime   | on update                                               |

Keep transcript rows separate from recordings so future re-transcription can replace transcript output without changing recording metadata.

## Storage Layout

```text
backend/data/
  sona.db

data/projects/
  <project_id>/
    <recording_id>.mp3
    <recording_id>.wav
    ...
```

Audio files should live under repo-root `data/projects/` because they are user data. DB lives under `backend/data/` because it is backend implementation state.

All paths should be resolved through `sona_ai.core.paths.PROJECT_ROOT`.

## Backend Plan

### New module: `sona_ai/db/`

- `engine.py`
  - Build SQLAlchemy engine.
  - Define `SessionLocal`.
  - Define declarative `Base`.
  - Enable SQLite foreign keys with `PRAGMA foreign_keys=ON`.
- `models.py`
  - `Project`
  - `Recording`
  - `Transcript`
  - Status enum/constants.
- `session.py`
  - `get_db()` FastAPI dependency.

For now, call `Base.metadata.create_all(engine)` from FastAPI startup. Alembic can wait until the schema becomes less experimental.

### New module: `sona_ai/storage/`

- `audio.py`
  - `save_upload(project_id, recording_id, upload_file) -> SavedAudio`
  - `delete_recording_file(stored_path)`
  - `delete_project_dir(project_id)`
  - Safe path resolution under `PROJECT_ROOT`.

`SavedAudio` should include at least:

- `stored_path`
- `mime_type`
- `file_size_bytes`

### New routes

Project routes:

- `POST /projects`
  - Body: `{name, description?}`
  - Returns created project.
- `GET /projects`
  - Returns project list ordered by `created_at desc`.
- `GET /projects/{project_id}`
  - Returns project plus recordings, but no transcript segments.
- `DELETE /projects/{project_id}`
  - Deletes DB rows, then removes `data/projects/<project_id>/`.

Recording routes:

- `POST /projects/{project_id}/recordings`
  - Multipart upload with one `file`.
  - Form fields: `language?`, `model?`, `min_speakers?`, `max_speakers?`.
  - `model` is a high-level transcription engine choice, such as `whisperx` or `parakeet`.
  - Backend resolves model-specific config internally.
  - Saves audio, creates recording with `pending`, schedules background task, returns immediately.
- `GET /recordings/{recording_id}`
  - Returns recording details and transcript segments when available.
- `DELETE /recordings/{recording_id}`
  - Deletes transcript row, recording row, and audio file.

Keep existing routes:

- `POST /transcribe`
  - Keep temporarily for compatibility.
  - Add a comment that the project-scoped recording route is the new path.
- `POST /summarize`
  - Keep unchanged for now.

### Background Worker

Add:

```text
sona_ai/services/recording_worker.py
```

Responsibilities:

- Open its own DB session. Do not reuse the request session.
- Set recording `status=processing`.
- Resolve the requested recording `model` to backend pipeline config.
- Call `TranscriptionService.transcribe(...)`.
- Sanitize result with `sanitize_for_json`.
- Persist `Transcript`.
- Set recording `status=done`.
- On exception, set `status=failed` and store a readable error.

Important constraint:

The speech pipeline is model-heavy and not safe for concurrent transcription calls. Put a `threading.Lock` inside `TranscriptionService.transcribe()` so both the old `/transcribe` route and the new background worker share the same protection.

Model-selection constraint:

The UI can send a high-level `model` value, but the backend should remain the source of truth for full model config. Recommended API values:

- `whisperx`
- `parakeet`

The backend should map those values to known configs, for example:

- `whisperx` -> WhisperX transcription + optional wav2vec alignment + Pyannote diarization.
- `parakeet` -> Parakeet transcription + Pyannote diarization + overlap-based speaker assignment.

Do not accept arbitrary config paths or model names from the frontend in this first version.

### Startup Recovery

On FastAPI startup:

- Create DB tables.
- Mark any stale `processing` recordings as `failed`.
- Error message: `Interrupted by server restart`.

This is simpler than re-enqueueing jobs and is fine for local single-user usage.

## Frontend Plan

Current frontend is a single Next.js page with upload/result state. The revamp should move toward an app shell.

### API Client

Update `frontend/src/api/sonaApi.ts` with typed clients:

- `createProject`
- `listProjects`
- `getProject`
- `deleteProject`
- `uploadProjectRecording`
- `getRecording`
- `deleteRecording`
- keep `summarizeTranscript`
- keep `transcribeAudio` only during migration

### Components

Reuse where practical:

- `AudioUploader`
- `TranscriptPanel`
- `SummaryPanel`

New components:

- `ProjectList`
- `NewProjectForm`
- `ProjectRow` or `ProjectCard`
- `ProjectHeader`
- `RecordingUploader`
- `ModelSelector`
- `RecordingSidebar`
- `RecordingListItem`
- `RecordingDetail`
- `RecordingStatusBadge`

Recommended adjustment:

`AudioUploader` is currently tightly coupled to the single-shot transcription flow. Either rename/refactor it into `RecordingUploader`, or make it accept a mode that changes the submit label and allows `multiple` files.

### Batch Upload Behavior

Frontend should accept multiple files:

```tsx
<input type="file" accept="audio/*" multiple />
```

Then submit each file to:

```text
POST /projects/{project_id}/recordings
```

Each file gets its own recording row and status. This avoids one failed file blocking the whole batch.

Each uploaded file should use the same selected language/model settings from the uploader. Later we can add per-file settings if needed.

### Polling

On project detail:

- Poll every 3 seconds while any recording is `pending` or `processing`.
- Use a single interval.
- Refresh either the full project or each active recording.
- Stop polling when all recordings are terminal: `done` or `failed`.

For a small local app, polling is simpler than WebSocket/SSE.

## Suggested Order Of Work

1. Schema and storage
   - Add SQLAlchemy dependency.
   - Add `db/` and `storage/` modules.
   - Wire table creation into startup.
   - Update `.gitignore`.

2. Project routes
   - Implement create/list/get/delete.
   - Test with curl or a small script.

3. Recording upload and worker
   - Implement save upload.
   - Create recording row.
   - Schedule background task.
   - Implement status transitions.
   - Add transcription lock in `TranscriptionService`.

4. Recording read/delete routes
   - Fetch transcript by recording ID.
   - Delete recording and audio file.

5. Frontend API client
   - Add typed project/recording calls.

6. Project list page
   - Replace current `/` with project list/create/delete.

7. Project detail page
   - Add `/projects/[id]`.
   - Add uploader, recordings sidebar, selected transcript panel, and polling.

8. Cleanup
   - Keep old `/transcribe` route until the project flow is stable.
   - Later remove old single-page state and `transcribeAudio`.

## Open Risks

- Pipeline concurrency can cause model-state issues or memory spikes. Use a single process-local lock.
- FastAPI `BackgroundTasks` is not durable. Server restart interrupts work. Mark interrupted jobs failed at startup.
- File and DB writes can drift if one succeeds and the other fails. Use try/except cleanup around upload creation.
- Large transcript JSON in SQLite is fine for now, but later search/RAG should move transcript text into a retrieval/indexing layer.
- Project deletion should remove audio files only after DB delete succeeds, or should recover cleanly if filesystem delete fails.

## Confirmations Before Building

Confirmed:

- One uploaded audio file equals one recording.
- Project and recording deletion permanently deletes audio and transcript data. No archive for now.
- Summaries stay on-demand. Do not auto-generate or store summaries yet.
- UI should expose language and high-level transcription model. Backend owns all remaining model/pipeline config.

Still to decide during implementation:

- Exact visual placement of the model selector and speaker controls in the upload form.

## Recommendation

Build the project/recording workflow first without stored summaries or re-transcription. The first implementation should optimize for a clean data model, reliable status transitions, and a usable project detail screen. Once that is stable, add re-transcription and RAG/search on top of persisted transcripts.
