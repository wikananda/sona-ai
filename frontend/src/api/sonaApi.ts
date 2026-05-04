export interface SpeakerSegment {
    speaker: string;
    text: string;
    start: number;
    end: number;
}

export type RecordingStatus = "pending" | "processing" | "done" | "failed";
export type TranscriptionModel = "parakeet" | "whisperx";
export type SummaryModel = "qwen" | "llama" | "gemma";

export interface Project {
    id: string;
    name: string;
    description?: string | null;
    created_at: string;
    updated_at: string;
    recordings?: Recording[];
}

export interface Transcript {
    id: string;
    recording_id: string;
    segments: SpeakerSegment[];
    language?: string | null;
    transcription_engine: string;
    diarization_engine?: string | null;
    model_config?: Record<string, unknown> | null;
    created_at: string;
    updated_at: string;
}

export interface Recording {
    id: string;
    project_id: string;
    original_name: string;
    stored_path: string;
    mime_type?: string | null;
    file_size_bytes?: number | null;
    language_hint?: string | null;
    model: TranscriptionModel;
    min_speakers?: number | null;
    max_speakers?: number | null;
    status: RecordingStatus;
    error?: string | null;
    created_at: string;
    updated_at: string;
    transcript?: Transcript | null;
}

export interface TranscribeParams {
    file: File;
    language?: string;
    model?: TranscriptionModel;
    minSpeakers?: number | "";
    maxSpeakers?: number | "";
}

export interface UploadProjectRecordingParams extends TranscribeParams {
    projectId: string;
}

export interface SummarizeParams {
    text: string;
    prompt?: string;
    maxLength?: number;
    model?: SummaryModel;
}

const BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export async function createProject(params: {
    name: string;
    description?: string;
}): Promise<Project> {
    return requestJson("/projects", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(params),
    });
}

export async function listProjects(): Promise<Project[]> {
    return requestJson("/projects");
}

export async function getProject(projectId: string): Promise<Project> {
    return requestJson(`/projects/${projectId}`);
}

export async function deleteProject(projectId: string): Promise<void> {
    await requestJson(`/projects/${projectId}`, { method: "DELETE" });
}

export async function uploadProjectRecording(
    params: UploadProjectRecordingParams,
): Promise<Recording> {
    const formData = buildRecordingFormData(params);
    return requestJson(`/projects/${params.projectId}/recordings`, {
        method: "POST",
        body: formData,
    });
}

export async function getRecording(recordingId: string): Promise<Recording> {
    return requestJson(`/recordings/${recordingId}`);
}

export async function deleteRecording(recordingId: string): Promise<void> {
    await requestJson(`/recordings/${recordingId}`, { method: "DELETE" });
}

export async function transcribeAudio(params: TranscribeParams): Promise<SpeakerSegment[]> {
    const formData = new FormData();
    formData.append("file", params.file);

    const url = new URL(`${BASE_URL}/transcribe`);
    appendSearchParam(url, "language", params.language);
    appendSearchParam(url, "model", params.model);
    appendSearchParam(url, "min_speakers", params.minSpeakers);
    appendSearchParam(url, "max_speakers", params.maxSpeakers);

    const response = await fetch(url.toString(), {
        method: "POST",
        body: formData,
    });

    if (!response.ok) {
        throw new Error(await errorMessage(response, "transcribe"));
    }

    const data = await response.json();
    return data.transcript;
}

export async function summarizeTranscript(params: SummarizeParams): Promise<string> {
    const data = await requestJson("/summarize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            text: params.text,
            prompt: params.prompt,
            max_length: params.maxLength,
            model: params.model ?? "qwen",
        }),
    });

    return data.summary;
}

function buildRecordingFormData(params: TranscribeParams): FormData {
    const formData = new FormData();
    formData.append("file", params.file);
    appendFormValue(formData, "language", params.language);
    appendFormValue(formData, "model", params.model ?? "parakeet");
    appendFormValue(formData, "min_speakers", params.minSpeakers);
    appendFormValue(formData, "max_speakers", params.maxSpeakers);
    return formData;
}

async function requestJson(path: string, init?: RequestInit) {
    const response = await fetch(`${BASE_URL}${path}`, init);
    if (!response.ok) {
        throw new Error(await errorMessage(response, path));
    }
    return response.json();
}

function appendFormValue(
    formData: FormData,
    key: string,
    value?: string | number | "",
) {
    if (value !== undefined && value !== "") {
        formData.append(key, String(value));
    }
}

function appendSearchParam(
    url: URL,
    key: string,
    value?: string | number | "",
) {
    if (value !== undefined && value !== "") {
        url.searchParams.append(key, String(value));
    }
}

async function errorMessage(response: Response, label: string): Promise<string> {
    try {
        const data = await response.json();
        return `${label} API error: ${data.detail ?? response.status}`;
    } catch {
        return `${label} API error: ${response.status}`;
    }
}
