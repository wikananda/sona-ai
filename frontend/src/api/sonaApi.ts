export interface SpeakerSegment {
    speaker: string;
    text: string;
    start: number;
    end: number;
}

export type RecordingStatus = "pending" | "processing" | "done" | "failed";
export type TranscriptionModel =
    | "parakeet"
    | "faster-whisper-large-v3"
    | "faster-whisper-turbo";
export type SummaryMode = "local" | "byok";
export type BYOKProvider = "openai" | "groq" | "openrouter" | "custom";
export type LocalLLMModel = "qwen" | "llama" | "gemma";
export type RuntimeDevice = "auto" | "cpu" | "mps" | "cuda";
export type RecordingChatRole = "user" | "assistant";

export interface RuntimeDevices {
    default: RuntimeDevice;
    available: RuntimeDevice[];
    torch: {
        cuda: boolean;
        mps: boolean;
    };
}

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

export interface RecordingSummary {
    id: string;
    recording_id: string;
    text: string;
    mode: SummaryMode;
    model?: string | null;
    device?: RuntimeDevice | null;
    provider?: BYOKProvider | null;
    provider_model?: string | null;
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
    model: string;
    device: RuntimeDevice;
    min_speakers?: number | null;
    max_speakers?: number | null;
    status: RecordingStatus;
    error?: string | null;
    created_at: string;
    updated_at: string;
    transcript?: Transcript | null;
    summary?: RecordingSummary | null;
}

export interface RecordingChatMessage {
    role: RecordingChatRole;
    content: string;
}

export interface RecordingChatParams {
    question: string;
    history: RecordingChatMessage[];
    byok: BYOKSummarySettings;
}

export interface TranscribeParams {
    file: File;
    language?: string;
    model?: TranscriptionModel;
    device?: RuntimeDevice;
    minSpeakers?: number | "";
    maxSpeakers?: number | "";
}

export interface UploadProjectRecordingParams extends TranscribeParams {
    projectId: string;
}

export interface RetranscribeParams {
    language?: string;
    model: TranscriptionModel;
    device: RuntimeDevice;
    minSpeakers?: number | "";
    maxSpeakers?: number | "";
}

export interface BYOKSummarySettings {
    provider: BYOKProvider;
    apiKey: string;
    model: string;
    baseUrl?: string;
}

export interface SummarizeParams {
    text: string;
    prompt?: string;
    maxLength?: number;
    mode?: SummaryMode;
    model?: LocalLLMModel;
    device?: RuntimeDevice;
    byok?: BYOKSummarySettings;
}

export interface RecordingSummaryParams {
    prompt?: string;
    maxLength?: number;
    mode?: SummaryMode;
    model?: LocalLLMModel;
    device?: RuntimeDevice;
    byok?: BYOKSummarySettings;
}

const BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "http://localhost:8000";

export function recordingAudioUrl(recordingId: string): string {
    return `${BASE_URL}/recordings/${recordingId}/audio`;
}

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

export async function getRuntimeDevices(): Promise<RuntimeDevices> {
    return requestJson("/runtime/devices");
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

export async function renameRecording(
    recordingId: string,
    name: string,
): Promise<Recording> {
    return requestJson(`/recordings/${recordingId}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
    });
}

export async function retranscribeRecording(
    recordingId: string,
    params?: RetranscribeParams,
): Promise<Recording> {
    return requestJson(`/recordings/${recordingId}/retranscribe`, {
        method: "POST",
        headers: params ? { "Content-Type": "application/json" } : undefined,
        body: params
            ? JSON.stringify({
                language: params.language ?? "auto",
                model: params.model,
                device: params.device,
                min_speakers: emptyToNull(params.minSpeakers),
                max_speakers: emptyToNull(params.maxSpeakers),
            })
            : undefined,
    });
}

export async function chatWithRecording(
    recordingId: string,
    params: RecordingChatParams,
): Promise<string> {
    const data = await requestJson(`/recordings/${recordingId}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            question: params.question,
            history: params.history,
            byok_settings: {
                provider: params.byok.provider,
                api_key: params.byok.apiKey,
                model: params.byok.model,
                base_url: params.byok.baseUrl,
            },
        }),
    });

    return data.answer;
}

export async function renameTranscriptSpeakers(
    recordingId: string,
    speakers: Record<string, string>,
): Promise<Recording> {
    return requestJson(`/recordings/${recordingId}/transcript/speakers`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ speakers }),
    });
}

export async function transcribeAudio(params: TranscribeParams): Promise<SpeakerSegment[]> {
    const formData = new FormData();
    formData.append("file", params.file);

    const url = new URL(`${BASE_URL}/transcribe`);
    appendSearchParam(url, "language", params.language);
    appendSearchParam(url, "model", params.model);
    appendSearchParam(url, "device", params.device);
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
            mode: params.mode ?? "local",
            model: params.model ?? "qwen",
            device: params.device ?? "auto",
            byok: params.byok
                ? {
                    provider: params.byok.provider,
                    api_key: params.byok.apiKey,
                    model: params.byok.model,
                    base_url: params.byok.baseUrl,
                }
                : undefined,
        }),
    });

    return data.summary;
}

export async function summarizeRecording(
    recordingId: string,
    params: RecordingSummaryParams,
): Promise<Recording> {
    return requestJson(`/recordings/${recordingId}/summary`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            prompt: params.prompt,
            max_length: params.maxLength,
            mode: params.mode ?? "local",
            model: params.model ?? "qwen",
            device: params.device ?? "auto",
            byok: params.byok
                ? {
                    provider: params.byok.provider,
                    api_key: params.byok.apiKey,
                    model: params.byok.model,
                    base_url: params.byok.baseUrl,
                }
                : undefined,
        }),
    });
}

function buildRecordingFormData(params: TranscribeParams): FormData {
    const formData = new FormData();
    formData.append("file", params.file);
    appendFormValue(formData, "language", params.language);
    appendFormValue(formData, "model", params.model ?? "parakeet");
    appendFormValue(formData, "device", params.device ?? "auto");
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

function emptyToNull(value?: string | number | ""): number | null {
    if (value === undefined || value === "") return null;
    return Number(value);
}

async function errorMessage(response: Response, label: string): Promise<string> {
    try {
        const data = await response.json();
        return `${label} API error: ${formatApiErrorDetail(data.detail ?? response.status)}`;
    } catch {
        return `${label} API error: ${response.status}`;
    }
}

function formatApiErrorDetail(detail: unknown): string {
    if (typeof detail === "string") return detail;
    if (Array.isArray(detail)) {
        return detail
            .map((item) => {
                if (
                    item &&
                    typeof item === "object" &&
                    "msg" in item &&
                    typeof item.msg === "string"
                ) {
                    const location = "loc" in item && Array.isArray(item.loc)
                        ? item.loc.join(".")
                        : "";
                    return location ? `${location}: ${item.msg}` : item.msg;
                }
                return JSON.stringify(item);
            })
            .join("; ");
    }
    if (detail && typeof detail === "object") {
        return JSON.stringify(detail);
    }
    return String(detail);
}
