export interface SpeakerSegment {
    speaker?: string;
    text: string;
    start: number;
    end: number;
    words?: {
        word: string;
        start?: number | null;
        end?: number | null;
        score?: number | null;
        speaker?: string | null;
    }[];
}

export type RecordingStatus = "pending" | "processing" | "done" | "failed" | "canceled";
export type TranscriptionModel =
    | "parakeet"
    | "faster-whisper-large-v3"
    | "faster-whisper-turbo";
export type SummaryMode = "local" | "byok";
export type BYOKProvider = "openai" | "groq" | "openrouter" | "custom";
export type LocalLLMModel = "qwen" | "llama" | "gemma";
export type RuntimeDevice = "auto" | "cpu" | "mps" | "cuda";
export type RecordingChatRole = "user" | "assistant";
export type RuntimeModelStatus = "missing" | "installed" | "running" | "failed";
export type ModelDownloadJobStatus = "queued" | "running" | "installed" | "failed";
export type RecordingProgressStage =
    | "queued"
    | "preparing"
    | "transcribing"
    | "aligning"
    | "diarizing"
    | "assigning_speakers"
    | "done"
    | "failed"
    | "canceled"
    | "processing";

export interface RuntimeDevices {
    default: RuntimeDevice;
    available: RuntimeDevice[];
    torch: {
        cuda: boolean;
        mps: boolean;
    };
}

export interface RuntimeModel {
    id: string;
    label: string;
    type: string;
    model_names: string[];
    config_name: string;
    environment: string;
    cache_subdir?: string | null;
    cache_path: string;
    requires_hf_token: boolean;
    hf_token_available: boolean;
    installed: boolean;
    status: RuntimeModelStatus;
    active_job_id?: string | null;
    error?: string | null;
}

export interface ModelDownloadJob {
    job_id: string;
    model_id: string;
    status: ModelDownloadJobStatus;
    message: string;
    started_at: string;
    finished_at?: string | null;
    error?: string | null;
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
    format_name?: string | null;
    plan?: Record<string, unknown> | null;
    created_at: string;
    updated_at: string;
}

export interface RecordingProgress {
    stage: RecordingProgressStage | string;
    label: string;
    completed_steps: number;
    total_steps: number;
    percent: number;
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
    progress: RecordingProgress;
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
    extractSpeakers?: boolean;
}

export interface UploadProjectRecordingParams extends TranscribeParams {
    projectId: string;
}

export interface LiveTranscriptionChunkParams {
    projectId: string;
    file: File;
    chunkIndex: number;
    chunkStart: number;
    language?: string;
    model: TranscriptionModel;
    device: RuntimeDevice;
}

export interface LiveTranscriptionChunkResult {
    chunk_index: number;
    chunk_start: number;
    segments: SpeakerSegment[];
    language?: string | null;
}

export interface SaveLiveRecordingParams {
    projectId: string;
    file: File;
    segments: SpeakerSegment[];
    language?: string;
    model: TranscriptionModel;
    device: RuntimeDevice;
}

export interface RetranscribeParams {
    language?: string;
    model: TranscriptionModel;
    device: RuntimeDevice;
    minSpeakers?: number | "";
    maxSpeakers?: number | "";
    extractSpeakers?: boolean;
}

export interface SpeakerExtractionParams {
    minSpeakers?: number | "";
    maxSpeakers?: number | "";
}

export interface TranscriptSegmentUpdateParams {
    text: string;
    speaker?: string | null;
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

export interface RecordingSummaryUpdateParams {
    text: string;
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

export async function getRuntimeModels(): Promise<RuntimeModel[]> {
    return requestJson("/runtime/models");
}

export async function startRuntimeModelDownload(modelId: string): Promise<ModelDownloadJob> {
    return requestJson(`/runtime/models/${modelId}/download`, {
        method: "POST",
    });
}

export async function getRuntimeModelDownloadJob(jobId: string): Promise<ModelDownloadJob> {
    return requestJson(`/runtime/model-downloads/${jobId}`);
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

export async function transcribeLiveChunk(
    params: LiveTranscriptionChunkParams,
): Promise<LiveTranscriptionChunkResult> {
    const formData = new FormData();
    formData.append("file", params.file);
    appendFormValue(formData, "chunk_index", params.chunkIndex);
    appendFormValue(formData, "chunk_start", params.chunkStart);
    appendFormValue(formData, "language", params.language);
    appendFormValue(formData, "model", params.model);
    appendFormValue(formData, "device", params.device);

    return requestJson(`/projects/${params.projectId}/live-transcription/chunks`, {
        method: "POST",
        body: formData,
    });
}

export async function saveLiveRecording(
    params: SaveLiveRecordingParams,
): Promise<Recording> {
    const formData = new FormData();
    formData.append("file", params.file);
    formData.append("segments_json", JSON.stringify(params.segments));
    appendFormValue(formData, "language", params.language);
    appendFormValue(formData, "model", params.model);
    appendFormValue(formData, "device", params.device);

    return requestJson(`/projects/${params.projectId}/live-transcription/recordings`, {
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
                extract_speakers: params.extractSpeakers ?? true,
            })
            : undefined,
    });
}

export async function cancelRecording(recordingId: string): Promise<Recording> {
    return requestJson(`/recordings/${recordingId}/cancel`, {
        method: "POST",
    });
}

export async function extractRecordingSpeakers(
    recordingId: string,
    params?: SpeakerExtractionParams,
): Promise<Recording> {
    return requestJson(`/recordings/${recordingId}/speakers/extract`, {
        method: "POST",
        headers: params ? { "Content-Type": "application/json" } : undefined,
        body: params
            ? JSON.stringify({
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

export async function updateTranscriptSegment(
    recordingId: string,
    segmentIndex: number,
    params: TranscriptSegmentUpdateParams,
): Promise<Recording> {
    return requestJson(`/recordings/${recordingId}/transcript/segments/${segmentIndex}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            text: params.text,
            speaker: params.speaker,
        }),
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
    appendSearchParam(url, "extract_speakers", params.extractSpeakers);

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

export async function updateRecordingSummary(
    recordingId: string,
    params: RecordingSummaryUpdateParams,
): Promise<Recording> {
    return requestJson(`/recordings/${recordingId}/summary`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
            text: params.text,
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
    appendFormValue(formData, "extract_speakers", params.extractSpeakers ?? true);
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
    value?: string | number | boolean | "",
) {
    if (value !== undefined && value !== "") {
        formData.append(key, String(value));
    }
}

function appendSearchParam(
    url: URL,
    key: string,
    value?: string | number | boolean | "",
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
