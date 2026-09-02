import {
    apiWebSocketUrl,
    RuntimeDevice,
    SpeakerSegment,
    TranscriptionModel,
} from "@/src/api/sonaApi";

const READY_TIMEOUT_MS = 130000;
const FINAL_TIMEOUT_MS = 30000;
const MAX_BUFFERED_BYTES = 1024 * 1024;

interface LiveEventBase {
    type: string;
    version: number;
}

export interface LiveTranscriptEvent extends LiveEventBase {
    type: "transcript";
    session_id: string;
    revision: number;
    committed: SpeakerSegment[];
    provisional: SpeakerSegment | null;
    language?: string | null;
}

export interface LiveFinalEvent extends LiveEventBase {
    type: "final";
    session_id: string;
    revision: number;
    committed: SpeakerSegment[];
    provisional: null;
    segments: SpeakerSegment[];
    language?: string | null;
}

interface LiveReadyEvent extends LiveEventBase {
    type: "ready";
    session_id: string;
    engine: string;
    model: string;
    sample_rate: number;
    format: string;
}

interface LiveErrorEvent extends LiveEventBase {
    type: "error";
    code: string;
    message: string;
    recoverable: boolean;
}

type ServerEvent = LiveTranscriptEvent | LiveFinalEvent | LiveReadyEvent | LiveErrorEvent;

export interface LiveTranscriptionSocket {
    sendAudio: (frame: ArrayBuffer) => boolean;
    finish: () => Promise<LiveFinalEvent>;
    abort: () => void;
}

export async function connectLiveTranscriptionSocket(params: {
    projectId: string;
    model: TranscriptionModel;
    device: RuntimeDevice;
    language?: string;
    onTranscript: (event: LiveTranscriptEvent) => void;
    onError: (message: string) => void;
}): Promise<LiveTranscriptionSocket> {
    const socket = new WebSocket(apiWebSocketUrl(
        `/projects/${encodeURIComponent(params.projectId)}/live-transcription/ws`,
    ));
    socket.binaryType = "arraybuffer";
    let latestRevision = -1;
    let ready = false;
    let intentionalClose = false;
    let finalEvent: LiveFinalEvent | null = null;
    let resolveFinal: ((event: LiveFinalEvent) => void) | null = null;
    let rejectFinal: ((error: Error) => void) | null = null;
    let finalTimeout: number | null = null;

    const readyEvent = await new Promise<LiveReadyEvent>((resolve, reject) => {
        const timeout = window.setTimeout(() => {
            socket.close();
            reject(new Error("Timed out while preparing realtime Whisper."));
        }, READY_TIMEOUT_MS);

        socket.addEventListener("open", () => {
            socket.send(JSON.stringify({
                type: "start",
                version: 1,
                model: params.model,
                device: params.device,
                language: params.language ?? "auto",
                audio: {
                    encoding: "pcm_s16le",
                    sample_rate: 16000,
                    channels: 1,
                },
                word_timestamps: true,
            }));
        });
        socket.addEventListener("message", handleMessage);
        socket.addEventListener("error", () => {
            if (!ready) reject(new Error("Could not connect to realtime Whisper."));
        });
        socket.addEventListener("close", () => {
            if (!ready) reject(new Error("Realtime Whisper closed before it was ready."));
        });

        function handleMessage(event: MessageEvent) {
            const message = parseServerEvent(event.data);
            if (message?.type === "ready") {
                window.clearTimeout(timeout);
                ready = true;
                resolve(message);
            } else if (message?.type === "error") {
                window.clearTimeout(timeout);
                reject(new Error(message.message));
            }
        }
    });
    void readyEvent;

    socket.addEventListener("message", (event) => {
        const message = parseServerEvent(event.data);
        if (!message) return;
        if (message.type === "transcript") {
            if (message.revision <= latestRevision) return;
            latestRevision = message.revision;
            params.onTranscript(message);
        } else if (message.type === "final") {
            if (message.revision <= latestRevision && finalEvent) return;
            latestRevision = Math.max(latestRevision, message.revision);
            finalEvent = message;
            if (finalTimeout !== null) window.clearTimeout(finalTimeout);
            resolveFinal?.(message);
        } else if (message.type === "error") {
            const error = new Error(message.message);
            params.onError(message.message);
            rejectFinal?.(error);
        }
    });
    socket.addEventListener("close", () => {
        if (!intentionalClose && !finalEvent) {
            const error = new Error("Realtime Whisper disconnected. Recording is continuing.");
            params.onError(error.message);
            rejectFinal?.(error);
        }
    });
    socket.addEventListener("error", () => {
        if (ready && !intentionalClose) {
            params.onError("Realtime Whisper connection failed. Recording is continuing.");
        }
    });

    return {
        sendAudio(frame) {
            if (socket.readyState !== WebSocket.OPEN || intentionalClose) return false;
            if (socket.bufferedAmount > MAX_BUFFERED_BYTES) {
                params.onError("Realtime preview could not keep up. Recording is continuing.");
                intentionalClose = true;
                socket.close();
                return false;
            }
            socket.send(frame);
            return true;
        },
        finish() {
            if (finalEvent) return Promise.resolve(finalEvent);
            if (socket.readyState !== WebSocket.OPEN) {
                return Promise.reject(new Error("Realtime Whisper is no longer connected."));
            }
            return new Promise<LiveFinalEvent>((resolve, reject) => {
                resolveFinal = (event) => {
                    intentionalClose = true;
                    socket.close();
                    resolve(event);
                };
                rejectFinal = reject;
                socket.send(JSON.stringify({ type: "stop" }));
                finalTimeout = window.setTimeout(() => {
                    intentionalClose = true;
                    socket.close();
                    reject(new Error("Timed out finalizing the realtime transcript."));
                }, FINAL_TIMEOUT_MS);
            });
        },
        abort() {
            intentionalClose = true;
            socket.close();
        },
    };
}

function parseServerEvent(data: unknown): ServerEvent | null {
    if (typeof data !== "string") return null;
    try {
        const event = JSON.parse(data) as ServerEvent;
        return event && typeof event === "object" && typeof event.type === "string"
            ? event
            : null;
    } catch {
        return null;
    }
}
