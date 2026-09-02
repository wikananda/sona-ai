"use client";

import { ChangeEvent, useEffect, useRef, useState } from "react";
import {
    Recording,
    RuntimeDevice,
    RuntimeDevices,
    SpeakerSegment,
    TranscriptionModel,
    saveLiveRecording,
    transcribeLiveChunk,
    uploadProjectRecording,
} from "@/src/api/sonaApi";
import {
    connectLiveTranscriptionSocket,
    LiveTranscriptionSocket,
    LiveTranscriptEvent,
} from "@/src/api/liveTranscriptionSocket";
import {
    LivePcmCapture,
    startLivePcmCapture,
} from "@/src/audio/livePcmCapture";
import {
    deviceLabel,
    TRANSCRIPTION_LANGUAGES,
    TRANSCRIPTION_MODELS,
} from "@/src/utils/transcriptionSettings";

type LiveState =
    | "idle"
    | "requesting"
    | "recording"
    | "stopping"
    | "saving"
    | "save-error";
type LiveTransport = "legacy" | "whisper-stream";

interface Props {
    projectId: string;
    runtimeDevices: RuntimeDevices;
    onBeforeStart: (params: {
        model: TranscriptionModel;
        device: RuntimeDevice;
        language?: string;
    }) => Promise<boolean>;
    onSaved: (recording: Recording) => void;
    onActiveChange?: (active: boolean) => void;
}

const LEGACY_CHUNK_MS = 8000;
const MIME_TYPES = [
    "audio/webm;codecs=opus",
    "audio/webm",
    "audio/mp4",
    "audio/ogg;codecs=opus",
];
const AUDIO_BITS_PER_SECOND = 256000;
const AUDIO_CAPTURE_CONSTRAINTS: MediaTrackConstraints = {
    echoCancellation: false,
    noiseSuppression: false,
    autoGainControl: false,
    sampleRate: { ideal: 48000 },
    channelCount: { ideal: 2 },
};

export default function LiveTranscriptionPanel({
    projectId,
    runtimeDevices,
    onBeforeStart,
    onSaved,
    onActiveChange,
}: Props) {
    const mediaRecorderRef = useRef<MediaRecorder | null>(null);
    const streamRef = useRef<MediaStream | null>(null);
    const recordingAudioContextRef = useRef<AudioContext | null>(null);
    const pcmCaptureRef = useRef<LivePcmCapture | null>(null);
    const liveSocketRef = useRef<LiveTranscriptionSocket | null>(null);
    const transportRef = useRef<LiveTransport>("legacy");
    const chunksRef = useRef<Blob[]>([]);
    const chunkIndexRef = useRef(0);
    const startedAtRef = useRef(0);
    const processingQueueRef = useRef<Promise<void>>(Promise.resolve());
    const finalMimeTypeRef = useRef("audio/webm");
    const segmentsRef = useRef<SpeakerSegment[]>([]);
    const provisionalRef = useRef<SpeakerSegment | null>(null);

    const [includeMicrophone, setIncludeMicrophone] = useState(true);
    const [includeSystemAudio, setIncludeSystemAudio] = useState(true);
    const [state, setState] = useState<LiveState>("idle");
    const [elapsedSeconds, setElapsedSeconds] = useState(0);
    const [language, setLanguage] = useState("auto");
    const [model, setModel] = useState<TranscriptionModel>("faster-whisper-turbo");
    const [device, setDevice] = useState<RuntimeDevice>(runtimeDevices.default);
    const [segments, setSegments] = useState<SpeakerSegment[]>([]);
    const [provisional, setProvisional] = useState<SpeakerSegment | null>(null);
    const [notice, setNotice] = useState("");
    const [error, setError] = useState("");

    const selectedDevice = runtimeDevices.available.includes(device)
        ? device
        : runtimeDevices.default;
    const isBusy = state === "requesting" || state === "stopping" || state === "saving";
    const isRecording = state === "recording";
    const isSetupMode = state === "idle" || state === "requesting";
    const hasNonEnglishLanguage = !["auto", "en"].includes(language);
    const isActive = state !== "idle";

    useEffect(() => {
        if (hasNonEnglishLanguage && model === "parakeet") {
            setModel("faster-whisper-turbo");
        }
    }, [hasNonEnglishLanguage, model]);

    useEffect(() => {
        onActiveChange?.(isActive);
    }, [isActive, onActiveChange]);

    useEffect(() => {
        if (!isRecording) return;
        const intervalId = window.setInterval(() => {
            setElapsedSeconds(Math.floor((Date.now() - startedAtRef.current) / 1000));
        }, 500);
        return () => window.clearInterval(intervalId);
    }, [isRecording]);

    useEffect(() => {
        return () => {
            liveSocketRef.current?.abort();
            void pcmCaptureRef.current?.abort();
            releaseCaptureStream();
            onActiveChange?.(false);
        };
    }, [onActiveChange]);

    const startLiveTranscription = async () => {
        if (!hasMediaRecorderSupport()) {
            setError("Audio recording is not supported in this browser.");
            return;
        }
        if (!includeMicrophone && !includeSystemAudio) {
            setError("Choose microphone, system audio, or both.");
            return;
        }

        prepareNewSession();
        setState("requesting");

        try {
            // Keep display capture inside the original click gesture.
            const capture = await createCaptureStream({
                includeMicrophone,
                includeSystemAudio,
            });
            streamRef.current = capture.cleanupStream;
            recordingAudioContextRef.current = capture.audioContext;

            const canStart = await onBeforeStart({
                model,
                device: selectedDevice,
                language,
            });
            if (!canStart) {
                releaseCaptureStream();
                setState("idle");
                return;
            }

            const mimeType = chooseMimeType();
            const recorder = new MediaRecorder(
                capture.recordingStream,
                mediaRecorderOptions(mimeType),
            );
            mediaRecorderRef.current = recorder;
            finalMimeTypeRef.current = recorder.mimeType || mimeType || "audio/webm";

            if (model !== "parakeet") {
                await prepareWhisperStreaming(capture.recordingStream);
            }

            recorder.addEventListener("dataavailable", (event) => {
                if (event.data.size <= 0) return;
                chunksRef.current.push(event.data);
                if (transportRef.current === "legacy") enqueueLegacyChunk();
            });
            recorder.addEventListener("stop", () => {
                void finalizeLiveRecording();
            }, { once: true });

            startedAtRef.current = Date.now();
            setElapsedSeconds(0);
            recorder.start(LEGACY_CHUNK_MS);
            setState("recording");
        } catch (err) {
            await cleanupRealtime();
            releaseCaptureStream();
            mediaRecorderRef.current = null;
            setState("idle");
            setError(errorMessage(err, includeMicrophone));
        }
    };

    const prepareWhisperStreaming = async (recordingStream: MediaStream) => {
        try {
            const socket = await connectLiveTranscriptionSocket({
                projectId,
                model,
                device: selectedDevice,
                language,
                onTranscript: applyStreamingTranscript,
                onError: activateLegacyFallback,
            });
            liveSocketRef.current = socket;
            pcmCaptureRef.current = await startLivePcmCapture(recordingStream, (frame) => {
                if (!liveSocketRef.current?.sendAudio(frame)) {
                    activateLegacyFallback(
                        "Realtime preview stopped, so Sona switched to compatibility mode.",
                    );
                }
            });
            transportRef.current = "whisper-stream";
            setNotice("Realtime Whisper connected. Speech will appear as it is committed.");
        } catch (err) {
            await cleanupRealtime();
            transportRef.current = "legacy";
            const message = err instanceof Error ? err.message : "Realtime Whisper is unavailable.";
            setNotice(`${message} Using compatibility transcription while recording continues.`);
        }
    };

    const applyStreamingTranscript = (event: LiveTranscriptEvent) => {
        if (event.committed.length > 0) {
            segmentsRef.current = [...segmentsRef.current, ...event.committed];
            setSegments(segmentsRef.current);
        }
        provisionalRef.current = event.provisional;
        setProvisional(event.provisional);
    };

    const activateLegacyFallback = (message: string) => {
        if (transportRef.current !== "whisper-stream") return;
        transportRef.current = "legacy";
        setNotice(`${message} The original recording is still safe.`);
        liveSocketRef.current?.abort();
        liveSocketRef.current = null;
        void pcmCaptureRef.current?.abort();
        pcmCaptureRef.current = null;
        const recorder = mediaRecorderRef.current;
        if (recorder?.state === "recording") recorder.requestData();
    };

    const stopLiveTranscription = () => {
        if (state === "save-error") {
            void finalizeLiveRecording();
            return;
        }
        const recorder = mediaRecorderRef.current;
        if (!recorder || recorder.state === "inactive") return;

        setState("stopping");
        if (transportRef.current === "whisper-stream") {
            processingQueueRef.current = processingQueueRef.current.then(async () => {
                await pcmCaptureRef.current?.finish();
                pcmCaptureRef.current = null;
                const final = await liveSocketRef.current?.finish();
                liveSocketRef.current = null;
                if (final) {
                    segmentsRef.current = final.segments;
                    provisionalRef.current = null;
                    setSegments(final.segments);
                    setProvisional(null);
                }
            }).catch((err) => {
                const message = err instanceof Error
                    ? err.message
                    : "Realtime transcript finalization failed.";
                setNotice(`${message} Saving the captured recording anyway.`);
            });
        }
        recorder.stop();
    };

    const enqueueLegacyChunk = () => {
        const chunkIndex = chunkIndexRef.current;
        const fullRecordingBlob = new Blob(chunksRef.current, {
            type: finalMimeTypeRef.current,
        });
        const file = new File(
            [fullRecordingBlob],
            `live-chunk-${chunkIndex}.${extensionForMimeType(finalMimeTypeRef.current)}`,
            { type: finalMimeTypeRef.current },
        );
        chunkIndexRef.current += 1;

        processingQueueRef.current = processingQueueRef.current
            .then(async () => {
                const result = await transcribeLiveChunk({
                    projectId,
                    file,
                    chunkIndex,
                    chunkStart: 0,
                    language,
                    model,
                    device: selectedDevice,
                });
                segmentsRef.current = result.segments;
                provisionalRef.current = null;
                setSegments(result.segments);
                setProvisional(null);
            })
            .catch((err) => {
                const message = err instanceof Error
                    ? err.message
                    : "Compatibility transcription failed.";
                setNotice(`${message} The original recording will still be saved.`);
            });
    };

    const finalizeLiveRecording = async () => {
        releaseCaptureStream();
        mediaRecorderRef.current = null;
        setState("saving");
        setError("");

        try {
            await processingQueueRef.current;
            const finalMimeType = finalMimeTypeRef.current;
            const finalFile = new File(
                chunksRef.current,
                `live-recording-${timestampForFilename()}.${extensionForMimeType(finalMimeType)}`,
                { type: finalMimeType },
            );
            if (finalFile.size === 0) {
                throw new Error("The browser did not produce any recorded audio.");
            }

            const finalSegments = segmentsRef.current;
            const recording = finalSegments.length > 0
                ? await saveLiveRecording({
                    projectId,
                    file: finalFile,
                    segments: finalSegments,
                    language,
                    model,
                    device: selectedDevice,
                })
                : await uploadProjectRecording({
                    projectId,
                    file: finalFile,
                    language,
                    model,
                    device: selectedDevice,
                    extractSpeakers: false,
                });
            await cleanupRealtime();
            onSaved(recording);
            resetLiveTranscription();
        } catch (err) {
            await cleanupRealtime();
            setState("save-error");
            setError(err instanceof Error ? err.message : "Failed to save live recording");
        }
    };

    const resetLiveTranscription = () => {
        liveSocketRef.current?.abort();
        liveSocketRef.current = null;
        void pcmCaptureRef.current?.abort();
        pcmCaptureRef.current = null;
        releaseCaptureStream();
        prepareNewSession();
        setElapsedSeconds(0);
        setState("idle");
    };

    if (isSetupMode) {
        return (
            <section className="rounded-lg border border-zinc-200 bg-white">
                <div className="border-b border-zinc-200 px-4 py-3">
                    <h2 className="text-sm font-semibold text-zinc-900">Live transcription</h2>
                    <p className="mt-1 text-sm text-zinc-500">
                        Whisper streams continuously; Parakeet uses compatibility chunks.
                    </p>
                </div>

                <div className="flex max-w-xl flex-col gap-4 p-4">
                    <div className="flex flex-col gap-2 rounded-md border border-zinc-200 p-3">
                        <p className="text-xs font-medium text-zinc-500">Audio sources</p>
                        <SourceCheckbox
                            label="Microphone"
                            description="Capture your voice."
                            checked={includeMicrophone}
                            disabled={state !== "idle"}
                            onChange={(event) => setIncludeMicrophone(event.target.checked)}
                        />
                        <SourceCheckbox
                            label="System audio"
                            description="Capture shared tab, window, or meeting audio."
                            checked={includeSystemAudio}
                            disabled={state !== "idle"}
                            onChange={(event) => setIncludeSystemAudio(event.target.checked)}
                        />
                    </div>

                    <label className="flex flex-col gap-1">
                        <span className="text-xs font-medium text-zinc-500">Language</span>
                        <select
                            value={language}
                            onChange={(event) => setLanguage(event.target.value)}
                            disabled={state !== "idle"}
                            className="min-h-10 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none hover:cursor-pointer focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                            {TRANSCRIPTION_LANGUAGES.map((item) => (
                                <option key={item.value} value={item.value}>{item.label}</option>
                            ))}
                        </select>
                    </label>

                    <label className="flex flex-col gap-1">
                        <span className="text-xs font-medium text-zinc-500">Model</span>
                        <select
                            value={model}
                            onChange={(event) => setModel(event.target.value as TranscriptionModel)}
                            disabled={state !== "idle"}
                            className="min-h-10 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none hover:cursor-pointer focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                            {TRANSCRIPTION_MODELS.map((item) => (
                                <option
                                    key={item.value}
                                    value={item.value}
                                    disabled={hasNonEnglishLanguage && item.value === "parakeet"}
                                >
                                    {item.label}
                                </option>
                            ))}
                        </select>
                        {hasNonEnglishLanguage && (
                            <span className="text-xs text-zinc-500">
                                Parakeet currently supports English only.
                            </span>
                        )}
                    </label>

                    <label className="flex flex-col gap-1">
                        <span className="text-xs font-medium text-zinc-500">Device</span>
                        <select
                            value={selectedDevice}
                            onChange={(event) => setDevice(event.target.value as RuntimeDevice)}
                            disabled={state !== "idle"}
                            className="min-h-10 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none hover:cursor-pointer focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                            {runtimeDevices.available.map((item) => (
                                <option key={item} value={item}>{deviceLabel(item)}</option>
                            ))}
                        </select>
                    </label>

                    <button
                        type="button"
                        onClick={startLiveTranscription}
                        disabled={isBusy}
                        className="min-h-10 rounded-md bg-zinc-950 px-4 text-sm font-medium text-white hover:cursor-pointer hover:bg-zinc-700 disabled:cursor-not-allowed disabled:opacity-40"
                    >
                        {state === "requesting" ? "Preparing audio and model" : "Start live"}
                    </button>
                    {error && <p className="text-sm text-red-700">{error}</p>}
                </div>
            </section>
        );
    }

    return (
        <section className="rounded-lg border border-zinc-200 bg-white">
            <div className="border-b border-zinc-200 px-4 py-3">
                <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                        <h2 className="text-sm font-semibold text-zinc-900">Live transcription</h2>
                        <p className="mt-1 text-sm text-zinc-500">
                            {transportRef.current === "whisper-stream"
                                ? "Streaming low-latency Whisper preview"
                                : "Compatibility transcription preview"}
                        </p>
                    </div>
                    <span className="rounded-full bg-zinc-100 px-3 py-1 text-sm font-medium text-zinc-700">
                        {formatDuration(elapsedSeconds)}
                    </span>
                </div>
            </div>

            <div className="flex flex-col gap-4 p-4">
                <div className="flex justify-end gap-2">
                    {state === "save-error" && (
                        <button
                            type="button"
                            onClick={resetLiveTranscription}
                            className="min-h-10 rounded-md border border-zinc-300 px-4 text-sm font-medium text-zinc-700 hover:border-zinc-400 hover:cursor-pointer"
                        >
                            Discard local recording
                        </button>
                    )}
                    <button
                        type="button"
                        onClick={stopLiveTranscription}
                        disabled={!isRecording && state !== "save-error"}
                        className="min-h-10 rounded-md bg-red-700 px-4 text-sm font-medium text-white hover:bg-red-800 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                        {state === "stopping"
                            ? "Finalizing transcript"
                            : state === "saving"
                                ? "Saving recording"
                                : state === "save-error"
                                    ? "Retry save"
                                    : "Stop and save"}
                    </button>
                </div>

                {notice && (
                    <p className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
                        {notice}
                    </p>
                )}

                <div className="h-[min(520px,60vh)] overflow-y-auto rounded-md border border-zinc-200 bg-zinc-50 p-4 pr-2" aria-live="polite">
                    {segments.length === 0 && !provisional ? (
                        <div className="flex min-h-full items-center text-sm text-zinc-500">
                            Start speaking and the live transcript will appear here.
                        </div>
                    ) : (
                        <div className="flex flex-col gap-3">
                            {segments.map((segment, index) => (
                                <TranscriptLine
                                    key={`${segment.start}-${segment.end}-${index}`}
                                    segment={segment}
                                />
                            ))}
                            {provisional && <TranscriptLine segment={provisional} provisional />}
                        </div>
                    )}
                </div>

                {error && <p className="text-sm text-red-700">{error}</p>}
            </div>
        </section>
    );

    function prepareNewSession() {
        chunksRef.current = [];
        chunkIndexRef.current = 0;
        processingQueueRef.current = Promise.resolve();
        segmentsRef.current = [];
        provisionalRef.current = null;
        transportRef.current = "legacy";
        setSegments([]);
        setProvisional(null);
        setNotice("");
        setError("");
    }

    async function cleanupRealtime() {
        liveSocketRef.current?.abort();
        liveSocketRef.current = null;
        await pcmCaptureRef.current?.abort();
        pcmCaptureRef.current = null;
    }

    function releaseCaptureStream() {
        streamRef.current?.getTracks().forEach((track) => track.stop());
        streamRef.current = null;
        recordingAudioContextRef.current?.close().catch(() => undefined);
        recordingAudioContextRef.current = null;
    }
}

function TranscriptLine({
    segment,
    provisional = false,
}: {
    segment: SpeakerSegment;
    provisional?: boolean;
}) {
    return (
        <div className={`text-sm ${provisional ? "opacity-55" : ""}`}>
            <span className="mr-3 font-mono text-xs text-zinc-500">
                {formatTime(segment.start)}
            </span>
            <span className={provisional ? "italic text-zinc-600" : "text-zinc-900"}>
                {segment.text}
            </span>
        </div>
    );
}

function SourceCheckbox({
    label,
    description,
    checked,
    disabled,
    onChange,
}: {
    label: string;
    description: string;
    checked: boolean;
    disabled: boolean;
    onChange: (event: ChangeEvent<HTMLInputElement>) => void;
}) {
    return (
        <label className="flex items-start gap-2 text-sm text-zinc-700 hover:cursor-pointer">
            <input
                type="checkbox"
                checked={checked}
                disabled={disabled}
                onChange={onChange}
                className="mt-0.5 h-4 w-4 rounded border-zinc-300 text-zinc-950 focus:ring-zinc-950 disabled:cursor-not-allowed disabled:opacity-50"
            />
            <span>
                {label}
                <span className="block text-xs text-zinc-500">{description}</span>
            </span>
        </label>
    );
}

async function createCaptureStream({
    includeMicrophone,
    includeSystemAudio,
}: {
    includeMicrophone: boolean;
    includeSystemAudio: boolean;
}): Promise<{
    recordingStream: MediaStream;
    cleanupStream: MediaStream;
    audioContext: AudioContext | null;
}> {
    if (includeMicrophone && !includeSystemAudio) {
        const stream = await navigator.mediaDevices.getUserMedia({
            audio: AUDIO_CAPTURE_CONSTRAINTS,
        });
        return { recordingStream: stream, cleanupStream: stream, audioContext: null };
    }

    if (!includeMicrophone && includeSystemAudio) {
        const displayStream = await captureSystemAudio();
        return {
            recordingStream: new MediaStream(displayStream.getAudioTracks()),
            cleanupStream: displayStream,
            audioContext: null,
        };
    }

    const displayStream = await captureSystemAudio();
    let micStream: MediaStream | null = null;
    try {
        micStream = await navigator.mediaDevices.getUserMedia({
            audio: AUDIO_CAPTURE_CONSTRAINTS,
        });
        const audioContext = new AudioContext();
        const destination = audioContext.createMediaStreamDestination();
        audioContext.createMediaStreamSource(micStream).connect(destination);
        audioContext.createMediaStreamSource(displayStream).connect(destination);
        const cleanupStream = new MediaStream([
            ...micStream.getTracks(),
            ...displayStream.getTracks(),
        ]);
        return { recordingStream: destination.stream, cleanupStream, audioContext };
    } catch (error) {
        micStream?.getTracks().forEach((track) => track.stop());
        displayStream.getTracks().forEach((track) => track.stop());
        throw error;
    }
}

async function captureSystemAudio(): Promise<MediaStream> {
    const displayStream = await navigator.mediaDevices.getDisplayMedia({
        video: true,
        audio: AUDIO_CAPTURE_CONSTRAINTS,
    });
    if (displayStream.getAudioTracks().length === 0) {
        displayStream.getTracks().forEach((track) => track.stop());
        throw new Error("NO_SCREEN_AUDIO");
    }
    return displayStream;
}

function chooseMimeType(): string {
    if (!hasMediaRecorderSupport()) return "";
    return MIME_TYPES.find((mimeType) => MediaRecorder.isTypeSupported(mimeType)) ?? "";
}

function mediaRecorderOptions(mimeType: string): MediaRecorderOptions {
    return {
        ...(mimeType ? { mimeType } : {}),
        audioBitsPerSecond: AUDIO_BITS_PER_SECOND,
    };
}

function hasMediaRecorderSupport(): boolean {
    return typeof MediaRecorder !== "undefined" && Boolean(navigator.mediaDevices);
}

function extensionForMimeType(mimeType: string): string {
    if (mimeType.includes("mp4")) return "m4a";
    if (mimeType.includes("ogg")) return "ogg";
    return "webm";
}

function timestampForFilename(): string {
    return new Date().toISOString().replace(/[:.]/g, "-").slice(0, 19);
}

function formatDuration(seconds: number): string {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes}:${String(remainingSeconds).padStart(2, "0")}`;
}

function formatTime(seconds: number): string {
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = Math.floor(seconds % 60);
    return `${minutes}:${String(remainingSeconds).padStart(2, "0")}`;
}

function errorMessage(err: unknown, hasMicrophone: boolean): string {
    if (err instanceof Error && err.message === "NO_SCREEN_AUDIO") {
        return "No audio track was shared. Choose a tab/window with audio enabled, or use microphone mode.";
    }
    if (err instanceof DOMException && err.name === "NotAllowedError") {
        return "Recording permission was denied.";
    }
    if (err instanceof DOMException && err.name === "NotFoundError") {
        return hasMicrophone
            ? "No microphone was found."
            : "No screen or audio source was selected.";
    }
    return err instanceof Error ? err.message : "Failed to start live transcription.";
}
