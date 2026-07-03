"use client";

import { FormEvent, useState } from "react";
import {
    Recording,
    RetranscribeParams,
    RuntimeDevice,
    RuntimeDevices,
    TranscriptionModel,
} from "@/src/api/sonaApi";
import Modal from "@/src/components/ui/Modal";
import {
    deviceLabel,
    isTranscriptionModel,
    numberOrEmpty,
    TRANSCRIPTION_LANGUAGES,
    TRANSCRIPTION_MODELS,
} from "@/src/utils/transcriptionSettings";

interface Props {
    recording: Recording;
    runtimeDevices: RuntimeDevices;
    isRetranscribing: boolean;
    onClose: () => void;
    onSubmit: (settings: RetranscribeParams) => Promise<boolean>;
}

export default function RetranscribeEditorModal({
    recording,
    runtimeDevices,
    isRetranscribing,
    onClose,
    onSubmit,
}: Props) {
    const [retranscribeLanguage, setRetranscribeLanguage] = useState(
        recording.language_hint ?? "auto",
    );
    const [retranscribeModel, setRetranscribeModel] = useState<TranscriptionModel>(
        isTranscriptionModel(recording.model) ? recording.model : "parakeet",
    );
    const [retranscribeDevice, setRetranscribeDevice] = useState<RuntimeDevice>(
        runtimeDevices.available.includes(recording.device)
            ? recording.device
            : runtimeDevices.default,
    );
    const [retranscribeMinSpeakers, setRetranscribeMinSpeakers] = useState<number | "">(
        recording.min_speakers ?? "",
    );
    const [retranscribeMaxSpeakers, setRetranscribeMaxSpeakers] = useState<number | "">(
        recording.max_speakers ?? "",
    );
    const [retranscribeExtractSpeakers, setRetranscribeExtractSpeakers] = useState(true);
    const [retranscribeError, setRetranscribeError] = useState("");

    const handleRetranscribeSubmit = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        const selectedDevice = runtimeDevices.available.includes(retranscribeDevice)
            ? retranscribeDevice
            : runtimeDevices.default;
        if (
            retranscribeExtractSpeakers &&
            (retranscribeMinSpeakers === "" || retranscribeMaxSpeakers === "")
        ) {
            setRetranscribeError("Min and max speakers are required when extracting speakers.");
            return;
        }

        setRetranscribeError("");
        const started = await onSubmit({
            language: retranscribeLanguage,
            model: retranscribeModel,
            device: selectedDevice,
            minSpeakers: retranscribeExtractSpeakers ? retranscribeMinSpeakers : "",
            maxSpeakers: retranscribeExtractSpeakers ? retranscribeMaxSpeakers : "",
            extractSpeakers: retranscribeExtractSpeakers,
        });
        if (started) {
            onClose();
        }
    };

    return (
        <Modal backdropClassName="bg-black/35">
            <form
                onSubmit={handleRetranscribeSubmit}
                className="w-full max-w-2xl rounded-lg bg-white p-5 shadow-xl"
            >
                <div className="flex items-start justify-between gap-4">
                    <h3 className="text-base font-semibold text-zinc-950">
                        Re-transcribe settings
                    </h3>
                    <button
                        type="button"
                        onClick={onClose}
                        disabled={isRetranscribing}
                        className="text-sm font-medium text-zinc-500 hover:cursor-pointer hover:text-zinc-950 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                        Close
                    </button>
                </div>

                <div className="mt-5 grid gap-4 md:grid-cols-2">
                    <label className="flex flex-col gap-1">
                        <span className="text-xs font-medium text-zinc-500">
                            Language
                        </span>
                        <select
                            value={retranscribeLanguage}
                            onChange={(event) => setRetranscribeLanguage(event.target.value)}
                            disabled={isRetranscribing}
                            className="min-h-10 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                            {TRANSCRIPTION_LANGUAGES.map((item) => (
                                <option key={item.value} value={item.value}>
                                    {item.label}
                                </option>
                            ))}
                        </select>
                    </label>

                    <label className="flex flex-col gap-1">
                        <span className="text-xs font-medium text-zinc-500">
                            Model
                        </span>
                        <select
                            value={retranscribeModel}
                            onChange={(event) => setRetranscribeModel(
                                event.target.value as TranscriptionModel,
                            )}
                            disabled={isRetranscribing}
                            className="min-h-10 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                            {TRANSCRIPTION_MODELS.map((item) => (
                                <option key={item.value} value={item.value}>
                                    {item.label}
                                </option>
                            ))}
                        </select>
                    </label>

                    <label className="flex flex-col gap-1">
                        <span className="text-xs font-medium text-zinc-500">
                            Device
                        </span>
                        <select
                            value={
                                runtimeDevices.available.includes(retranscribeDevice)
                                    ? retranscribeDevice
                                    : runtimeDevices.default
                            }
                            onChange={(event) => setRetranscribeDevice(
                                event.target.value as RuntimeDevice,
                            )}
                            disabled={isRetranscribing}
                            className="min-h-10 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                            {runtimeDevices.available.map((device) => (
                                <option key={device} value={device}>
                                    {deviceLabel(device)}
                                </option>
                            ))}
                        </select>
                    </label>

                    <label className="flex items-start gap-2 md:col-span-2">
                        <input
                            type="checkbox"
                            checked={retranscribeExtractSpeakers}
                            onChange={(event) => {
                                setRetranscribeExtractSpeakers(event.target.checked);
                                setRetranscribeError("");
                            }}
                            disabled={isRetranscribing}
                            className="mt-0.5 h-4 w-4 rounded border-zinc-300 text-zinc-950 focus:ring-zinc-950 disabled:cursor-not-allowed disabled:opacity-50"
                        />
                        <span className="text-sm text-zinc-600">
                            Extract speakers during re-transcribe
                            <span className="block text-xs text-zinc-500">
                                Uncheck this to re-run ASR now and extract speakers later.
                            </span>
                        </span>
                    </label>

                    {retranscribeExtractSpeakers && (
                        <div className="grid gap-4 sm:grid-cols-2 md:col-span-2">
                            <label className="flex flex-col gap-1">
                                <span className="text-xs font-medium text-zinc-500">
                                    Min speakers
                                </span>
                                <input
                                    type="number"
                                    min="1"
                                    value={retranscribeMinSpeakers}
                                    onChange={(event) => {
                                        setRetranscribeMinSpeakers(
                                            numberOrEmpty(event.target.value),
                                        );
                                        setRetranscribeError("");
                                    }}
                                    disabled={isRetranscribing}
                                    placeholder="Required"
                                    className="min-h-10 rounded-md border border-zinc-300 px-3 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                                />
                            </label>

                            <label className="flex flex-col gap-1">
                                <span className="text-xs font-medium text-zinc-500">
                                    Max speakers
                                </span>
                                <input
                                    type="number"
                                    min="1"
                                    value={retranscribeMaxSpeakers}
                                    onChange={(event) => {
                                        setRetranscribeMaxSpeakers(
                                            numberOrEmpty(event.target.value),
                                        );
                                        setRetranscribeError("");
                                    }}
                                    disabled={isRetranscribing}
                                    placeholder="Required"
                                    className="min-h-10 rounded-md border border-zinc-300 px-3 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                                />
                            </label>
                        </div>
                    )}
                </div>

                {retranscribeError && (
                    <p className="mt-4 text-sm text-red-700">{retranscribeError}</p>
                )}

                <div className="mt-6 flex justify-end gap-3">
                    <button
                        type="button"
                        onClick={onClose}
                        disabled={isRetranscribing}
                        className="min-h-10 rounded-md border border-zinc-300 px-4 text-sm font-medium text-zinc-700 hover:cursor-pointer hover:border-zinc-400 hover:text-zinc-950 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                        Cancel
                    </button>
                    <button
                        type="submit"
                        disabled={isRetranscribing}
                        className="min-h-10 rounded-md bg-zinc-950 px-4 text-sm font-medium text-white hover:cursor-pointer disabled:cursor-not-allowed disabled:opacity-50"
                    >
                        {isRetranscribing ? "Starting..." : "Start re-transcribe"}
                    </button>
                </div>
            </form>
        </Modal>
    );
}
