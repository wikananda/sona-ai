"use client";

import { FormEvent, useState } from "react";
import { Recording, SpeakerExtractionParams } from "@/src/api/sonaApi";
import Modal from "@/src/components/ui/Modal";
import { numberOrEmpty } from "@/src/utils/transcriptionSettings";

interface Props {
    recording: Recording;
    isExtractingSpeakers: boolean;
    onClose: () => void;
    onSubmit: (settings: SpeakerExtractionParams) => Promise<void>;
}

export default function SpeakerExtractionEditorModal({
    recording,
    isExtractingSpeakers,
    onClose,
    onSubmit,
}: Props) {
    const [extractMinSpeakers, setExtractMinSpeakers] = useState<number | "">(
        recording.min_speakers ?? "",
    );
    const [extractMaxSpeakers, setExtractMaxSpeakers] = useState<number | "">(
        recording.max_speakers ?? "",
    );
    const [speakerExtractionError, setSpeakerExtractionError] = useState("");

    const handleSpeakerExtractionSubmit = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        if (extractMinSpeakers === "" || extractMaxSpeakers === "") {
            setSpeakerExtractionError("Min and max speakers are required.");
            return;
        }

        setSpeakerExtractionError("");
        await onSubmit({
            minSpeakers: extractMinSpeakers,
            maxSpeakers: extractMaxSpeakers,
        });
        onClose();
    };

    return (
        <Modal backdropClassName="bg-black/35">
            <form
                onSubmit={handleSpeakerExtractionSubmit}
                className="w-full max-w-md rounded-lg bg-white p-5 shadow-xl"
            >
                <div className="flex items-start justify-between gap-4">
                    <div>
                        <h3 className="text-base font-semibold text-zinc-950">
                            Extract speaker settings
                        </h3>
                        <p className="mt-1 text-sm text-zinc-500">
                            Set speaker bounds for diarization.
                        </p>
                    </div>
                    <button
                        type="button"
                        onClick={onClose}
                        disabled={isExtractingSpeakers}
                        className="text-sm font-medium text-zinc-500 hover:text-zinc-950 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                        Close
                    </button>
                </div>

                <div className="mt-5 grid gap-4 sm:grid-cols-2">
                    <label className="flex flex-col gap-1">
                        <span className="text-xs font-medium text-zinc-500">
                            Min speakers
                        </span>
                        <input
                            type="number"
                            min="1"
                            value={extractMinSpeakers}
                            onChange={(event) => {
                                setExtractMinSpeakers(
                                    numberOrEmpty(event.target.value),
                                );
                                setSpeakerExtractionError("");
                            }}
                            disabled={isExtractingSpeakers}
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
                            value={extractMaxSpeakers}
                            onChange={(event) => {
                                setExtractMaxSpeakers(
                                    numberOrEmpty(event.target.value),
                                );
                                setSpeakerExtractionError("");
                            }}
                            disabled={isExtractingSpeakers}
                            placeholder="Required"
                            className="min-h-10 rounded-md border border-zinc-300 px-3 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                        />
                    </label>
                </div>

                {speakerExtractionError && (
                    <p className="mt-4 text-sm text-red-700">{speakerExtractionError}</p>
                )}

                <div className="mt-6 flex justify-end gap-3">
                    <button
                        type="button"
                        onClick={onClose}
                        disabled={isExtractingSpeakers}
                        className="min-h-10 rounded-md border border-zinc-300 px-4 text-sm font-medium text-zinc-700 hover:border-zinc-400 hover:text-zinc-950 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                        Cancel
                    </button>
                    <button
                        type="submit"
                        disabled={isExtractingSpeakers}
                        className="min-h-10 rounded-md bg-zinc-950 px-4 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-50"
                    >
                        {isExtractingSpeakers ? "Starting..." : "Start extraction"}
                    </button>
                </div>
            </form>
        </Modal>
    );
}
