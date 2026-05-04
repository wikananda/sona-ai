import { FormEvent, useMemo, useState } from "react";
import { SpeakerSegment } from "@/src/api/sonaApi";

interface Props {
    segments: SpeakerSegment[];
    isSavingSpeakers?: boolean;
    onRenameSpeakers?: (speakers: Record<string, string>) => Promise<void>;
    isSpeakerEditorOpen?: boolean;
    onSpeakerEditorClose?: () => void;
}

export default function TranscriptPanel({
    segments,
    isSavingSpeakers = false,
    onRenameSpeakers,
    isSpeakerEditorOpen = false,
    onSpeakerEditorClose,
}: Props) {
    const speakers = useMemo(
        () => Array.from(
            new Set(
                segments
                    .map((segment) => segment.speaker)
                    .filter((speaker): speaker is string => Boolean(speaker)),
            ),
        ).sort(),
        [segments],
    );
    const speakerColumnCh = useMemo(() => {
        const longestSpeaker = speakers.reduce(
            (longest, speaker) => Math.max(longest, speaker.length),
            0,
        );
        return Math.min(Math.max(longestSpeaker + 1, 6), 20);
    }, [speakers]);
    const [speakerError, setSpeakerError] = useState("");

    if (segments.length === 0) return null;

    const closeSpeakerEditor = () => {
        if (isSavingSpeakers) return;

        onSpeakerEditorClose?.();
        setSpeakerError("");
    };

    const handleSpeakerSave = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        const formData = new FormData(event.currentTarget);
        const nextSpeakerNames = Object.fromEntries(
            speakers.map((speaker) => [
                speaker,
                String(formData.get(speaker) ?? "").trim(),
            ]),
        );
        if (Object.values(nextSpeakerNames).some((speaker) => !speaker)) {
            setSpeakerError("Speaker names cannot be empty.");
            return;
        }

        const hasChanges = speakers.some(
            (speaker) => nextSpeakerNames[speaker] !== speaker,
        );
        if (!hasChanges) {
            closeSpeakerEditor();
            return;
        }

        setSpeakerError("");
        try {
            await onRenameSpeakers?.(nextSpeakerNames);
            onSpeakerEditorClose?.();
        } catch (err) {
            setSpeakerError(err instanceof Error ? err.message : "Failed to save speaker names.");
        }
    };

    return (
        <div className="flex max-w-full flex-col">
            <div className="flex max-h-[520px] flex-col gap-1 overflow-y-auto pr-2">
                {segments.map((segment, index) => (
                    <div
                        key={index}
                        style={{
                            gridTemplateColumns: `${speakerColumnCh}ch minmax(0, 1fr)`,
                        }}
                        className="grid items-start gap-3 border-b border-zinc-100 py-3 last:border-b-0"
                    >
                        <span className="min-w-0 break-words text-sm font-bold leading-relaxed text-zinc-700">
                            {segment.speaker}
                        </span>
                        <p className="text-sm leading-relaxed text-zinc-700">
                            {segment.text}
                        </p>
                    </div>
                ))}
            </div>

            {isSpeakerEditorOpen && speakers.length > 0 && (
                <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/35 px-4">
                    <form
                        onSubmit={handleSpeakerSave}
                        className="w-full max-w-md rounded-lg bg-white p-5 shadow-xl"
                    >
                        <div className="flex items-start justify-between gap-4">
                            <h3 className="text-base font-semibold text-zinc-950">
                                Edit speakers
                            </h3>
                            <button
                                type="button"
                                onClick={closeSpeakerEditor}
                                disabled={isSavingSpeakers}
                                className="text-sm font-medium text-zinc-500 hover:text-zinc-950 disabled:cursor-not-allowed disabled:opacity-50"
                            >
                                Close
                            </button>
                        </div>

                        <div className="mt-5 flex flex-col gap-4">
                            {speakers.map((speaker) => (
                                <label key={speaker} className="flex flex-col gap-1">
                                    <span className="text-xs font-medium text-zinc-500">
                                        {speaker}
                                    </span>
                                    <input
                                        type="text"
                                        name={speaker}
                                        defaultValue={speaker}
                                        disabled={isSavingSpeakers}
                                        className="min-h-10 rounded-md border border-zinc-300 px-3 text-sm text-zinc-950 outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                                    />
                                </label>
                            ))}
                        </div>

                        {speakerError && (
                            <p className="mt-4 text-sm text-red-700">{speakerError}</p>
                        )}

                        <div className="mt-6 flex justify-end gap-3">
                            <button
                                type="button"
                                onClick={closeSpeakerEditor}
                                disabled={isSavingSpeakers}
                                className="min-h-10 rounded-md border border-zinc-300 px-4 text-sm font-medium text-zinc-700 hover:border-zinc-400 hover:text-zinc-950 disabled:cursor-not-allowed disabled:opacity-50"
                            >
                                Cancel
                            </button>
                            <button
                                type="submit"
                                disabled={isSavingSpeakers}
                                className="min-h-10 rounded-md bg-zinc-950 px-4 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-50"
                            >
                                {isSavingSpeakers ? "Saving..." : "Save"}
                            </button>
                        </div>
                    </form>
                </div>
            )}
        </div>
    );
}
