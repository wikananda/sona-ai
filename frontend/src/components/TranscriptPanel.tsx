import {
    useEffect,
    useMemo,
    useRef,
    useState,
    type FocusEvent,
    type FormEvent,
    type KeyboardEvent,
    type PointerEvent,
} from "react";
import { SpeakerSegment, TranscriptSegmentUpdateParams } from "@/src/api/sonaApi";
import { exportTranscriptPdf } from "@/src/utils/pdfExport";

const EDIT_HOLD_MS = 600;

interface Props {
    recordingName: string;
    segments: SpeakerSegment[];
    isSavingSpeakers?: boolean;
    onRenameSpeakers?: (speakers: Record<string, string>) => Promise<void>;
    isSpeakerEditorOpen?: boolean;
    onSpeakerEditorClose?: () => void;
    isSavingSegment?: boolean;
    onUpdateSegment?: (
        segmentIndex: number,
        params: TranscriptSegmentUpdateParams,
    ) => Promise<void>;
    activeSegmentIndex?: number;
    onSeekToSegment?: (start: number) => void;
}

export default function TranscriptPanel({
    recordingName,
    segments,
    isSavingSpeakers = false,
    onRenameSpeakers,
    isSpeakerEditorOpen = false,
    onSpeakerEditorClose,
    isSavingSegment = false,
    onUpdateSegment,
    activeSegmentIndex = -1,
    onSeekToSegment,
}: Props) {
    const holdTimerRef = useRef<number | null>(null);
    const longPressTriggeredRef = useRef(false);
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
    const [editingIndex, setEditingIndex] = useState<number | null>(null);
    const [draftText, setDraftText] = useState("");
    const [draftSpeaker, setDraftSpeaker] = useState("");
    const [segmentError, setSegmentError] = useState("");
    const [isExportingPdf, setIsExportingPdf] = useState(false);
    const [exportError, setExportError] = useState("");

    useEffect(() => {
        return () => {
            if (holdTimerRef.current !== null) {
                window.clearTimeout(holdTimerRef.current);
            }
        };
    }, []);

    if (segments.length === 0) return null;

    const clearHoldTimer = () => {
        if (holdTimerRef.current !== null) {
            window.clearTimeout(holdTimerRef.current);
            holdTimerRef.current = null;
        }
    };

    const startSegmentEdit = (index: number) => {
        if (!onUpdateSegment || isSavingSegment) return;

        clearHoldTimer();
        longPressTriggeredRef.current = true;
        setSegmentError("");
        setEditingIndex(index);
        setDraftText(segments[index]?.text ?? "");
        setDraftSpeaker(segments[index]?.speaker ?? "");
    };

    const cancelSegmentEdit = () => {
        setEditingIndex(null);
        setDraftText("");
        setDraftSpeaker("");
        setSegmentError("");
    };

    const saveSegmentEdit = async () => {
        if (editingIndex === null || !onUpdateSegment || isSavingSegment) return;

        const segment = segments[editingIndex];
        if (!segment) {
            cancelSegmentEdit();
            return;
        }

        const nextText = draftText.trim();
        const nextSpeaker = draftSpeaker.trim();
        if (!nextText) {
            setSegmentError("Transcript text cannot be empty.");
            return;
        }
        if (segment.speaker && !nextSpeaker) {
            setSegmentError("Speaker name cannot be empty.");
            return;
        }

        const speakerPayload = segment.speaker ? nextSpeaker : undefined;
        if (nextText === segment.text && speakerPayload === segment.speaker) {
            cancelSegmentEdit();
            return;
        }

        setSegmentError("");
        try {
            await onUpdateSegment(editingIndex, {
                text: nextText,
                speaker: speakerPayload,
            });
            cancelSegmentEdit();
        } catch (err) {
            setSegmentError(
                err instanceof Error ? err.message : "Failed to save transcript segment.",
            );
        }
    };

    const handleSegmentPointerDown = (
        event: PointerEvent<HTMLDivElement>,
        index: number,
    ) => {
        if (!onUpdateSegment || editingIndex !== null || event.button !== 0) return;

        clearHoldTimer();
        longPressTriggeredRef.current = false;
        holdTimerRef.current = window.setTimeout(() => {
            startSegmentEdit(index);
        }, EDIT_HOLD_MS);
    };

    const handleSegmentClick = (segment: SpeakerSegment) => {
        clearHoldTimer();
        if (editingIndex !== null) return;
        if (longPressTriggeredRef.current) {
            longPressTriggeredRef.current = false;
            return;
        }

        onSeekToSegment?.(segment.start);
    };

    const handleSegmentKeyDown = (
        event: KeyboardEvent<HTMLDivElement>,
        segment: SpeakerSegment,
    ) => {
        if (!onSeekToSegment || editingIndex !== null) return;
        if (event.key !== "Enter" && event.key !== " ") return;

        event.preventDefault();
        onSeekToSegment(segment.start);
    };

    const handleEditKeyDown = (event: KeyboardEvent<HTMLInputElement | HTMLTextAreaElement>) => {
        if (event.key === "Escape") {
            event.preventDefault();
            cancelSegmentEdit();
            return;
        }

        if (event.key === "Enter" && !event.shiftKey) {
            event.preventDefault();
            void saveSegmentEdit();
        }
    };

    const handleSegmentBlur = (event: FocusEvent<HTMLDivElement>, isEditing: boolean) => {
        if (!isEditing) return;
        const nextTarget = event.relatedTarget;
        if (nextTarget instanceof Node && event.currentTarget.contains(nextTarget)) return;

        void saveSegmentEdit();
    };

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

    const handleExportPdf = async () => {
        if (isExportingPdf) return;

        setExportError("");
        setIsExportingPdf(true);
        try {
            await exportTranscriptPdf({ recordingName, segments });
        } catch (err) {
            setExportError(
                err instanceof Error ? err.message : "Failed to export transcript PDF.",
            );
        } finally {
            setIsExportingPdf(false);
        }
    };

    return (
        <div className="flex max-w-full flex-col">
            <div className="mb-3 flex flex-wrap items-center justify-between gap-2">
                <div>
                    <h3 className="text-sm font-semibold text-zinc-900">Transcript</h3>
                    <p className="mt-1 text-sm text-zinc-500">
                        Review and edit the detected transcript segments.
                    </p>
                </div>
                <button
                    type="button"
                    onClick={handleExportPdf}
                    disabled={isExportingPdf || segments.length === 0}
                    className="min-h-9 rounded-md border border-zinc-300 px-3 text-sm font-medium text-zinc-700 hover:border-zinc-400 hover:text-zinc-950 disabled:cursor-not-allowed disabled:opacity-50"
                >
                    {isExportingPdf ? "Exporting..." : "Export PDF"}
                </button>
            </div>
            {exportError && (
                <p className="mb-3 text-sm text-red-700">{exportError}</p>
            )}
            <div className="flex max-h-[520px] flex-col gap-1 overflow-y-auto pr-2">
                {segments.map((segment, index) => {
                    const isEditing = editingIndex === index;
                    const isSeekable = Boolean(onSeekToSegment) && !isEditing;
                    const canEditSegment = Boolean(onUpdateSegment);

                    return (
                        <div
                            key={index}
                            role={isSeekable ? "button" : undefined}
                            tabIndex={isSeekable ? 0 : undefined}
                            onPointerDown={(event) => handleSegmentPointerDown(event, index)}
                            onPointerUp={clearHoldTimer}
                            onPointerCancel={clearHoldTimer}
                            onPointerLeave={clearHoldTimer}
                            onClick={() => handleSegmentClick(segment)}
                            onKeyDown={(event) => handleSegmentKeyDown(event, segment)}
                            onBlur={(event) => handleSegmentBlur(event, isEditing)}
                            style={{
                                gridTemplateColumns: speakers.length
                                    ? `${speakerColumnCh}ch minmax(0, 1fr) auto`
                                    : "minmax(0, 1fr) auto",
                            }}
                            className={`group grid w-full items-start gap-3 border-b border-zinc-100 py-3 text-left transition-colors last:border-b-0 ${activeSegmentIndex === index && !isEditing
                                    ? "bg-zinc-100"
                                    : isSeekable
                                        ? "hover:bg-zinc-50"
                                        : ""
                                } ${isSeekable ? "cursor-pointer" : "cursor-default"}`}
                        >
                            {speakers.length > 0 && (
                                isEditing && segment.speaker ? (
                                    <input
                                        type="text"
                                        value={draftSpeaker}
                                        disabled={isSavingSegment}
                                        onChange={(event) => setDraftSpeaker(event.target.value)}
                                        onClick={(event) => event.stopPropagation()}
                                        onPointerDown={(event) => event.stopPropagation()}
                                        onKeyDown={handleEditKeyDown}
                                        className="min-h-9 min-w-0 rounded-md border border-zinc-300 px-2 text-sm font-bold leading-relaxed text-zinc-900 outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-60"
                                    />
                                ) : (
                                    <span className="min-w-0 break-words text-sm font-bold leading-relaxed text-zinc-700">
                                        {segment.speaker}
                                    </span>
                                )
                            )}

                            {isEditing ? (
                                <div className="min-w-0">
                                    <textarea
                                        value={draftText}
                                        autoFocus
                                        disabled={isSavingSegment}
                                        onChange={(event) => setDraftText(event.target.value)}
                                        onClick={(event) => event.stopPropagation()}
                                        onPointerDown={(event) => event.stopPropagation()}
                                        onKeyDown={handleEditKeyDown}
                                        rows={Math.max(2, Math.min(6, draftText.split("\n").length))}
                                        className="w-full resize-y rounded-md border border-zinc-300 px-3 py-2 text-sm leading-relaxed text-zinc-900 outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-60"
                                    />
                                    {segmentError ? (
                                        <p className="mt-1 text-xs text-red-700">
                                            {segmentError}
                                        </p>
                                    ) : (
                                        <p className="mt-1 text-xs text-zinc-500">
                                            Enter saves. Shift+Enter adds a line. Escape cancels.
                                        </p>
                                    )}
                                </div>
                            ) : (
                                <p className="min-w-0 text-sm leading-relaxed text-zinc-700">
                                    {segment.text}
                                </p>
                            )}

                            <div className="flex min-w-16 justify-end">
                                {isEditing ? (
                                    <span className="text-xs font-medium text-zinc-500">
                                        {isSavingSegment ? "Saving..." : "Editing"}
                                    </span>
                                ) : (
                                    canEditSegment && (
                                        <button
                                            type="button"
                                            onPointerDown={(event) => event.stopPropagation()}
                                            onClick={(event) => {
                                                event.stopPropagation();
                                                startSegmentEdit(index);
                                            }}
                                            className="rounded-md px-2 py-1 text-xs font-medium text-zinc-500 opacity-0 transition-opacity hover:bg-zinc-100 hover:text-zinc-950 cursor-pointer focus:opacity-100 group-hover:opacity-100"
                                        >
                                            Edit
                                        </button>
                                    )
                                )}
                            </div>
                        </div>
                    );
                })}
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
