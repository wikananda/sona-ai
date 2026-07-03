"use client";

import { Recording } from "@/src/api/sonaApi";

export default function RecordingProgressPanel({
    recording,
    helperText,
    isCanceling = false,
    onCancel,
}: {
    recording: Recording;
    helperText?: string;
    isCanceling?: boolean;
    onCancel?: () => Promise<void>;
}) {
    const progress = recording.progress;
    const percent = Math.max(0, Math.min(100, Math.round(progress?.percent ?? 0)));
    const totalSteps = progress?.total_steps ?? 0;
    const completedSteps = progress?.completed_steps ?? 0;
    const detail = totalSteps > 0
        ? `Completed ${completedSteps} of ${totalSteps} stages`
        : "Preparing progress details";

    return (
        <div className="rounded-md border border-amber-200 bg-amber-50 p-4">
            <div className="flex items-center justify-between gap-3">
                <div className="min-w-0">
                    <p className="text-sm font-medium text-amber-950">
                        {progress?.label ?? "Processing"}...
                    </p>
                    <p className="mt-1 text-xs text-amber-800">
                        {helperText ?? detail}
                    </p>
                </div>
                <div className="flex shrink-0 items-center gap-3">
                    <span className="text-sm font-semibold tabular-nums text-amber-950">
                        {percent}%
                    </span>
                    {onCancel && (
                        <button
                            type="button"
                            onClick={onCancel}
                            disabled={isCanceling}
                            aria-label="Cancel processing"
                            title="Cancel processing"
                            className="inline-flex h-7 w-7 items-center justify-center rounded-full border border-amber-300 text-lg leading-none text-amber-950 transition-colors hover:cursor-pointer hover:border-amber-500 hover:bg-amber-100 disabled:cursor-not-allowed disabled:opacity-50"
                        >
                            x
                        </button>
                    )}
                </div>
            </div>
            <div className="mt-3 h-2 overflow-hidden rounded-full bg-amber-100">
                <div
                    className="h-full rounded-full bg-amber-600 transition-[width] duration-300"
                    style={{ width: `${percent}%` }}
                />
            </div>
            {helperText && (
                <p className="mt-2 text-xs text-amber-800">{detail}</p>
            )}
        </div>
    );
}
