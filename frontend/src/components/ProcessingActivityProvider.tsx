"use client";

import Link from "next/link";
import {
    createContext,
    ReactNode,
    useCallback,
    useContext,
    useEffect,
    useMemo,
    useState,
} from "react";
import {
    cancelRecording,
    getRuntimeModels,
    listProcessingRecordings,
    ProcessingRecording,
    RuntimeModel,
} from "@/src/api/sonaApi";

const POLL_INTERVAL_MS = 3000;

export interface LocalProcessingActivity {
    id: string;
    label: string;
    detail: string;
    projectId?: string;
    blocksNavigation?: boolean;
}

interface ProcessingActivityContextValue {
    setLocalActivity: (activity: LocalProcessingActivity) => void;
    removeLocalActivity: (activityId: string) => void;
    refresh: () => Promise<void>;
    hasBlockingLocalActivity: boolean;
}

const ProcessingActivityContext = createContext<ProcessingActivityContextValue | null>(null);

export default function ProcessingActivityProvider({ children }: { children: ReactNode }) {
    const [processingRecordings, setProcessingRecordings] = useState<ProcessingRecording[]>([]);
    const [busyRuntimeModels, setBusyRuntimeModels] = useState<RuntimeModel[]>([]);
    const [localActivities, setLocalActivities] = useState<Record<string, LocalProcessingActivity>>({});
    const [isExpanded, setIsExpanded] = useState(false);
    const [cancelingId, setCancelingId] = useState<string>();
    const [actionError, setActionError] = useState("");

    const refresh = useCallback(async () => {
        const [recordingsResult, modelsResult] = await Promise.allSettled([
            listProcessingRecordings(),
            getRuntimeModels(),
        ]);
        if (recordingsResult.status === "fulfilled") {
            setProcessingRecordings(recordingsResult.value);
        }
        if (modelsResult.status === "fulfilled") {
            setBusyRuntimeModels(modelsResult.value.filter((model) => model.is_busy));
        }
    }, []);

    useEffect(() => {
        void refresh();
        const intervalId = window.setInterval(refresh, POLL_INTERVAL_MS);
        return () => window.clearInterval(intervalId);
    }, [refresh]);

    const setLocalActivity = useCallback((activity: LocalProcessingActivity) => {
        setLocalActivities((current) => ({ ...current, [activity.id]: activity }));
    }, []);

    const removeLocalActivity = useCallback((activityId: string) => {
        setLocalActivities((current) => {
            if (!(activityId in current)) return current;
            const next = { ...current };
            delete next[activityId];
            return next;
        });
    }, []);

    const localActivityList = Object.values(localActivities);
    const hasBlockingLocalActivity = localActivityList.some(
        (activity) => activity.blocksNavigation !== false,
    );
    const shouldWarnBeforeUnload = localActivityList.length > 0
        || processingRecordings.length > 0
        || busyRuntimeModels.length > 0;

    useEffect(() => {
        if (!shouldWarnBeforeUnload) return;

        const warnBeforeUnload = (event: BeforeUnloadEvent) => {
            event.preventDefault();
            event.returnValue = "";
        };
        window.addEventListener("beforeunload", warnBeforeUnload);
        return () => window.removeEventListener("beforeunload", warnBeforeUnload);
    }, [shouldWarnBeforeUnload]);

    const contextValue = useMemo<ProcessingActivityContextValue>(() => ({
        setLocalActivity,
        removeLocalActivity,
        refresh,
        hasBlockingLocalActivity,
    }), [hasBlockingLocalActivity, refresh, removeLocalActivity, setLocalActivity]);

    const totalActivities = localActivityList.length
        + processingRecordings.length
        + busyRuntimeModels.length;

    const handleCancel = async (recordingId: string) => {
        if (!window.confirm("Cancel this background task? The original audio will be kept.")) return;
        setCancelingId(recordingId);
        setActionError("");
        try {
            await cancelRecording(recordingId);
            setProcessingRecordings((current) => current.filter(
                (recording) => recording.id !== recordingId,
            ));
        } catch (error) {
            setActionError(error instanceof Error ? error.message : "Could not cancel task");
            await refresh();
        } finally {
            setCancelingId(undefined);
        }
    };

    return (
        <ProcessingActivityContext.Provider value={contextValue}>
            {children}
            {totalActivities > 0 && (
                <aside className="fixed bottom-4 right-4 z-40 w-[min(380px,calc(100vw-2rem))] overflow-hidden rounded-lg border border-zinc-300 bg-white shadow-xl">
                    <button
                        type="button"
                        onClick={() => setIsExpanded((current) => !current)}
                        aria-expanded={isExpanded}
                        className="flex min-h-12 w-full items-center justify-between gap-4 px-4 py-3 text-left"
                    >
                        <span className="flex min-w-0 items-center gap-2">
                            <span className="h-2.5 w-2.5 shrink-0 animate-pulse rounded-full bg-emerald-500" />
                            <span className="truncate text-sm font-semibold text-zinc-900">
                                {totalActivities} active {totalActivities === 1 ? "task" : "tasks"}
                            </span>
                        </span>
                        <span className="text-xs font-medium text-zinc-500">
                            {isExpanded ? "Hide" : "View"}
                        </span>
                    </button>

                    {isExpanded && (
                        <div className="max-h-80 divide-y divide-zinc-200 overflow-y-auto border-t border-zinc-200">
                            {localActivityList.map((activity) => (
                                <div key={activity.id} className="px-4 py-3">
                                    <div className="flex items-start justify-between gap-3">
                                        <div className="min-w-0">
                                            <p className="truncate text-sm font-medium text-zinc-900">
                                                {activity.label}
                                            </p>
                                            <p className="mt-1 text-xs text-amber-700">
                                                {activity.detail}
                                            </p>
                                        </div>
                                        <span className="shrink-0 rounded-full bg-amber-100 px-2 py-1 text-[11px] font-semibold text-amber-800">
                                            Keep open
                                        </span>
                                    </div>
                                </div>
                            ))}

                            {processingRecordings.map((recording) => (
                                <div key={recording.id} className="px-4 py-3">
                                    <div className="flex items-start justify-between gap-3">
                                        <div className="min-w-0">
                                            <Link
                                                href={`/projects/${recording.project_id}?recording=${recording.id}`}
                                                className="block truncate text-sm font-medium text-zinc-900 hover:underline"
                                            >
                                                {recording.original_name}
                                            </Link>
                                            <p className="mt-1 truncate text-xs text-zinc-500">
                                                {recording.project_name} · {recording.progress.label}
                                            </p>
                                            <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-zinc-200">
                                                <div
                                                    className="h-full rounded-full bg-zinc-900 transition-[width]"
                                                    style={{ width: `${recording.progress.percent}%` }}
                                                />
                                            </div>
                                        </div>
                                        <button
                                            type="button"
                                            onClick={() => void handleCancel(recording.id)}
                                            disabled={cancelingId === recording.id}
                                            className="shrink-0 text-xs font-semibold text-red-700 disabled:opacity-50"
                                        >
                                            {cancelingId === recording.id ? "Canceling" : "Cancel"}
                                        </button>
                                    </div>
                                </div>
                            ))}
                            {busyRuntimeModels.map((model) => (
                                <div key={model.id} className="px-4 py-3">
                                    <p className="truncate text-sm font-medium text-zinc-900">
                                        {model.label}
                                    </p>
                                    <p className="mt-1 text-xs text-zinc-500">
                                        Model management is running in the background
                                    </p>
                                    <div className="mt-2 h-1.5 overflow-hidden rounded-full bg-zinc-200">
                                        <div className="h-full w-2/5 animate-pulse rounded-full bg-blue-600" />
                                    </div>
                                </div>
                            ))}
                            {actionError && (
                                <p className="px-4 py-3 text-xs text-red-700">{actionError}</p>
                            )}
                        </div>
                    )}
                </aside>
            )}
        </ProcessingActivityContext.Provider>
    );
}

export function useProcessingActivity(activity: LocalProcessingActivity | null) {
    const { setLocalActivity, removeLocalActivity } = useProcessingActivities();
    const activityId = activity?.id;
    const label = activity?.label;
    const detail = activity?.detail;
    const projectId = activity?.projectId;
    const blocksNavigation = activity?.blocksNavigation;

    useEffect(() => {
        if (!activityId || !label || !detail) return;
        setLocalActivity({
            id: activityId,
            label,
            detail,
            projectId,
            blocksNavigation,
        });
        return () => removeLocalActivity(activityId);
    }, [
        activityId,
        blocksNavigation,
        detail,
        label,
        projectId,
        removeLocalActivity,
        setLocalActivity,
    ]);
}

export function useProcessingActivities(): ProcessingActivityContextValue {
    const context = useContext(ProcessingActivityContext);
    if (!context) {
        throw new Error("useProcessingActivities must be used within ProcessingActivityProvider");
    }
    return context;
}
