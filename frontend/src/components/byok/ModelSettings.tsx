"use client";

import { ModelJob, ModelJobAction, RuntimeModel } from "@/src/api/sonaApi";
import {
    isActiveJob,
    modelJobLabel,
    modelJobStageLabel,
    modelStatusLabel,
    modelTypeLabel,
    statusClassName,
} from "@/src/components/byok/byokHelpers";

export function ModelSettings({
    models,
    jobs,
    error,
    isLoading,
    onRefresh,
    onAction,
}: {
    models: RuntimeModel[];
    jobs: Record<string, ModelJob>;
    error: string | null;
    isLoading: boolean;
    onRefresh: () => void;
    onAction: (model: RuntimeModel, action: ModelJobAction) => void;
}) {
    return (
        <div className="flex flex-col gap-4">
            <div className="flex items-start justify-between gap-3">
                <div>
                    <h3 className="text-sm font-semibold text-zinc-950">
                        Speech models
                    </h3>
                    <p className="mt-1 text-sm text-zinc-500">
                        Download ASR, alignment, and diarization models into the backend cache.
                    </p>
                </div>
                <button
                    type="button"
                    onClick={onRefresh}
                    className="min-h-9 rounded-md border border-zinc-300 px-3 text-sm font-medium text-zinc-700 hover:cursor-pointer hover:border-zinc-400 hover:text-zinc-950"
                >
                    {isLoading ? "Refreshing..." : "Refresh"}
                </button>
            </div>

            {error && (
                <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                    {error}
                </div>
            )}

            <div className="flex flex-col gap-3">
                {models.length === 0 && !isLoading ? (
                    <div className="rounded-md border border-zinc-200 px-3 py-6 text-center text-sm text-zinc-500">
                        No model information available.
                    </div>
                ) : (
                    models.map((model) => (
                        <ModelRow
                            key={model.id}
                            model={model}
                            job={model.active_job_id ? jobs[model.active_job_id] : undefined}
                            onAction={onAction}
                        />
                    ))
                )}
            </div>
        </div>
    );
}

function ModelRow({
    model,
    job,
    onAction,
}: {
    model: RuntimeModel;
    job?: ModelJob;
    onAction: (model: RuntimeModel, action: ModelJobAction) => void;
}) {
    const isRunning = model.status === "running" || Boolean(job && isActiveJob(job));
    const isBlocked = model.requires_hf_token && !model.hf_token_available;
    const progressLabel = job ? modelJobLabel(job) : null;

    return (
        <div className="rounded-md border border-zinc-200 p-3">
            <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                        <h4 className="text-sm font-semibold text-zinc-950">
                            {model.label}
                        </h4>
                        <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${statusClassName(model.status)}`}>
                            {modelStatusLabel(model.status)}
                        </span>
                    </div>
                    <p className="mt-1 text-xs text-zinc-500">
                        {modelTypeLabel(model.type)} model
                    </p>
                    {/* <p className="mt-1 break-all text-xs text-zinc-500">
                        {model.model_names.join(", ")}
                    </p> */}
                    {/* <p className="mt-1 break-all text-xs text-zinc-400">
                        Cache: {model.cache_path}
                    </p> */}
                    {isBlocked && (
                        <p className="mt-2 text-xs font-medium text-amber-700">
                            Requires backend HF_TOKEN in .env.
                        </p>
                    )}
                    {/* {message && (
                        <p className="mt-2 text-xs text-zinc-500">
                            {message}
                        </p>
                    )} */}
                    {job && isActiveJob(job) && (
                        <div className="mt-3">
                            <div className="mb-1 flex items-center justify-between gap-3">
                                <span className="text-xs font-medium text-zinc-700">
                                    {progressLabel}
                                </span>
                                <span className="text-xs text-zinc-500">
                                    {modelJobStageLabel(job.stage)}
                                </span>
                            </div>
                            <div className="h-2 overflow-hidden rounded-full bg-zinc-200">
                                <div className="h-full w-2/5 animate-pulse rounded-full bg-zinc-900" />
                            </div>
                        </div>
                    )}
                    {job?.error && (
                        <p className="mt-2 text-xs text-red-700">
                            {job.error}
                        </p>
                    )}
                </div>
                <div className="flex flex-wrap items-center justify-end gap-2">
                    {!model.installed && (
                        <button
                            type="button"
                            onClick={() => onAction(model, "download")}
                            disabled={isRunning || isBlocked}
                            className="min-h-9 rounded-md bg-zinc-950 px-3 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-zinc-200 disabled:text-zinc-500"
                        >
                            {isRunning ? modelJobLabel(job) : "Download"}
                        </button>
                    )}
                    {model.installed && model.can_uninstall && (
                        <button
                            type="button"
                            onClick={() => onAction(model, "uninstall")}
                            disabled={isRunning}
                            className="min-h-9 rounded-md border border-zinc-300 px-3 text-sm font-medium text-zinc-700 hover:cursor-pointer hover:border-zinc-400 hover:text-zinc-950 disabled:cursor-not-allowed disabled:bg-zinc-100 disabled:text-zinc-400"
                        >
                            {isRunning && job?.action === "uninstall" ? "Uninstalling" : "Uninstall"}
                        </button>
                    )}
                    {model.installed && model.can_redownload && (
                        <button
                            type="button"
                            onClick={() => onAction(model, "redownload")}
                            disabled={isRunning || isBlocked}
                            className="min-h-9 rounded-md bg-zinc-950 px-3 text-sm font-medium text-white disabled:cursor-not-allowed hover:cursor-pointer hover:bg-zinc-400 disabled:bg-zinc-200 disabled:text-zinc-500"
                        >
                            {isRunning && job?.action === "redownload" ? "Re-downloading" : "Re-download"}
                        </button>
                    )}
                    {model.installed && !model.can_uninstall && !model.can_redownload && (
                        <span className="rounded-md border border-zinc-200 px-3 py-2 text-xs text-zinc-500">
                            Managed only
                        </span>
                    )}
                </div>
            </div>
        </div>
    );
}
