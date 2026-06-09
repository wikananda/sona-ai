"use client";

import { useMemo, useState } from "react";
import {
    ModelJob,
    RuntimeModel,
    getRuntimeModelJob,
    startRuntimeModelDownload,
} from "@/src/api/sonaApi";

interface Props {
    models: RuntimeModel[];
    onInstalled: () => void;
    onCancel: () => void;
}

type LocalStatus = "missing" | "downloading" | "installed" | "failed";

export default function MissingSpeechModelsModal({
    models,
    onInstalled,
    onCancel,
}: Props) {
    const [statuses, setStatuses] = useState<Record<string, LocalStatus>>(() =>
        Object.fromEntries(models.map((model) => [model.id, "missing"])),
    );
    const [error, setError] = useState<string | null>(null);
    const [isDownloading, setIsDownloading] = useState(false);

    const blockedModels = useMemo(
        () => models.filter((model) => model.requires_hf_token && !model.hf_token_available),
        [models],
    );

    const handleDownload = async () => {
        setError(null);
        setIsDownloading(true);
        try {
            for (const model of models) {
                setStatuses((current) => ({ ...current, [model.id]: "downloading" }));
                const job = await startRuntimeModelDownload(model.id);
                const completedJob = await waitForJob(job.job_id);
                if (completedJob.status !== "installed") {
                    throw new Error(
                        completedJob.error ||
                        completedJob.message ||
                        `Failed to install ${model.label}.`,
                    );
                }
                setStatuses((current) => ({ ...current, [model.id]: "installed" }));
            }
            onInstalled();
        } catch (downloadError) {
            setError(
                downloadError instanceof Error
                    ? downloadError.message
                    : "Failed to download required models.",
            );
            setStatuses((current) => {
                const next = { ...current };
                for (const model of models) {
                    if (next[model.id] === "downloading") {
                        next[model.id] = "failed";
                    }
                }
                return next;
            });
        } finally {
            setIsDownloading(false);
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 px-4">
            <div className="w-full max-w-lg rounded-lg bg-white shadow-xl">
                <div className="border-b border-zinc-200 px-5 py-4">
                    <h2 className="text-base font-semibold text-zinc-950">
                        Download required models
                    </h2>
                    <p className="mt-1 text-sm text-zinc-500">
                        This action needs speech models that are not installed yet.
                    </p>
                </div>

                <div className="flex flex-col gap-3 px-5 py-4">
                    {models.map((model) => (
                        <div
                            key={model.id}
                            className="rounded-md border border-zinc-200 px-3 py-3"
                        >
                            <div className="flex items-center justify-between gap-3">
                                <div>
                                    <p className="text-sm font-medium text-zinc-950">
                                        {model.label}
                                    </p>
                                    <p className="mt-1 text-xs text-zinc-500">
                                        {model.model_names.join(", ")}
                                    </p>
                                    <p className="mt-1 text-xs text-zinc-400">
                                        {model.environment} / {model.cache_path}
                                    </p>
                                    {model.requires_hf_token && !model.hf_token_available && (
                                        <p className="mt-2 text-xs font-medium text-amber-700">
                                            Requires backend HF_TOKEN in `.env`.
                                        </p>
                                    )}
                                </div>
                                <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${statusClassName(statuses[model.id] ?? "missing")}`}>
                                    {statusLabel(statuses[model.id] ?? "missing")}
                                </span>
                            </div>
                            {statuses[model.id] === "downloading" && (
                                <div className="mt-3">
                                    <div className="mb-1 flex items-center justify-between gap-3">
                                        <span className="text-xs font-medium text-zinc-700">
                                            Downloading
                                        </span>
                                        <span className="text-xs text-zinc-500">
                                            In progress
                                        </span>
                                    </div>
                                    <div className="h-2 overflow-hidden rounded-full bg-zinc-200">
                                        <div className="h-full w-2/5 animate-pulse rounded-full bg-zinc-900" />
                                    </div>
                                </div>
                            )}
                        </div>
                    ))}

                    {error && (
                        <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-700">
                            {error}
                        </div>
                    )}
                </div>

                <div className="flex justify-end gap-3 border-t border-zinc-200 px-5 py-4">
                    <button
                        type="button"
                        onClick={onCancel}
                        disabled={isDownloading}
                        className="min-h-10 rounded-md border border-zinc-300 px-4 text-sm font-medium text-zinc-700 hover:border-zinc-400 hover:text-zinc-950 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                        Cancel
                    </button>
                    <button
                        type="button"
                        onClick={handleDownload}
                        disabled={isDownloading || blockedModels.length > 0}
                        className="min-h-10 rounded-md bg-zinc-950 px-4 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-zinc-200 disabled:text-zinc-500"
                    >
                        {isDownloading ? "Downloading" : "Download and continue"}
                    </button>
                </div>
            </div>
        </div>
    );
}

async function waitForJob(jobId: string): Promise<ModelJob> {
    while (true) {
        const job = await getRuntimeModelJob(jobId);
        if (job.status === "installed" || job.status === "failed") {
            return job;
        }
        await new Promise((resolve) => window.setTimeout(resolve, 1200));
    }
}

function statusLabel(status: LocalStatus): string {
    if (status === "downloading") return "Downloading";
    if (status === "installed") return "Installed";
    if (status === "failed") return "Failed";
    return "Missing";
}

function statusClassName(status: LocalStatus): string {
    if (status === "downloading") return "bg-blue-100 text-blue-800";
    if (status === "installed") return "bg-emerald-100 text-emerald-800";
    if (status === "failed") return "bg-red-100 text-red-800";
    return "bg-zinc-100 text-zinc-600";
}
