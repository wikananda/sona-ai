"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import {
    BYOKProvider,
    ModelDownloadJob,
    RuntimeModel,
    getRuntimeModelDownloadJob,
    getRuntimeModels,
    startRuntimeModelDownload,
} from "@/src/api/sonaApi";
import { BYOK_PROVIDERS } from "@/src/utils/constants";
import {
    BYOKProviderSettings,
    BYOKSettingsState,
    clearAllKeys,
    defaultBYOKSettings,
} from "@/src/hooks/useBYOKSettings";

type SettingsTab = "api" | "models";

interface Props {
    settings: BYOKSettingsState;
    onSave: (settings: BYOKSettingsState) => void;
    onClearSavedKeys: () => void;
    onClose: () => void;
}

export default function BYOKSettingsModal({
    settings,
    onSave,
    onClearSavedKeys,
    onClose,
}: Props) {
    const [activeTab, setActiveTab] = useState<SettingsTab>("api");
    const [draft, setDraft] = useState<BYOKSettingsState>(settings);
    const [models, setModels] = useState<RuntimeModel[]>([]);
    const [downloadJobs, setDownloadJobs] = useState<Record<string, ModelDownloadJob>>({});
    const [modelError, setModelError] = useState<string | null>(null);
    const [isLoadingModels, setIsLoadingModels] = useState(false);

    const selectedProvider = draft.selectedProvider;
    const selectedSettings = draft.providers[selectedProvider];

    const activeDownloadJobs = useMemo(
        () => Object.values(downloadJobs).filter((job) => isActiveJob(job)),
        [downloadJobs],
    );

    const loadModels = useCallback(async () => {
        setIsLoadingModels(true);
        setModelError(null);
        try {
            setModels(await getRuntimeModels());
        } catch (error) {
            setModelError(error instanceof Error ? error.message : "Failed to load models.");
        } finally {
            setIsLoadingModels(false);
        }
    }, []);

    useEffect(() => {
        if (activeTab === "models") {
            void loadModels();
        }
    }, [activeTab, loadModels]);

    useEffect(() => {
        if (activeTab !== "models" || activeDownloadJobs.length === 0) return;

        const interval = window.setInterval(async () => {
            const updates = await Promise.allSettled(
                activeDownloadJobs.map((job) => getRuntimeModelDownloadJob(job.job_id)),
            );
            setDownloadJobs((current) => {
                const next = { ...current };
                for (const update of updates) {
                    if (update.status === "fulfilled") {
                        next[update.value.job_id] = update.value;
                    }
                }
                return next;
            });
            void loadModels();
        }, 1500);

        return () => window.clearInterval(interval);
    }, [activeDownloadJobs, activeTab, loadModels]);

    const updateRememberKeys = (rememberKeys: boolean) => {
        setDraft((current) => ({
            ...current,
            rememberKeys,
        }));
    };

    const updateSelectedProvider = (provider: BYOKProvider) => {
        setDraft((current) => ({
            ...current,
            selectedProvider: provider,
        }));
    };

    const updateSelectedSettings = (patch: Partial<BYOKProviderSettings>) => {
        setDraft((current) => ({
            ...current,
            providers: {
                ...current.providers,
                [current.selectedProvider]: {
                    ...current.providers[current.selectedProvider],
                    ...patch,
                },
            },
        }));
    };

    const clearSelectedProvider = () => {
        const defaults = defaultBYOKSettings();
        setDraft((current) => ({
            ...current,
            providers: {
                ...current.providers,
                [current.selectedProvider]: {
                    ...defaults.providers[current.selectedProvider],
                },
            },
        }));
    };

    const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        onSave({
            rememberKeys: draft.rememberKeys,
            selectedProvider: draft.selectedProvider,
            providers: trimSettings(draft.providers),
        });
    };

    const handleClearSavedKeys = () => {
        setDraft((current) => clearAllKeys(current));
        onClearSavedKeys();
    };

    const handleDownloadModel = async (model: RuntimeModel) => {
        setModelError(null);
        try {
            const job = await startRuntimeModelDownload(model.id);
            setDownloadJobs((current) => ({
                ...current,
                [job.job_id]: job,
            }));
            void loadModels();
        } catch (error) {
            setModelError(error instanceof Error ? error.message : "Failed to start download.");
        }
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 px-4">
            <form
                onSubmit={handleSubmit}
                className="flex max-h-[90vh] w-full max-w-3xl flex-col rounded-lg bg-white shadow-xl"
            >
                <div className="flex items-start justify-between gap-4 border-b border-zinc-200 px-5 py-4">
                    <div>
                        <h2 className="text-base font-semibold text-zinc-950">
                            Settings
                        </h2>
                        <p className="mt-1 text-sm text-zinc-500">
                            Manage API providers and local speech models.
                        </p>
                    </div>
                    <button
                        type="button"
                        onClick={onClose}
                        className="rounded-md px-2 py-1 text-xl leading-none text-zinc-500 hover:text-zinc-950"
                        aria-label="Close settings"
                    >
                        x
                    </button>
                </div>

                <div className="border-b border-zinc-200 px-5 pt-4">
                    <div className="flex rounded-md bg-zinc-100 p-1">
                        <TabButton
                            active={activeTab === "api"}
                            onClick={() => setActiveTab("api")}
                        >
                            API providers
                        </TabButton>
                        <TabButton
                            active={activeTab === "models"}
                            onClick={() => setActiveTab("models")}
                        >
                            Models
                        </TabButton>
                    </div>
                </div>

                <div className="min-h-0 flex-1 overflow-y-auto p-5">
                    {activeTab === "api" ? (
                        <ApiProviderSettings
                            draft={draft}
                            selectedProvider={selectedProvider}
                            selectedSettings={selectedSettings}
                            onProviderChange={updateSelectedProvider}
                            onSelectedSettingsChange={updateSelectedSettings}
                            onRememberKeysChange={updateRememberKeys}
                        />
                    ) : (
                        <ModelSettings
                            models={models}
                            jobs={downloadJobs}
                            error={modelError}
                            isLoading={isLoadingModels}
                            onRefresh={loadModels}
                            onDownload={handleDownloadModel}
                        />
                    )}
                </div>

                <div className="flex flex-wrap justify-between gap-3 border-t border-zinc-200 px-5 py-4">
                    {activeTab === "api" ? (
                        <>
                            <div className="flex flex-wrap gap-3">
                                <button
                                    type="button"
                                    onClick={clearSelectedProvider}
                                    className="min-h-10 rounded-md border border-zinc-300 px-4 text-sm font-medium text-zinc-700 hover:border-zinc-400 hover:text-zinc-950"
                                >
                                    Clear provider
                                </button>
                                <button
                                    type="button"
                                    onClick={handleClearSavedKeys}
                                    className="min-h-10 rounded-md border border-red-200 px-4 text-sm font-medium text-red-700 hover:border-red-300 hover:text-red-800"
                                >
                                    Clear saved keys
                                </button>
                            </div>
                            <div className="flex gap-3">
                                <button
                                    type="button"
                                    onClick={onClose}
                                    className="min-h-10 rounded-md border border-zinc-300 px-4 text-sm font-medium text-zinc-700 hover:border-zinc-400 hover:text-zinc-950"
                                >
                                    Cancel
                                </button>
                                <button
                                    type="submit"
                                    className="min-h-10 rounded-md bg-zinc-950 px-4 text-sm font-medium text-white"
                                >
                                    Save settings
                                </button>
                            </div>
                        </>
                    ) : (
                        <div className="ml-auto flex gap-3">
                            <button
                                type="button"
                                onClick={onClose}
                                className="min-h-10 rounded-md bg-zinc-950 px-4 text-sm font-medium text-white"
                            >
                                Done
                            </button>
                        </div>
                    )}
                </div>
            </form>
        </div>
    );
}

function TabButton({
    active,
    children,
    onClick,
}: {
    active: boolean;
    children: string;
    onClick: () => void;
}) {
    return (
        <button
            type="button"
            onClick={onClick}
            className={`min-h-10 flex-1 rounded-md px-3 text-sm font-medium ${
                active
                    ? "bg-white text-zinc-950 shadow-sm"
                    : "text-zinc-500 hover:text-zinc-950"
            }`}
        >
            {children}
        </button>
    );
}

function ApiProviderSettings({
    draft,
    selectedProvider,
    selectedSettings,
    onProviderChange,
    onSelectedSettingsChange,
    onRememberKeysChange,
}: {
    draft: BYOKSettingsState;
    selectedProvider: BYOKProvider;
    selectedSettings: BYOKProviderSettings;
    onProviderChange: (provider: BYOKProvider) => void;
    onSelectedSettingsChange: (patch: Partial<BYOKProviderSettings>) => void;
    onRememberKeysChange: (rememberKeys: boolean) => void;
}) {
    return (
        <div className="flex flex-col gap-4">
            <label className="flex flex-col gap-1">
                <span className="text-xs font-medium text-zinc-500">
                    Provider
                </span>
                <select
                    value={selectedProvider}
                    onChange={(event) => onProviderChange(event.target.value as BYOKProvider)}
                    className="min-h-10 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none focus:border-zinc-900"
                >
                    {BYOK_PROVIDERS.map((provider) => (
                        <option key={provider.value} value={provider.value}>
                            {provider.label}
                        </option>
                    ))}
                </select>
            </label>

            <label className="flex flex-col gap-1">
                <span className="text-xs font-medium text-zinc-500">
                    API key
                </span>
                <input
                    type="password"
                    value={selectedSettings.apiKey}
                    onChange={(event) =>
                        onSelectedSettingsChange({ apiKey: event.target.value })
                    }
                    placeholder="sk-..."
                    className="min-h-10 rounded-md border border-zinc-300 px-3 text-sm outline-none focus:border-zinc-900"
                />
            </label>

            <label className="flex flex-col gap-1">
                <span className="text-xs font-medium text-zinc-500">
                    Model
                </span>
                <input
                    type="text"
                    value={selectedSettings.model}
                    onChange={(event) =>
                        onSelectedSettingsChange({ model: event.target.value })
                    }
                    placeholder="Model name"
                    className="min-h-10 rounded-md border border-zinc-300 px-3 text-sm outline-none focus:border-zinc-900"
                />
            </label>

            {selectedProvider === "custom" && (
                <label className="flex flex-col gap-1">
                    <span className="text-xs font-medium text-zinc-500">
                        Base URL
                    </span>
                    <input
                        type="text"
                        value={selectedSettings.baseUrl}
                        onChange={(event) =>
                            onSelectedSettingsChange({ baseUrl: event.target.value })
                        }
                        placeholder="https://.../v1"
                        className="min-h-10 rounded-md border border-zinc-300 px-3 text-sm outline-none focus:border-zinc-900"
                    />
                </label>
            )}

            <label className="flex items-start gap-2 rounded-md border border-zinc-200 bg-zinc-50 px-3 py-2">
                <input
                    type="checkbox"
                    checked={draft.rememberKeys}
                    onChange={(event) => onRememberKeysChange(event.target.checked)}
                    className="mt-0.5 h-4 w-4 rounded border-zinc-300 text-zinc-950 focus:ring-zinc-950"
                />
                <span className="text-xs text-zinc-600">
                    <span className="block font-medium text-zinc-800">
                        Remember API keys on this browser
                    </span>
                    {draft.rememberKeys
                        ? "Keys will be stored as browser localStorage on this device."
                        : "Keys stay only in this tab session and disappear on refresh."}
                </span>
            </label>
        </div>
    );
}

function ModelSettings({
    models,
    jobs,
    error,
    isLoading,
    onRefresh,
    onDownload,
}: {
    models: RuntimeModel[];
    jobs: Record<string, ModelDownloadJob>;
    error: string | null;
    isLoading: boolean;
    onRefresh: () => void;
    onDownload: (model: RuntimeModel) => void;
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
                    className="min-h-9 rounded-md border border-zinc-300 px-3 text-sm font-medium text-zinc-700 hover:border-zinc-400 hover:text-zinc-950"
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
                            onDownload={() => onDownload(model)}
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
    onDownload,
}: {
    model: RuntimeModel;
    job?: ModelDownloadJob;
    onDownload: () => void;
}) {
    const isRunning = model.status === "running" || Boolean(job && isActiveJob(job));
    const isBlocked = model.requires_hf_token && !model.hf_token_available;
    const message = job?.message ?? model.error ?? modelStatusLabel(model.status);

    return (
        <div className="rounded-md border border-zinc-200 p-3">
            <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                    <div className="flex flex-wrap items-center gap-2">
                        <h4 className="text-sm font-semibold text-zinc-950">
                            {model.label}
                        </h4>
                        <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${statusClassName(model.status)}`}>
                            {modelStatusLabel(model.status)}
                        </span>
                    </div>
                    <p className="mt-1 text-xs text-zinc-500">
                        {model.type} / {model.environment}
                    </p>
                    <p className="mt-1 break-all text-xs text-zinc-500">
                        {model.model_names.join(", ")}
                    </p>
                    <p className="mt-1 break-all text-xs text-zinc-400">
                        Cache: {model.cache_path}
                    </p>
                    {isBlocked && (
                        <p className="mt-2 text-xs font-medium text-amber-700">
                            Requires backend HF_TOKEN in .env.
                        </p>
                    )}
                    {message && (
                        <p className="mt-2 text-xs text-zinc-500">
                            {message}
                        </p>
                    )}
                    {job?.error && (
                        <p className="mt-2 text-xs text-red-700">
                            {job.error}
                        </p>
                    )}
                </div>
                <button
                    type="button"
                    onClick={onDownload}
                    disabled={model.installed || isRunning || isBlocked}
                    className="min-h-9 rounded-md bg-zinc-950 px-3 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-zinc-200 disabled:text-zinc-500"
                >
                    {model.installed ? "Installed" : isRunning ? "Downloading" : "Download"}
                </button>
            </div>
        </div>
    );
}

function trimSettings(
    providers: BYOKSettingsState["providers"],
): BYOKSettingsState["providers"] {
    return {
        openai: trimProvider(providers.openai),
        groq: trimProvider(providers.groq),
        openrouter: trimProvider(providers.openrouter),
        custom: trimProvider(providers.custom),
    };
}

function trimProvider(settings: BYOKProviderSettings): BYOKProviderSettings {
    return {
        apiKey: settings.apiKey.trim(),
        model: settings.model.trim(),
        baseUrl: settings.baseUrl.trim(),
    };
}

function isActiveJob(job: ModelDownloadJob): boolean {
    return job.status === "queued" || job.status === "running";
}

function modelStatusLabel(status: RuntimeModel["status"]): string {
    if (status === "installed") return "Installed";
    if (status === "running") return "Downloading";
    if (status === "failed") return "Failed";
    return "Missing";
}

function statusClassName(status: RuntimeModel["status"]): string {
    if (status === "installed") return "bg-emerald-100 text-emerald-800";
    if (status === "running") return "bg-blue-100 text-blue-800";
    if (status === "failed") return "bg-red-100 text-red-800";
    return "bg-zinc-100 text-zinc-600";
}
