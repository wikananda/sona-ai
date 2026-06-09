"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import {
    BYOKProvider,
    ModelJob,
    ModelJobAction,
    RuntimeModel,
    getRuntimeModelJob,
    getRuntimeModels,
    redownloadRuntimeModel,
    startRuntimeModelDownload,
    uninstallRuntimeModel,
} from "@/src/api/sonaApi";
import { BYOK_PROVIDERS } from "@/src/utils/constants";
import {
    BYOKEntry,
    BYOKEntryDraft,
    BYOKSettingsState,
    byokEntryLabel,
    clearAllKeys,
    createBYOKEntry,
    isBYOKEntryConfigured,
    providerDefaultModel,
    providerLabel,
} from "@/src/hooks/useBYOKSettings";

type SettingsTab = "api" | "models";
type ConfirmableModelAction = Extract<ModelJobAction, "uninstall" | "redownload">;

interface PendingModelAction {
    model: RuntimeModel;
    action: ConfirmableModelAction;
}

interface PendingApiEntryDelete {
    entry: BYOKEntry;
}

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
    const [jobs, setJobs] = useState<Record<string, ModelJob>>({});
    const [modelError, setModelError] = useState<string | null>(null);
    const [isLoadingModels, setIsLoadingModels] = useState(false);
    const [pendingModelAction, setPendingModelAction] = useState<PendingModelAction | null>(null);
    const [editingEntry, setEditingEntry] = useState<BYOKEntry | null>(null);
    const [isAddingEntry, setIsAddingEntry] = useState(false);
    const [pendingApiEntryDelete, setPendingApiEntryDelete] =
        useState<PendingApiEntryDelete | null>(null);

    const activeDownloadJobs = useMemo(
        () => Object.values(jobs).filter((job) => isActiveJob(job)),
        [jobs],
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
                activeDownloadJobs.map((job) => getRuntimeModelJob(job.job_id)),
            );
            setJobs((current) => {
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

    const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        onSave({
            rememberKeys: draft.rememberKeys,
            entries: draft.entries.map(trimEntry),
        });
    };

    const handleClearSavedKeys = () => {
        setDraft((current) => clearAllKeys(current));
        onClearSavedKeys();
    };

    const handleModelAction = async (model: RuntimeModel, action: ModelJobAction) => {
        if (action === "uninstall" || action === "redownload") {
            setPendingModelAction({ model, action });
            return;
        }

        await startModelAction(model, action);
    };

    const startModelAction = async (model: RuntimeModel, action: ModelJobAction) => {
        setModelError(null);
        try {
            const job = await startModelJob(model.id, action);
            setJobs((current) => ({
                ...current,
                [job.job_id]: job,
            }));
            void loadModels();
        } catch (error) {
            setModelError(error instanceof Error ? error.message : `Failed to start ${action}.`);
        }
    };

    const handleConfirmModelAction = async () => {
        if (!pendingModelAction) return;

        const nextAction = pendingModelAction;
        setPendingModelAction(null);
        await startModelAction(nextAction.model, nextAction.action);
    };

    const handleSaveApiEntry = (entryDraft: BYOKEntryDraft, entryId?: string) => {
        const nextEntry: BYOKEntry = {
            id: entryId ?? createBYOKEntry().id,
            provider: entryDraft.provider,
            apiKey: entryDraft.apiKey.trim(),
            model: entryDraft.model.trim(),
            baseUrl: entryDraft.provider === "custom" ? entryDraft.baseUrl.trim() : "",
        };

        setDraft((current) => {
            if (entryId) {
                return {
                    ...current,
                    entries: current.entries.map((entry) =>
                        entry.id === entryId ? nextEntry : entry,
                    ),
                };
            }

            return {
                ...current,
                entries: [...current.entries, nextEntry],
            };
        });
        setEditingEntry(null);
        setIsAddingEntry(false);
    };

    const handleConfirmDeleteApiEntry = () => {
        if (!pendingApiEntryDelete) return;

        const deleteId = pendingApiEntryDelete.entry.id;
        setDraft((current) => ({
            ...current,
            entries: current.entries.filter((entry) => entry.id !== deleteId),
        }));
        setPendingApiEntryDelete(null);
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
                            onAdd={() => setIsAddingEntry(true)}
                            onEdit={setEditingEntry}
                            onDelete={(entry) => setPendingApiEntryDelete({ entry })}
                            onRememberKeysChange={updateRememberKeys}
                        />
                    ) : (
                        <ModelSettings
                            models={models}
                            jobs={jobs}
                            error={modelError}
                            isLoading={isLoadingModels}
                            onRefresh={loadModels}
                            onAction={handleModelAction}
                        />
                    )}
                </div>

                <div className="flex flex-wrap justify-between gap-3 border-t border-zinc-200 px-5 py-4">
                    {activeTab === "api" ? (
                        <>
                            <div className="flex flex-wrap gap-3">
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
            {pendingModelAction && (
                <ConfirmModelActionModal
                    model={pendingModelAction.model}
                    action={pendingModelAction.action}
                    onCancel={() => setPendingModelAction(null)}
                    onConfirm={() => void handleConfirmModelAction()}
                />
            )}
            {(isAddingEntry || editingEntry) && (
                <ApiEntryFormModal
                    entry={editingEntry}
                    onCancel={() => {
                        setEditingEntry(null);
                        setIsAddingEntry(false);
                    }}
                    onSave={handleSaveApiEntry}
                />
            )}
            {pendingApiEntryDelete && (
                <ConfirmApiEntryDeleteModal
                    entry={pendingApiEntryDelete.entry}
                    onCancel={() => setPendingApiEntryDelete(null)}
                    onConfirm={handleConfirmDeleteApiEntry}
                />
            )}
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
    onAdd,
    onEdit,
    onDelete,
    onRememberKeysChange,
}: {
    draft: BYOKSettingsState;
    onAdd: () => void;
    onEdit: (entry: BYOKEntry) => void;
    onDelete: (entry: BYOKEntry) => void;
    onRememberKeysChange: (rememberKeys: boolean) => void;
}) {
    return (
        <div className="flex flex-col gap-4">
            <div className="flex flex-wrap items-start justify-between gap-3">
                <div>
                    <h3 className="text-sm font-semibold text-zinc-950">
                        API provider presets
                    </h3>
                    <p className="mt-1 text-sm text-zinc-500">
                        Save OpenAI-compatible API presets for summary and chat.
                    </p>
                </div>
                <button
                    type="button"
                    onClick={onAdd}
                    className="min-h-9 rounded-md bg-zinc-950 px-3 text-sm font-medium text-white"
                >
                    + Add
                </button>
            </div>

            <div className="flex flex-col gap-3">
                {draft.entries.length === 0 ? (
                    <div className="rounded-md border border-zinc-200 px-3 py-6 text-center text-sm text-zinc-500">
                        No API provider presets yet.
                    </div>
                ) : (
                    draft.entries.map((entry) => (
                        <ApiEntryRow
                            key={entry.id}
                            entry={entry}
                            onEdit={() => onEdit(entry)}
                            onDelete={() => onDelete(entry)}
                        />
                    ))
                )}
            </div>

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

function ApiEntryRow({
    entry,
    onEdit,
    onDelete,
}: {
    entry: BYOKEntry;
    onEdit: () => void;
    onDelete: () => void;
}) {
    const isConfigured = isBYOKEntryConfigured(entry);

    return (
        <div className="rounded-md border border-zinc-200 p-3">
            <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                        <h4 className="text-sm font-semibold text-zinc-950">
                            {byokEntryLabel(entry)}
                        </h4>
                        <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${
                            isConfigured
                                ? "bg-emerald-100 text-emerald-800"
                                : "bg-amber-100 text-amber-800"
                        }`}
                        >
                            {isConfigured ? "Ready" : "Incomplete"}
                        </span>
                    </div>
                    <p className="mt-1 text-xs text-zinc-500">
                        {providerLabel(entry.provider)}
                    </p>
                    {entry.provider === "custom" && (
                        <p className="mt-1 break-all text-xs text-zinc-400">
                            Base URL: {entry.baseUrl || "Missing"}
                        </p>
                    )}
                </div>
                <div className="flex gap-2">
                    <button
                        type="button"
                        onClick={onEdit}
                        className="min-h-9 rounded-md border border-zinc-300 px-3 text-sm font-medium text-zinc-700 hover:border-zinc-400 hover:text-zinc-950"
                    >
                        Edit
                    </button>
                    <button
                        type="button"
                        onClick={onDelete}
                        className="min-h-9 rounded-md border border-red-200 px-3 text-sm font-medium text-red-700 hover:border-red-300 hover:text-red-800"
                    >
                        Delete
                    </button>
                </div>
            </div>
        </div>
    );
}

function ApiEntryFormModal({
    entry,
    onCancel,
    onSave,
}: {
    entry: BYOKEntry | null;
    onCancel: () => void;
    onSave: (draft: BYOKEntryDraft, entryId?: string) => void;
}) {
    const [draft, setDraft] = useState<BYOKEntryDraft>(() => ({
        provider: entry?.provider ?? "openai",
        apiKey: entry?.apiKey ?? "",
        model: entry?.model ?? providerDefaultModel(entry?.provider ?? "openai"),
        baseUrl: entry?.baseUrl ?? "",
    }));
    const [error, setError] = useState("");

    const updateProvider = (provider: BYOKProvider) => {
        setDraft((current) => ({
            ...current,
            provider,
            model: current.model.trim() ? current.model : providerDefaultModel(provider),
            baseUrl: provider === "custom" ? current.baseUrl : "",
        }));
        setError("");
    };

    const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        const nextDraft = {
            provider: draft.provider,
            apiKey: draft.apiKey.trim(),
            model: draft.model.trim(),
            baseUrl: draft.baseUrl.trim(),
        };

        if (!nextDraft.apiKey) {
            setError("API key is required.");
            return;
        }
        if (!nextDraft.model) {
            setError("Model name is required.");
            return;
        }
        if (nextDraft.provider === "custom" && !nextDraft.baseUrl) {
            setError("Base URL is required for custom providers.");
            return;
        }

        onSave(nextDraft, entry?.id);
    };

    return (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/40 px-4">
            <form
                onSubmit={handleSubmit}
                className="w-full max-w-lg rounded-lg bg-white shadow-xl"
            >
                <div className="border-b border-zinc-200 px-5 py-4">
                    <h3 className="text-base font-semibold text-zinc-950">
                        {entry ? "Edit API preset" : "Add API preset"}
                    </h3>
                    <p className="mt-1 text-sm text-zinc-500">
                        Configure an OpenAI-compatible provider preset.
                    </p>
                </div>

                <div className="flex flex-col gap-4 px-5 py-4">
                    <label className="flex flex-col gap-1">
                        <span className="text-xs font-medium text-zinc-500">
                            Provider
                        </span>
                        <select
                            value={draft.provider}
                            onChange={(event) => updateProvider(event.target.value as BYOKProvider)}
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
                            value={draft.apiKey}
                            onChange={(event) => {
                                setDraft((current) => ({ ...current, apiKey: event.target.value }));
                                setError("");
                            }}
                            placeholder="sk-..."
                            className="min-h-10 rounded-md border border-zinc-300 px-3 text-sm outline-none focus:border-zinc-900"
                        />
                    </label>

                    <label className="flex flex-col gap-1">
                        <span className="text-xs font-medium text-zinc-500">
                            Model name
                        </span>
                        <input
                            type="text"
                            value={draft.model}
                            onChange={(event) => {
                                setDraft((current) => ({ ...current, model: event.target.value }));
                                setError("");
                            }}
                            placeholder="gpt-4o-mini"
                            className="min-h-10 rounded-md border border-zinc-300 px-3 text-sm outline-none focus:border-zinc-900"
                        />
                    </label>

                    {draft.provider === "custom" && (
                        <label className="flex flex-col gap-1">
                            <span className="text-xs font-medium text-zinc-500">
                                Base URL
                            </span>
                            <input
                                type="text"
                                value={draft.baseUrl}
                                onChange={(event) => {
                                    setDraft((current) => ({ ...current, baseUrl: event.target.value }));
                                    setError("");
                                }}
                                placeholder="https://.../v1"
                                className="min-h-10 rounded-md border border-zinc-300 px-3 text-sm outline-none focus:border-zinc-900"
                            />
                        </label>
                    )}

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
                        className="min-h-10 rounded-md border border-zinc-300 px-4 text-sm font-medium text-zinc-700 hover:border-zinc-400 hover:text-zinc-950"
                    >
                        Cancel
                    </button>
                    <button
                        type="submit"
                        className="min-h-10 rounded-md bg-zinc-950 px-4 text-sm font-medium text-white"
                    >
                        {entry ? "Save preset" : "Add preset"}
                    </button>
                </div>
            </form>
        </div>
    );
}

function ConfirmApiEntryDeleteModal({
    entry,
    onCancel,
    onConfirm,
}: {
    entry: BYOKEntry;
    onCancel: () => void;
    onConfirm: () => void;
}) {
    return (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/40 px-4">
            <div className="w-full max-w-lg rounded-lg bg-white shadow-xl">
                <div className="border-b border-zinc-200 px-5 py-4">
                    <h3 className="text-base font-semibold text-zinc-950">
                        Delete API preset
                    </h3>
                    <p className="mt-1 text-sm text-zinc-500">
                        This removes the saved provider preset from this browser.
                    </p>
                </div>

                <div className="px-5 py-4">
                    <div className="rounded-md border border-zinc-200 bg-zinc-50 px-3 py-3">
                        <p className="text-sm font-medium text-zinc-950">
                            {byokEntryLabel(entry)}
                        </p>
                        <p className="mt-1 text-xs text-zinc-500">
                            {providerLabel(entry.provider)}
                        </p>
                    </div>
                </div>

                <div className="flex justify-end gap-3 border-t border-zinc-200 px-5 py-4">
                    <button
                        type="button"
                        onClick={onCancel}
                        className="min-h-10 rounded-md border border-zinc-300 px-4 text-sm font-medium text-zinc-700 hover:border-zinc-400 hover:text-zinc-950"
                    >
                        Cancel
                    </button>
                    <button
                        type="button"
                        onClick={onConfirm}
                        className="min-h-10 rounded-md bg-red-700 px-4 text-sm font-medium text-white hover:bg-red-800"
                    >
                        Delete
                    </button>
                </div>
            </div>
        </div>
    );
}

function ModelSettings({
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
                            onAction={onAction}
                        />
                    ))
                )}
            </div>
        </div>
    );
}

function ConfirmModelActionModal({
    model,
    action,
    onCancel,
    onConfirm,
}: {
    model: RuntimeModel;
    action: ConfirmableModelAction;
    onCancel: () => void;
    onConfirm: () => void;
}) {
    const isRedownload = action === "redownload";

    return (
        <div className="fixed inset-0 z-[60] flex items-center justify-center bg-black/40 px-4">
            <div className="w-full max-w-lg rounded-lg bg-white shadow-xl">
                <div className="border-b border-zinc-200 px-5 py-4">
                    <h3 className="text-base font-semibold text-zinc-950">
                        {isRedownload ? "Re-download model" : "Uninstall model"}
                    </h3>
                    <p className="mt-1 text-sm text-zinc-500">
                        {isRedownload
                            ? "This will remove the current cached files before downloading the model again."
                            : "This will remove the cached model files from local storage."}
                    </p>
                </div>

                <div className="flex flex-col gap-3 px-5 py-4">
                    <div className="rounded-md border border-zinc-200 bg-zinc-50 px-3 py-3">
                        <p className="text-sm font-medium text-zinc-950">
                            {model.label}
                        </p>
                        <p className="mt-1 text-xs text-zinc-500">
                            {model.type} / {model.environment}
                        </p>
                        <p className="mt-2 break-all text-xs text-zinc-500">
                            Cache: {model.cache_path}
                        </p>
                    </div>

                    <div className="rounded-md border border-amber-200 bg-amber-50 px-3 py-3 text-sm text-amber-900">
                        {isRedownload
                            ? "The model cache will be deleted first, then Sona will fetch a fresh copy."
                            : "The model cache will be deleted. The next use will require downloading the model again."}
                    </div>
                </div>

                <div className="flex justify-end gap-3 border-t border-zinc-200 px-5 py-4">
                    <button
                        type="button"
                        onClick={onCancel}
                        className="min-h-10 rounded-md border border-zinc-300 px-4 text-sm font-medium text-zinc-700 hover:border-zinc-400 hover:text-zinc-950"
                    >
                        Cancel
                    </button>
                    <button
                        type="button"
                        onClick={onConfirm}
                        className={`min-h-10 rounded-md px-4 text-sm font-medium text-white ${
                            isRedownload
                                ? "bg-zinc-950"
                                : "bg-red-700 hover:bg-red-800"
                        }`}
                    >
                        {isRedownload ? "Re-download" : "Uninstall"}
                    </button>
                </div>
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
    const message = job?.message ?? model.error ?? model.management_note ?? modelStatusLabel(model.status);
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
                            className="min-h-9 rounded-md border border-zinc-300 px-3 text-sm font-medium text-zinc-700 hover:border-zinc-400 hover:text-zinc-950 disabled:cursor-not-allowed disabled:bg-zinc-100 disabled:text-zinc-400"
                        >
                            {isRunning && job?.action === "uninstall" ? "Uninstalling" : "Uninstall"}
                        </button>
                    )}
                    {model.installed && model.can_redownload && (
                        <button
                            type="button"
                            onClick={() => onAction(model, "redownload")}
                            disabled={isRunning || isBlocked}
                            className="min-h-9 rounded-md bg-zinc-950 px-3 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-zinc-200 disabled:text-zinc-500"
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

function trimEntry(entry: BYOKEntry): BYOKEntry {
    return {
        ...entry,
        apiKey: entry.apiKey.trim(),
        model: entry.model.trim(),
        baseUrl: entry.provider === "custom" ? entry.baseUrl.trim() : "",
    };
}

function isActiveJob(job: ModelJob): boolean {
    return job.status === "queued" || job.status === "running";
}

function modelStatusLabel(status: RuntimeModel["status"]): string {
    if (status === "installed") return "Installed";
    if (status === "running") return "Working";
    if (status === "failed") return "Failed";
    return "Missing";
}

function statusClassName(status: RuntimeModel["status"]): string {
    if (status === "installed") return "bg-emerald-100 text-emerald-800";
    if (status === "running") return "bg-blue-100 text-blue-800";
    if (status === "failed") return "bg-red-100 text-red-800";
    return "bg-zinc-100 text-zinc-600";
}

function modelJobLabel(job?: ModelJob): string {
    if (!job) return "Working";
    if (job.action === "uninstall") return "Uninstalling";
    if (job.action === "redownload") return "Re-downloading";
    return "Downloading";
}

function modelJobStageLabel(stage: ModelJob["stage"]): string {
    if (stage === "preparing") return "Preparing";
    if (stage === "downloading") return "Downloading";
    if (stage === "removing") return "Removing";
    if (stage === "verifying") return "Verifying";
    if (stage === "done") return "Done";
    if (stage === "failed") return "Failed";
    return "Queued";
}

async function startModelJob(modelId: string, action: ModelJobAction): Promise<ModelJob> {
    if (action === "uninstall") {
        return uninstallRuntimeModel(modelId);
    }
    if (action === "redownload") {
        return redownloadRuntimeModel(modelId);
    }
    return startRuntimeModelDownload(modelId);
}
