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
import Modal from "@/src/components/ui/Modal";
import { BYOK_PROVIDERS } from "@/src/utils/constants";
import { getErrorMessage } from "@/src/utils/errorHandling";
import { usePolling } from "@/src/hooks/usePolling";
import {
    BYOKConnection,
    BYOKConnectionDraft,
    BYOKModelPreset,
    BYOKModelPresetDraft,
    BYOKSettingsState,
    byokConnectionLabel,
    byokResolvedModelPresetLabel,
    clearAllKeys,
    createBYOKConnection,
    createBYOKModelPreset,
    isBYOKConnectionConfigured,
    isBYOKModelPresetConfigured,
    providerDefaultModel,
    providerLabel,
} from "@/src/hooks/useBYOKSettings";

type SettingsTab = "api" | "models";
type ConfirmableModelAction = Extract<ModelJobAction, "uninstall" | "redownload">;

interface PendingModelAction {
    model: RuntimeModel;
    action: ConfirmableModelAction;
}

interface PendingConnectionDelete {
    connection: BYOKConnection;
    linkedPresetCount: number;
}

interface PendingModelPresetDelete {
    preset: BYOKModelPreset;
    label: string;
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
    const [editingConnection, setEditingConnection] = useState<BYOKConnection | null>(null);
    const [isAddingConnection, setIsAddingConnection] = useState(false);
    const [pendingConnectionDelete, setPendingConnectionDelete] =
        useState<PendingConnectionDelete | null>(null);
    const [editingModelPreset, setEditingModelPreset] = useState<BYOKModelPreset | null>(null);
    const [isAddingModelPreset, setIsAddingModelPreset] = useState(false);
    const [pendingModelPresetDelete, setPendingModelPresetDelete] =
        useState<PendingModelPresetDelete | null>(null);

    const activeDownloadJobs = useMemo(
        () => Object.values(jobs).filter((job) => isActiveJob(job)),
        [jobs],
    );

    const connectionMap = useMemo(
        () => new Map(draft.connections.map((connection) => [connection.id, connection])),
        [draft.connections],
    );

    const loadModels = useCallback(async () => {
        setIsLoadingModels(true);
        setModelError(null);
        try {
            setModels(await getRuntimeModels());
        } catch (error) {
            setModelError(getErrorMessage(error, "Failed to load models."));
        } finally {
            setIsLoadingModels(false);
        }
    }, []);

    useEffect(() => {
        if (activeTab === "models") {
            void loadModels();
        }
    }, [activeTab, loadModels]);

    usePolling(async () => {
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
    }, 1500, activeTab === "models" && activeDownloadJobs.length > 0);

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
            connections: draft.connections.map(trimConnection),
            modelPresets: draft.modelPresets.map(trimModelPreset),
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
            setModelError(getErrorMessage(error, `Failed to start ${action}.`));
        }
    };

    const handleConfirmModelAction = async () => {
        if (!pendingModelAction) return;

        const nextAction = pendingModelAction;
        setPendingModelAction(null);
        await startModelAction(nextAction.model, nextAction.action);
    };

    const handleSaveConnection = (connectionDraft: BYOKConnectionDraft, connectionId?: string) => {
        const nextConnection: BYOKConnection = {
            id: connectionId ?? createBYOKConnection().id,
            name: connectionDraft.name.trim(),
            provider: connectionDraft.provider,
            apiKey: connectionDraft.apiKey.trim(),
            baseUrl: connectionDraft.provider === "custom" ? connectionDraft.baseUrl.trim() : "",
        };

        setDraft((current) => {
            if (connectionId) {
                return {
                    ...current,
                    connections: current.connections.map((connection) =>
                        connection.id === connectionId ? nextConnection : connection,
                    ),
                };
            }

            return {
                ...current,
                connections: [...current.connections, nextConnection],
            };
        });
        setEditingConnection(null);
        setIsAddingConnection(false);
    };

    const handleSaveModelPreset = (presetDraft: BYOKModelPresetDraft, presetId?: string) => {
        const nextPreset: BYOKModelPreset = {
            id: presetId ?? createBYOKModelPreset().id,
            connectionId: presetDraft.connectionId,
            model: presetDraft.model.trim(),
            name: presetDraft.name.trim(),
        };

        setDraft((current) => {
            if (presetId) {
                return {
                    ...current,
                    modelPresets: current.modelPresets.map((preset) =>
                        preset.id === presetId ? nextPreset : preset,
                    ),
                };
            }

            return {
                ...current,
                modelPresets: [...current.modelPresets, nextPreset],
            };
        });
        setEditingModelPreset(null);
        setIsAddingModelPreset(false);
    };

    const handleConfirmDeleteConnection = () => {
        if (!pendingConnectionDelete) return;

        const deleteId = pendingConnectionDelete.connection.id;
        setDraft((current) => ({
            ...current,
            connections: current.connections.filter((connection) => connection.id !== deleteId),
            modelPresets: current.modelPresets.filter((preset) => preset.connectionId !== deleteId),
        }));
        setPendingConnectionDelete(null);
    };

    const handleConfirmDeleteModelPreset = () => {
        if (!pendingModelPresetDelete) return;

        const deleteId = pendingModelPresetDelete.preset.id;
        setDraft((current) => ({
            ...current,
            modelPresets: current.modelPresets.filter((preset) => preset.id !== deleteId),
        }));
        setPendingModelPresetDelete(null);
    };

    return (
        <Modal>
            <form
                onSubmit={handleSubmit}
                className="flex max-h-[90vh] w-full max-w-4xl flex-col rounded-lg bg-white shadow-xl"
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
                        className="rounded-md px-2 py-1 text-xl leading-none text-zinc-500 hover:text-zinc-950 hover:cursor-pointer"
                        aria-label="Close settings"
                    >
                        x
                    </button>
                </div>

                <div className="border-b border-zinc-200 px-5 pt-4 pb-4">
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
                            connectionMap={connectionMap}
                            onAddConnection={() => setIsAddingConnection(true)}
                            onEditConnection={setEditingConnection}
                            onDeleteConnection={(connection) => {
                                const linkedPresetCount = draft.modelPresets.filter(
                                    (preset) => preset.connectionId === connection.id,
                                ).length;
                                setPendingConnectionDelete({ connection, linkedPresetCount });
                            }}
                            onAddModelPreset={() => setIsAddingModelPreset(true)}
                            onEditModelPreset={setEditingModelPreset}
                            onDeleteModelPreset={(preset) => {
                                const connection = connectionMap.get(preset.connectionId);
                                const label = connection
                                    ? byokResolvedModelPresetLabel({ ...preset, connection })
                                    : preset.name.trim() || preset.model.trim() || "Model preset";
                                setPendingModelPresetDelete({ preset, label });
                            }}
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
                                    className="min-h-10 rounded-md border border-red-200 px-4 text-sm hover:cursor-pointer font-medium text-red-700 hover:border-red-300 hover:text-red-800"
                                >
                                    Clear saved keys
                                </button>
                            </div>
                            <div className="flex gap-3">
                                <button
                                    type="button"
                                    onClick={onClose}
                                    className="min-h-10 rounded-md border border-zinc-300 px-4 text-sm hover:cursor-pointer font-medium text-zinc-700 hover:border-zinc-400 hover:text-zinc-950"
                                >
                                    Cancel
                                </button>
                                <button
                                    type="submit"
                                    className="min-h-10 rounded-md bg-zinc-950 px-4 text-sm hover:cursor-pointer hover:bg-zinc-400 font-medium text-white"
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
                                className="min-h-10 rounded-md bg-zinc-950 px-4 text-sm font-medium text-white hover:cursor-pointer hover:bg-zinc-400"
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
            {(isAddingConnection || editingConnection) && (
                <ConnectionFormModal
                    connection={editingConnection}
                    onCancel={() => {
                        setEditingConnection(null);
                        setIsAddingConnection(false);
                    }}
                    onSave={handleSaveConnection}
                />
            )}
            {(isAddingModelPreset || editingModelPreset) && (
                <ModelPresetFormModal
                    preset={editingModelPreset}
                    connections={draft.connections}
                    onCancel={() => {
                        setEditingModelPreset(null);
                        setIsAddingModelPreset(false);
                    }}
                    onSave={handleSaveModelPreset}
                />
            )}
            {pendingConnectionDelete && (
                <ConfirmConnectionDeleteModal
                    connection={pendingConnectionDelete.connection}
                    linkedPresetCount={pendingConnectionDelete.linkedPresetCount}
                    onCancel={() => setPendingConnectionDelete(null)}
                    onConfirm={handleConfirmDeleteConnection}
                />
            )}
            {pendingModelPresetDelete && (
                <ConfirmModelPresetDeleteModal
                    label={pendingModelPresetDelete.label}
                    onCancel={() => setPendingModelPresetDelete(null)}
                    onConfirm={handleConfirmDeleteModelPreset}
                />
            )}
        </Modal>
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
            className={`min-h-10 flex-1 rounded-md px-3 text-sm font-medium ${active
                ? "bg-white text-zinc-950 shadow-sm"
                : "text-zinc-500 hover:text-zinc-950 hover:cursor-pointer"
                }`}
        >
            {children}
        </button>
    );
}

function ApiProviderSettings({
    draft,
    connectionMap,
    onAddConnection,
    onEditConnection,
    onDeleteConnection,
    onAddModelPreset,
    onEditModelPreset,
    onDeleteModelPreset,
    onRememberKeysChange,
}: {
    draft: BYOKSettingsState;
    connectionMap: Map<string, BYOKConnection>;
    onAddConnection: () => void;
    onEditConnection: (connection: BYOKConnection) => void;
    onDeleteConnection: (connection: BYOKConnection) => void;
    onAddModelPreset: () => void;
    onEditModelPreset: (preset: BYOKModelPreset) => void;
    onDeleteModelPreset: (preset: BYOKModelPreset) => void;
    onRememberKeysChange: (rememberKeys: boolean) => void;
}) {
    return (
        <div className="flex flex-col gap-6">
            <div>
                <h3 className="text-sm font-semibold text-zinc-950">
                    API providers
                </h3>
                <p className="mt-1 text-sm text-zinc-500">
                    Save credentials once as reusable connections, then create model presets that point to them.
                </p>
            </div>

            <section className="flex flex-col gap-3">
                <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                        <h4 className="text-sm font-semibold text-zinc-950">
                            Connections
                        </h4>
                        <p className="mt-1 text-sm text-zinc-500">
                            A connection stores provider, API key, and base URL.
                        </p>
                    </div>
                    <button
                        type="button"
                        onClick={onAddConnection}
                        className="min-h-9 rounded-md bg-zinc-950 px-3 text-sm font-medium text-white hover:cursor-pointer hover:bg-zinc-400"
                    >
                        + Add connection
                    </button>
                </div>

                <div className="flex flex-col gap-3">
                    {draft.connections.length === 0 ? (
                        <div className="rounded-md border border-zinc-200 px-3 py-6 text-center text-sm text-zinc-500">
                            No connections yet.
                        </div>
                    ) : (
                        draft.connections.map((connection) => (
                            <ConnectionRow
                                key={connection.id}
                                connection={connection}
                                linkedPresetCount={draft.modelPresets.filter(
                                    (preset) => preset.connectionId === connection.id,
                                ).length}
                                onEdit={() => onEditConnection(connection)}
                                onDelete={() => onDeleteConnection(connection)}
                            />
                        ))
                    )}
                </div>
            </section>

            <section className="flex flex-col gap-3">
                <div className="flex flex-wrap items-start justify-between gap-3">
                    <div>
                        <h4 className="text-sm font-semibold text-zinc-950">
                            Model presets
                        </h4>
                        <p className="mt-1 text-sm text-zinc-500">
                            These are the flat model choices shown in Summary and Chat.
                        </p>
                    </div>
                    <button
                        type="button"
                        onClick={onAddModelPreset}
                        disabled={draft.connections.length === 0}
                        className="min-h-9 rounded-md bg-zinc-950 px-3 text-sm font-medium text-white disabled:cursor-not-allowed disabled:bg-zinc-200 disabled:text-zinc-500 hover:cursor-pointer hover:bg-zinc-400"
                    >
                        + Add model preset
                    </button>
                </div>

                <div className="flex flex-col gap-3">
                    {draft.modelPresets.length === 0 ? (
                        <div className="rounded-md border border-zinc-200 px-3 py-6 text-center text-sm text-zinc-500">
                            {draft.connections.length === 0
                                ? "Create a connection before adding model presets."
                                : "No model presets yet."}
                        </div>
                    ) : (
                        draft.modelPresets.map((preset) => (
                            <ModelPresetRow
                                key={preset.id}
                                preset={preset}
                                connection={connectionMap.get(preset.connectionId)}
                                onEdit={() => onEditModelPreset(preset)}
                                onDelete={() => onDeleteModelPreset(preset)}
                            />
                        ))
                    )}
                </div>
            </section>

            <label className="flex items-start gap-2 rounded-md border border-zinc-200 bg-zinc-50 px-3 py-2 hover:cursor-pointer">
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

function ConnectionRow({
    connection,
    linkedPresetCount,
    onEdit,
    onDelete,
}: {
    connection: BYOKConnection;
    linkedPresetCount: number;
    onEdit: () => void;
    onDelete: () => void;
}) {
    const isConfigured = isBYOKConnectionConfigured(connection);

    return (
        <div className="rounded-md border border-zinc-200 p-3">
            <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                        <h5 className="text-sm font-semibold text-zinc-950">
                            {byokConnectionLabel(connection)}
                        </h5>
                        <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${isConfigured
                            ? "bg-emerald-100 text-emerald-800"
                            : "bg-amber-100 text-amber-800"
                            }`}
                        >
                            {isConfigured ? "Ready" : "Incomplete"}
                        </span>
                    </div>
                    <p className="mt-1 text-xs text-zinc-500">
                        {providerLabel(connection.provider)}
                    </p>
                    {connection.provider === "custom" && (
                        <p className="mt-1 break-all text-xs text-zinc-400">
                            Base URL: {connection.baseUrl || "Missing"}
                        </p>
                    )}
                    <p className="mt-2 text-xs text-zinc-500">
                        {linkedPresetCount} model preset{linkedPresetCount === 1 ? "" : "s"}
                    </p>
                </div>
                <div className="flex gap-2">
                    <button
                        type="button"
                        onClick={onEdit}
                        className="min-h-9 rounded-md border border-zinc-300 px-3 text-sm hover:cursor-pointer font-medium text-zinc-700 hover:border-zinc-400 hover:text-zinc-950"
                    >
                        Edit
                    </button>
                    <button
                        type="button"
                        onClick={onDelete}
                        className="min-h-9 rounded-md border border-red-200 px-3 text-sm hover:cursor-pointer font-medium text-red-700 hover:border-red-300 hover:text-red-800"
                    >
                        Delete
                    </button>
                </div>
            </div>
        </div>
    );
}

function ModelPresetRow({
    preset,
    connection,
    onEdit,
    onDelete,
}: {
    preset: BYOKModelPreset;
    connection?: BYOKConnection;
    onEdit: () => void;
    onDelete: () => void;
}) {
    const isConfigured = connection
        ? isBYOKConnectionConfigured(connection) && isBYOKModelPresetConfigured(preset)
        : false;

    const label = connection
        ? byokResolvedModelPresetLabel({ ...preset, connection })
        : preset.name.trim() || preset.model.trim() || "Model preset";

    return (
        <div className="rounded-md border border-zinc-200 p-3">
            <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0 flex-1">
                    <div className="flex flex-wrap items-center gap-2">
                        <h5 className="text-sm font-semibold text-zinc-950">
                            {label}
                        </h5>
                        <span className={`rounded-full px-2 py-0.5 text-xs font-medium ${isConfigured
                            ? "bg-emerald-100 text-emerald-800"
                            : "bg-amber-100 text-amber-800"
                            }`}
                        >
                            {isConfigured ? "Ready" : "Incomplete"}
                        </span>
                    </div>
                    <p className="mt-1 text-xs text-zinc-500">
                        {connection ? byokConnectionLabel(connection) : "Missing connection"}
                    </p>
                    <p className="mt-1 text-xs text-zinc-400">
                        Model: {preset.model || "Missing"}
                    </p>
                </div>
                <div className="flex gap-2">
                    <button
                        type="button"
                        onClick={onEdit}
                        className="min-h-9 rounded-md border border-zinc-300 px-3 text-sm hover:cursor-pointer font-medium text-zinc-700 hover:border-zinc-400 hover:text-zinc-950"
                    >
                        Edit
                    </button>
                    <button
                        type="button"
                        onClick={onDelete}
                        className="min-h-9 rounded-md border border-red-200 px-3 text-sm hover:cursor-pointer font-medium text-red-700 hover:border-red-300 hover:text-red-800"
                    >
                        Delete
                    </button>
                </div>
            </div>
        </div>
    );
}

function ConnectionFormModal({
    connection,
    onCancel,
    onSave,
}: {
    connection: BYOKConnection | null;
    onCancel: () => void;
    onSave: (draft: BYOKConnectionDraft, connectionId?: string) => void;
}) {
    const [draft, setDraft] = useState<BYOKConnectionDraft>(() => ({
        name: connection?.name ?? providerLabel(connection?.provider ?? "openai"),
        provider: connection?.provider ?? "openai",
        apiKey: connection?.apiKey ?? "",
        baseUrl: connection?.baseUrl ?? "",
    }));
    const [error, setError] = useState("");

    const updateProvider = (provider: BYOKProvider) => {
        setDraft((current) => ({
            ...current,
            provider,
            name: current.name.trim() ? current.name : providerLabel(provider),
            baseUrl: provider === "custom" ? current.baseUrl : "",
        }));
        setError("");
    };

    const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        const nextDraft = {
            name: draft.name.trim(),
            provider: draft.provider,
            apiKey: draft.apiKey.trim(),
            baseUrl: draft.baseUrl.trim(),
        };

        if (!nextDraft.name) {
            setError("Connection name is required.");
            return;
        }
        if (!nextDraft.apiKey) {
            setError("API key is required.");
            return;
        }
        if (nextDraft.provider === "custom" && !nextDraft.baseUrl) {
            setError("Base URL is required for custom providers.");
            return;
        }

        onSave(nextDraft, connection?.id);
    };

    return (
        <Modal zClassName="z-[60]">
            <form
                onSubmit={handleSubmit}
                className="w-full max-w-lg rounded-lg bg-white shadow-xl"
            >
                <div className="border-b border-zinc-200 px-5 py-4">
                    <h3 className="text-base font-semibold text-zinc-950">
                        {connection ? "Edit connection" : "Add connection"}
                    </h3>
                    <p className="mt-1 text-sm text-zinc-500">
                        Configure reusable provider credentials.
                    </p>
                </div>

                <div className="flex flex-col gap-4 px-5 py-4">
                    <label className="flex flex-col gap-1">
                        <span className="text-xs font-medium text-zinc-500">
                            Connection name
                        </span>
                        <input
                            type="text"
                            value={draft.name}
                            onChange={(event) => {
                                setDraft((current) => ({ ...current, name: event.target.value }));
                                setError("");
                            }}
                            placeholder="Groq personal"
                            className="min-h-10 rounded-md border border-zinc-300 px-3 text-sm outline-none focus:border-zinc-900"
                        />
                    </label>

                    <label className="flex flex-col gap-1">
                        <span className="text-xs font-medium text-zinc-500">
                            Provider
                        </span>
                        <select
                            value={draft.provider}
                            onChange={(event) => updateProvider(event.target.value as BYOKProvider)}
                            className="min-h-10 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none focus:border-zinc-900 hover:cursor-pointer"
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
                        className="min-h-10 rounded-md border border-zinc-300 px-4 text-sm hover:cursor-pointer font-medium text-zinc-700 hover:border-zinc-400 hover:text-zinc-950"
                    >
                        Cancel
                    </button>
                    <button
                        type="submit"
                        className="min-h-10 rounded-md bg-zinc-950 px-4 text-sm hover:cursor-pointer hover:bg-zinc-400 font-medium text-white"
                    >
                        {connection ? "Save connection" : "Add connection"}
                    </button>
                </div>
            </form>
        </Modal>
    );
}

function ModelPresetFormModal({
    preset,
    connections,
    onCancel,
    onSave,
}: {
    preset: BYOKModelPreset | null;
    connections: BYOKConnection[];
    onCancel: () => void;
    onSave: (draft: BYOKModelPresetDraft, presetId?: string) => void;
}) {
    const initialConnection = connections.find((connection) => connection.id === preset?.connectionId) ?? connections[0];
    const [draft, setDraft] = useState<BYOKModelPresetDraft>(() => ({
        connectionId: preset?.connectionId ?? initialConnection?.id ?? "",
        model: preset?.model ?? providerDefaultModel(initialConnection?.provider ?? "openai"),
        name: preset?.name ?? "",
    }));
    const [error, setError] = useState("");

    const updateConnectionId = (connectionId: string) => {
        const connection = connections.find((item) => item.id === connectionId);
        setDraft((current) => ({
            ...current,
            connectionId,
            model: current.model.trim() ? current.model : providerDefaultModel(connection?.provider ?? "openai"),
        }));
        setError("");
    };

    const handleSubmit = (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();

        const nextDraft = {
            connectionId: draft.connectionId,
            model: draft.model.trim(),
            name: draft.name.trim(),
        };

        if (!nextDraft.connectionId) {
            setError("Connection is required.");
            return;
        }
        if (!nextDraft.model) {
            setError("Model name is required.");
            return;
        }

        onSave(nextDraft, preset?.id);
    };

    return (
        <Modal zClassName="z-[60]">
            <form
                onSubmit={handleSubmit}
                className="w-full max-w-lg rounded-lg bg-white shadow-xl"
            >
                <div className="border-b border-zinc-200 px-5 py-4">
                    <h3 className="text-base font-semibold text-zinc-950 ">
                        {preset ? "Edit model preset" : "Add model preset"}
                    </h3>
                    <p className="mt-1 text-sm text-zinc-500">
                        Create a reusable model choice for Summary and Chat.
                    </p>
                </div>

                <div className="flex flex-col gap-4 px-5 py-4">
                    <label className="flex flex-col gap-1">
                        <span className="text-xs font-medium text-zinc-500">
                            Connection
                        </span>
                        <select
                            value={draft.connectionId}
                            onChange={(event) => updateConnectionId(event.target.value)}
                            className="min-h-10 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none hover:cursor-pointer focus:border-zinc-900"
                        >
                            {connections.map((connection) => (
                                <option key={connection.id} value={connection.id}>
                                    {byokConnectionLabel(connection)}
                                </option>
                            ))}
                        </select>
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
                            placeholder="llama-3.1-8b-instant"
                            className="min-h-10 rounded-md border border-zinc-300 px-3 text-sm outline-none focus:border-zinc-900"
                        />
                    </label>

                    <label className="flex flex-col gap-1">
                        <span className="text-xs font-medium text-zinc-500">
                            Display name (optional)
                        </span>
                        <input
                            type="text"
                            value={draft.name}
                            onChange={(event) => {
                                setDraft((current) => ({ ...current, name: event.target.value }));
                                setError("");
                            }}
                            placeholder="Groq fast summary"
                            className="min-h-10 rounded-md border border-zinc-300 px-3 text-sm outline-none focus:border-zinc-900"
                        />
                    </label>

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
                        className="min-h-10 rounded-md border border-zinc-300 px-4 text-sm font-medium hover:cursor-pointer text-zinc-700 hover:border-zinc-400 hover:text-zinc-950"
                    >
                        Cancel
                    </button>
                    <button
                        type="submit"
                        className="min-h-10 rounded-md bg-zinc-950 px-4 text-sm font-medium text-white hover:cursor-pointer hover:bg-zinc-400"
                    >
                        {preset ? "Save preset" : "Add preset"}
                    </button>
                </div>
            </form>
        </Modal>
    );
}

function ConfirmConnectionDeleteModal({
    connection,
    linkedPresetCount,
    onCancel,
    onConfirm,
}: {
    connection: BYOKConnection;
    linkedPresetCount: number;
    onCancel: () => void;
    onConfirm: () => void;
}) {
    return (
        <Modal zClassName="z-[60]">
            <div className="w-full max-w-lg rounded-lg bg-white shadow-xl">
                <div className="border-b border-zinc-200 px-5 py-4">
                    <h3 className="text-base font-semibold text-zinc-950">
                        Delete connection
                    </h3>
                    <p className="mt-1 text-sm text-zinc-500">
                        This removes the saved connection from this browser.
                    </p>
                </div>

                <div className="flex flex-col gap-3 px-5 py-4">
                    <div className="rounded-md border border-zinc-200 bg-zinc-50 px-3 py-3">
                        <p className="text-sm font-medium text-zinc-950">
                            {byokConnectionLabel(connection)}
                        </p>
                        <p className="mt-1 text-xs text-zinc-500">
                            {providerLabel(connection.provider)}
                        </p>
                    </div>

                    <div className="rounded-md border border-amber-200 bg-amber-50 px-3 py-3 text-sm text-amber-900">
                        Deleting this connection will also delete {linkedPresetCount} linked model preset{linkedPresetCount === 1 ? "" : "s"}.
                    </div>
                </div>

                <div className="flex justify-end gap-3 border-t border-zinc-200 px-5 py-4">
                    <button
                        type="button"
                        onClick={onCancel}
                        className="min-h-10 rounded-md border border-zinc-300 px-4 text-sm font-medium hover:cursor-pointer text-zinc-700 hover:border-zinc-400 hover:text-zinc-950"
                    >
                        Cancel
                    </button>
                    <button
                        type="button"
                        onClick={onConfirm}
                        className="min-h-10 rounded-md bg-red-700 px-4 text-sm font-medium hover:cursor-pointer text-white hover:bg-red-800"
                    >
                        Delete
                    </button>
                </div>
            </div>
        </Modal>
    );
}

function ConfirmModelPresetDeleteModal({
    label,
    onCancel,
    onConfirm,
}: {
    label: string;
    onCancel: () => void;
    onConfirm: () => void;
}) {
    return (
        <Modal zClassName="z-[60]">
            <div className="w-full max-w-lg rounded-lg bg-white shadow-xl">
                <div className="border-b border-zinc-200 px-5 py-4">
                    <h3 className="text-base font-semibold text-zinc-950">
                        Delete model preset
                    </h3>
                    <p className="mt-1 text-sm text-zinc-500">
                        This removes the saved model preset from this browser.
                    </p>
                </div>

                <div className="px-5 py-4">
                    <div className="rounded-md border border-zinc-200 bg-zinc-50 px-3 py-3">
                        <p className="text-sm font-medium text-zinc-950">
                            {label}
                        </p>
                    </div>
                </div>

                <div className="flex justify-end gap-3 border-t border-zinc-200 px-5 py-4">
                    <button
                        type="button"
                        onClick={onCancel}
                        className="min-h-10 rounded-md border border-zinc-300 px-4 text-sm hover:cursor-pointer font-medium text-zinc-700 hover:border-zinc-400 hover:text-zinc-950"
                    >
                        Cancel
                    </button>
                    <button
                        type="button"
                        onClick={onConfirm}
                        className="min-h-10 rounded-md bg-red-700 px-4 text-sm hover:cursor-pointer font-medium text-white hover:bg-red-800"
                    >
                        Delete
                    </button>
                </div>
            </div>
        </Modal>
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
        <Modal zClassName="z-[60]">
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
                            {modelTypeLabel(model.type)} model
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
                        className="min-h-10 rounded-md border border-zinc-300 px-4 text-sm font-medium text-zinc-700 hover:cursor-pointer hover:border-zinc-400 hover:text-zinc-950"
                    >
                        Cancel
                    </button>
                    <button
                        type="button"
                        onClick={onConfirm}
                        className={`min-h-10 rounded-md px-4 text-sm font-medium text-white hover:cursor-pointer ${isRedownload
                            ? "bg-zinc-950 hover:bg-zinc-400"
                            : "bg-red-700 hover:bg-red-800"
                            }`}
                    >
                        {isRedownload ? "Re-download" : "Uninstall"}
                    </button>
                </div>
            </div>
        </Modal>
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

function trimConnection(connection: BYOKConnection): BYOKConnection {
    return {
        ...connection,
        name: connection.name.trim(),
        apiKey: connection.apiKey.trim(),
        baseUrl: connection.provider === "custom" ? connection.baseUrl.trim() : "",
    };
}

function trimModelPreset(preset: BYOKModelPreset): BYOKModelPreset {
    return {
        ...preset,
        connectionId: preset.connectionId.trim(),
        model: preset.model.trim(),
        name: preset.name.trim(),
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

function modelTypeLabel(type: string): string {
    if (type === "transcription") return "Transcription";
    if (type === "alignment") return "Timestamp aligner";
    if (type === "diarization") return "Speaker extraction";
    return type.charAt(0).toUpperCase() + type.slice(1);
}
