"use client";

import { FormEvent, useCallback, useEffect, useMemo, useState } from "react";
import {
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
import { getErrorMessage } from "@/src/utils/errorHandling";
import { usePolling } from "@/src/hooks/usePolling";
import {
    BYOKConnection,
    BYOKConnectionDraft,
    BYOKModelPreset,
    BYOKModelPresetDraft,
    BYOKSettingsState,
    byokResolvedModelPresetLabel,
    clearAllKeys,
    createBYOKConnection,
    createBYOKModelPreset,
} from "@/src/hooks/useBYOKSettings";
import {
    PendingConnectionDelete,
    PendingModelAction,
    PendingModelPresetDelete,
    SettingsTab,
} from "@/src/components/byok/types";
import { isActiveJob, trimConnection, trimModelPreset } from "@/src/components/byok/byokHelpers";
import { ApiProviderSettings } from "@/src/components/byok/ApiProviderSettings";
import { ConnectionFormModal } from "@/src/components/byok/ConnectionFormModal";
import { ModelPresetFormModal } from "@/src/components/byok/ModelPresetFormModal";
import {
    ConfirmConnectionDeleteModal,
    ConfirmModelActionModal,
    ConfirmModelPresetDeleteModal,
} from "@/src/components/byok/ConfirmDialogs";
import { ModelSettings } from "@/src/components/byok/ModelSettings";

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

async function startModelJob(modelId: string, action: ModelJobAction): Promise<ModelJob> {
    if (action === "uninstall") {
        return uninstallRuntimeModel(modelId);
    }
    if (action === "redownload") {
        return redownloadRuntimeModel(modelId);
    }
    return startRuntimeModelDownload(modelId);
}
