"use client";

import {
    BYOKConnection,
    BYOKModelPreset,
    BYOKSettingsState,
    byokConnectionLabel,
    byokResolvedModelPresetLabel,
    isBYOKConnectionConfigured,
    isBYOKModelPresetConfigured,
    providerLabel,
} from "@/src/hooks/useBYOKSettings";

export function ApiProviderSettings({
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
