"use client";

import { FormEvent, useState } from "react";
import Modal from "@/src/components/ui/Modal";
import {
    BYOKConnection,
    BYOKModelPreset,
    BYOKModelPresetDraft,
    byokConnectionLabel,
    providerDefaultModel,
} from "@/src/hooks/useBYOKSettings";

export function ModelPresetFormModal({
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
