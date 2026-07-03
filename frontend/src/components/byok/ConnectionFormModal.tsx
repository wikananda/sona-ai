"use client";

import { FormEvent, useState } from "react";
import { BYOKProvider } from "@/src/api/sonaApi";
import Modal from "@/src/components/ui/Modal";
import { BYOK_PROVIDERS } from "@/src/utils/constants";
import {
    BYOKConnection,
    BYOKConnectionDraft,
    providerLabel,
} from "@/src/hooks/useBYOKSettings";

export function ConnectionFormModal({
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
