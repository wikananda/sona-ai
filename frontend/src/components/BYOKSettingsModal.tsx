"use client";

import { FormEvent, useState } from "react";
import { BYOKProvider } from "@/src/api/sonaApi";
import { BYOK_PROVIDERS } from "@/src/utils/constants";
import {
    BYOKProviderSettings,
    BYOKSettingsState,
    defaultBYOKSettings,
} from "@/src/hooks/useBYOKSettings";

interface Props {
    settings: BYOKSettingsState;
    onSave: (settings: BYOKSettingsState) => void;
    onClose: () => void;
}

export default function BYOKSettingsModal({
    settings,
    onSave,
    onClose,
}: Props) {
    const [draft, setDraft] = useState<BYOKSettingsState>(settings);

    const selectedProvider = draft.selectedProvider;
    const selectedSettings = draft.providers[selectedProvider];

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
            selectedProvider: draft.selectedProvider,
            providers: trimSettings(draft.providers),
        });
    };

    return (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/40 px-4">
            <form
                onSubmit={handleSubmit}
                className="w-full max-w-xl rounded-lg bg-white shadow-xl"
            >
                <div className="flex items-start justify-between gap-4 border-b border-zinc-200 px-5 py-4">
                    <div>
                        <h2 className="text-base font-semibold text-zinc-950">
                            API settings
                        </h2>
                        <p className="mt-1 text-sm text-zinc-500">
                            Stored only in this browser. Used for BYOK summary and chat.
                        </p>
                    </div>
                    <button
                        type="button"
                        onClick={onClose}
                        className="rounded-md px-2 py-1 text-xl leading-none text-zinc-500 hover:text-zinc-950"
                        aria-label="Close API settings"
                    >
                        x
                    </button>
                </div>

                <div className="flex flex-col gap-4 p-5">
                    <label className="flex flex-col gap-1">
                        <span className="text-xs font-medium text-zinc-500">
                            Provider
                        </span>
                        <select
                            value={selectedProvider}
                            onChange={(event) =>
                                updateSelectedProvider(event.target.value as BYOKProvider)
                            }
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
                                updateSelectedSettings({ apiKey: event.target.value })
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
                                updateSelectedSettings({ model: event.target.value })
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
                                    updateSelectedSettings({ baseUrl: event.target.value })
                                }
                                placeholder="https://.../v1"
                                className="min-h-10 rounded-md border border-zinc-300 px-3 text-sm outline-none focus:border-zinc-900"
                            />
                        </label>
                    )}

                    <div className="rounded-md border border-zinc-200 bg-zinc-50 px-3 py-2 text-xs text-zinc-600">
                        The key is saved in browser localStorage on this device only. It is sent
                        to the backend only when you run BYOK summary or chat.
                    </div>
                </div>

                <div className="flex flex-wrap justify-between gap-3 border-t border-zinc-200 px-5 py-4">
                    <button
                        type="button"
                        onClick={clearSelectedProvider}
                        className="min-h-10 rounded-md border border-zinc-300 px-4 text-sm font-medium text-zinc-700 hover:border-zinc-400 hover:text-zinc-950"
                    >
                        Clear provider
                    </button>
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
                </div>
            </form>
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
