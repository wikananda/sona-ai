"use client";

import {
    BYOKProvider,
    RuntimeDevice,
    RuntimeDevices,
    SummaryMode,
    SummaryModel
} from "@/src/api/sonaApi";

interface Props {
    summary: string;
    isLoading: boolean;
    selectedModel: SummaryModel;
    onModelChange: (model: SummaryModel) => void;
    selectedDevice: RuntimeDevice;
    onDeviceChange: (device: RuntimeDevice) => void;
    runtimeDevices: RuntimeDevices;
    selectedMode: SummaryMode;
    onModeChange: (mode: SummaryMode) => void;
    byokProvider: BYOKProvider;
    onBYOKProviderChange: (provider: BYOKProvider) => void;
    byokApiKey: string;
    onBYOKApiKeyChange: (apiKey: string) => void;
    byokModel: string;
    onBYOKModelChange: (model: string) => void;
    byokBaseUrl: string;
    onBYOKBaseUrlChange: (baseUrl: string) => void;
    onSummarize: () => void;
    canSummarize: boolean;
}

const SUMMARY_MODELS: { value: SummaryModel; label: string; description: string }[] = [
    {
        value: "qwen",
        label: "Qwen",
        description: "khazarai/Qwen3-4B-Qwen3.6-plus-Reasoning-Distilled-GGUF",
    },
    {
        value: "llama",
        label: "Llama",
        description: "meta-llama/Llama-3.2-3B-Instruct",
    },
    {
        value: "gemma",
        label: "Gemma",
        description: "google/gemma-4-E2B-it",
    },
];

const BYOK_PROVIDERS = [
    { value: "openai", label: "OpenAI", defaultModel: "gpt-4o-mini" },
    { value: "groq", label: "Groq", defaultModel: "llama-3.1-8b-instant" },
    { value: "openrouter", label: "OpenRouter", defaultModel: "openai/gpt-4o-mini" },
    { value: "custom", label: "Custom", defaultModel: "" },
] as const;

export default function SummaryPanel({
    summary,
    isLoading,
    selectedModel,
    onModelChange,
    selectedDevice,
    onDeviceChange,
    runtimeDevices,
    selectedMode,
    onModeChange,
    byokProvider,
    onBYOKProviderChange,
    byokApiKey,
    onBYOKApiKeyChange,
    byokModel,
    onBYOKModelChange,
    byokBaseUrl,
    onBYOKBaseUrlChange,
    onSummarize,
    canSummarize,
}: Props) {
    return (
        <div className="flex flex-col gap-4">
            <div className="flex flex-col items-start justify-between gap-4">
                <div>
                    <h3 className="text-sm font-semibold text-zinc-900">Summary</h3>
                    <p className="mt-1 text-sm text-zinc-500">
                        Generate a concise summary from this recording transcript.
                    </p>
                </div>

                <div className="flex flex-col gap-2 items-start w-full">
                    <div className="inline-flex rounded-lg bg-zinc-100 p-1 max-w-fit">
                        {(["local", "byok"] as const).map((mode) => (
                            <button
                                key={mode}
                                type="button"
                                onClick={() => onModeChange(mode)}
                                className={`cursor-pointer flex items-center justify-center rounded-md px-4 py-1.5 text-sm font-medium transition-all ${selectedMode === mode
                                    ? "bg-white text-zinc-900 shadow-sm ring-1 ring-black/5"
                                    : "text-zinc-500 hover:text-zinc-700 hover:bg-zinc-200/50"
                                    }`}
                            >
                                {mode === "local" ? "Local" : "BYOK"}
                            </button>
                        ))}
                    </div>
                    <div className="flex flex-row justify-between gap-4 w-full">
                        <div className="flex flex-wrap items-center justify-end gap-3">
                            {selectedMode === "local" && (
                                <>
                                    <div className="flex flex-col gap-1">
                                        <label htmlFor="summary-model" className="text-xs font-medium text-zinc-500">
                                            LLM
                                        </label>
                                        <select
                                            id="summary-model"
                                            value={selectedModel}
                                            onChange={(event) => onModelChange(event.target.value as SummaryModel)}
                                            disabled={isLoading}
                                            className="min-h-10 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                                        >
                                            {SUMMARY_MODELS.map((model) => (
                                                <option key={model.value} value={model.value}>
                                                    {model.label}
                                                </option>
                                            ))}
                                        </select>
                                    </div>
                                    <div className="flex flex-col gap-1">
                                        <label htmlFor="summary-device" className="text-xs font-medium text-zinc-500">
                                            Device
                                        </label>
                                        <select
                                            id="summary-device"
                                            value={selectedDevice}
                                            onChange={(event) => onDeviceChange(event.target.value as RuntimeDevice)}
                                            disabled={isLoading}
                                            className="min-h-10 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                                        >
                                            {runtimeDevices.available.map((device) => (
                                                <option key={device} value={device}>
                                                    {deviceLabel(device)}
                                                </option>
                                            ))}
                                        </select>
                                    </div>
                                </>
                            )}

                            {selectedMode === "byok" && (
                                <>
                                    <div className="flex flex-col gap-1">
                                        <label htmlFor="byok-provider" className="text-xs font-medium text-zinc-500">
                                            Provider
                                        </label>
                                        <select
                                            id="byok-provider"
                                            value={byokProvider}
                                            onChange={(event) => {
                                                const newProvider = event.target.value as BYOKProvider;
                                                onBYOKProviderChange(newProvider);
                                                const defaultModel = BYOK_PROVIDERS.find((p) => p.value === newProvider)?.defaultModel;
                                                if (defaultModel !== undefined) {
                                                    onBYOKModelChange(defaultModel);
                                                }
                                            }}
                                            disabled={isLoading}
                                            className="min-h-10 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                                        >
                                            {BYOK_PROVIDERS.map((provider) => (
                                                <option key={provider.value} value={provider.value}>
                                                    {provider.label}
                                                </option>
                                            ))}
                                        </select>
                                    </div>

                                    <div className="flex flex-col gap-1">
                                        <label htmlFor="byok-api-key" className="text-xs font-medium text-zinc-500">
                                            API Key
                                        </label>
                                        <input
                                            id="byok-api-key"
                                            type="password"
                                            value={byokApiKey}
                                            onChange={(event) => onBYOKApiKeyChange(event.target.value)}
                                            disabled={isLoading}
                                            placeholder="sk-..."
                                            className="min-h-10 w-36 sm:w-48 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                                        />
                                    </div>

                                    <div className="flex flex-col gap-1">
                                        <label htmlFor="byok-model" className="text-xs font-medium text-zinc-500">
                                            Model
                                        </label>
                                        <input
                                            id="byok-model"
                                            type="text"
                                            value={byokModel}
                                            onChange={(event) => onBYOKModelChange(event.target.value)}
                                            disabled={isLoading}
                                            placeholder="Model name"
                                            className="min-h-10 w-32 sm:w-40 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                                        />
                                    </div>

                                    {byokProvider === "custom" && (
                                        <div className="flex flex-col gap-1">
                                            <label htmlFor="byok-base-url" className="text-xs font-medium text-zinc-500">
                                                Base URL
                                            </label>
                                            <input
                                                id="byok-base-url"
                                                type="text"
                                                value={byokBaseUrl}
                                                onChange={(event) => onBYOKBaseUrlChange(event.target.value)}
                                                disabled={isLoading}
                                                placeholder="https://.../v1"
                                                className="min-h-10 w-36 sm:w-48 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                                            />
                                        </div>
                                    )}
                                </>
                            )}

                        </div>
                        <button
                            type="button"
                            onClick={onSummarize}
                            disabled={isLoading || !canSummarize}
                            className="mt-5 min-h-10 shrink-0 rounded-md bg-zinc-950 px-4 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-40"
                        >
                            {isLoading ? "Summarizing" : summary ? "Re-summarize" : "Summarize"}
                        </button>
                    </div>
                </div>
            </div>

            {!summary && !isLoading && (
                <div className="rounded-md border border-zinc-200 bg-zinc-50 p-5 text-sm text-zinc-500">
                    No summary generated yet.
                </div>
            )}

            {isLoading && (
                <div className="flex min-h-24 items-center justify-center rounded-md border border-zinc-200 bg-zinc-50">
                    <div className="h-8 w-8 animate-spin rounded-full border-b-2 border-zinc-500" />
                    <span className="ml-2 text-sm text-zinc-600">Generating summary...</span>
                </div>
            )}

            {summary && !isLoading && (
                <div className="text-zinc-700 leading-relaxed">
                    {summary}
                </div>
            )}
        </div>
    );
}

function deviceLabel(device: RuntimeDevice): string {
    if (device === "auto") return "Auto";
    return device.toUpperCase();
}
