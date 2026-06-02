"use client";

import {
    BYOKProvider,
    RuntimeDevice,
    RuntimeDevices,
    SummaryMode,
    LocalLLMModel
} from "@/src/api/sonaApi";

import {
    LOCAL_LLM_MODELS,
    BYOK_PROVIDERS
} from "@/src/utils/constants";

interface Props {
    summary: string;
    isLoading: boolean;
    selectedModel: LocalLLMModel;
    onModelChange: (model: LocalLLMModel) => void;
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
    customInstruction: string;
    onCustomInstructionChange: (instruction: string) => void;
    formatName?: string | null;
    onSummarize: () => void;
    canSummarize: boolean;
}

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
    customInstruction,
    onCustomInstructionChange,
    formatName,
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
                                            onChange={(event) => onModelChange(event.target.value as LocalLLMModel)}
                                            disabled={isLoading}
                                            className="min-h-10 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                                        >
                                            {LOCAL_LLM_MODELS.map((model) => (
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
                            <label className="flex w-full flex-col gap-1">
                                <span className="text-xs font-medium text-zinc-500">
                                    Custom instruction
                                </span>
                                {formatName && (
                                    <p className="text-xs font-medium text-zinc-500">
                                        Format: <span className="text-zinc-800">{formatName}</span>
                                    </p>
                                )}
                                <textarea
                                    value={customInstruction}
                                    onChange={(event) => onCustomInstructionChange(event.target.value)}
                                    disabled={isLoading}
                                    rows={3}
                                    placeholder="Example: summarize this as meeting notes with decisions and action items."
                                    className="w-full resize-y rounded-md border border-zinc-300 px-3 py-2 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                                />
                            </label>

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
