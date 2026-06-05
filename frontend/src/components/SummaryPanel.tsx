"use client";

import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

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
    const [isInstructionOpen, setIsInstructionOpen] = useState(false);

    return (
        <div className="flex flex-col gap-4">
            <div className="flex w-full flex-wrap items-start justify-between gap-3">
                <div>
                    <h3 className="text-sm font-semibold text-zinc-900">Summary</h3>
                    <p className="mt-1 text-sm text-zinc-500">
                        Generate a concise summary from this recording transcript.
                    </p>
                </div>

                <div className="inline-flex shrink-0 rounded-lg bg-zinc-100 p-1">
                    {(["local", "byok"] as const).map((mode) => (
                        <button
                            key={mode}
                            type="button"
                            onClick={() => onModeChange(mode)}
                            className={`cursor-pointer rounded-md px-4 py-1.5 text-sm font-medium transition-all ${selectedMode === mode
                                ? "bg-white text-zinc-900 shadow-sm ring-1 ring-black/5"
                                : "text-zinc-500 hover:text-zinc-700 hover:bg-zinc-200/50"
                                } disabled:cursor-not-allowed disabled:opacity-50`}
                        >
                            {mode === "local" ? "Local" : "BYOK"}
                        </button>
                    ))}
                </div>
            </div>

            <div className="flex w-full flex-wrap items-end justify-between gap-4">
                <div className="flex flex-wrap items-end gap-3">
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
                                        const defaultModel = BYOK_PROVIDERS.find(
                                            (provider) => provider.value === newProvider,
                                        )?.defaultModel;
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
                                    className="min-h-10 w-48 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
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
                                    className="min-h-10 w-44 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
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
                                        className="min-h-10 w-56 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
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
                    className="min-h-10 shrink-0 rounded-md bg-zinc-950 px-4 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-50"
                >
                    {isLoading ? "Summarizing" : summary ? "Re-summarize" : "Summarize"}
                </button>
            </div>

            <div className="flex w-full border-b border-zinc-200 pb-5 flex-col gap-2">
                <button
                    type="button"
                    onClick={() => setIsInstructionOpen((current) => !current)}
                    disabled={isLoading}
                    className="w-fit cursor-pointer text-sm font-medium text-zinc-600 underline hover:text-zinc-950 disabled:cursor-not-allowed disabled:opacity-50"
                >
                    Custom instruction
                </button>

                {/* {formatName && (
                    <p className="text-xs font-medium text-zinc-500">
                        Format: <span className="text-zinc-800">{formatName}</span>
                    </p>
                )} */}

                {isInstructionOpen && (
                    <textarea
                        value={customInstruction}
                        onChange={(event) => onCustomInstructionChange(event.target.value)}
                        disabled={isLoading}
                        rows={3}
                        placeholder="Example: summarize this as meeting notes with decisions and action items."
                        className="w-full resize-y rounded-md border border-zinc-300 px-3 py-2 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                    />
                )}
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
                <div className="text-zinc-700">
                    <ReactMarkdown
                        remarkPlugins={[remarkGfm]}
                        components={{
                            h1: ({ children }) => (
                                <h1 className="mb-3 mt-5 text-xl font-semibold text-zinc-950">
                                    {children}
                                </h1>
                            ),
                            h2: ({ children }) => (
                                <h2 className="mb-2 mt-5 text-lg font-semibold text-zinc-950">
                                    {children}
                                </h2>
                            ),
                            h3: ({ children }) => (
                                <h3 className="mb-2 mt-4 text-base font-semibold text-zinc-950">
                                    {children}
                                </h3>
                            ),
                            p: ({ children }) => (
                                <p className="mb-3 leading-relaxed text-zinc-700">
                                    {children}
                                </p>
                            ),
                            ul: ({ children }) => (
                                <ul className="mb-3 list-disc space-y-1 pl-5">
                                    {children}
                                </ul>
                            ),
                            ol: ({ children }) => (
                                <ol className="mb-3 list-decimal space-y-1 pl-5">
                                    {children}
                                </ol>
                            ),
                            li: ({ children }) => (
                                <li className="leading-relaxed text-zinc-700">
                                    {children}
                                </li>
                            ),
                            strong: ({ children }) => (
                                <strong className="font-semibold text-zinc-950">
                                    {children}
                                </strong>
                            ),
                            a: ({ children, href }) => (
                                <a
                                    href={href}
                                    target="_blank"
                                    rel="noreferrer"
                                    className="font-medium text-zinc-950 underline"
                                >
                                    {children}
                                </a>
                            ),
                            code: ({ children }) => (
                                <code className="rounded bg-zinc-100 px-1 py-0.5 text-sm text-zinc-900">
                                    {children}
                                </code>
                            ),
                        }}
                    >
                        {summary}
                    </ReactMarkdown>
                </div>
            )}
        </div>
    );
}

function deviceLabel(device: RuntimeDevice): string {
    if (device === "auto") return "Auto";
    return device.toUpperCase();
}
