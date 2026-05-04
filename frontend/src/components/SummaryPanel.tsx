"use client";

import { RuntimeDevice, RuntimeDevices, SummaryModel } from "@/src/api/sonaApi";

interface Props {
    summary: string;
    isLoading: boolean;
    selectedModel: SummaryModel;
    onModelChange: (model: SummaryModel) => void;
    selectedDevice: RuntimeDevice;
    onDeviceChange: (device: RuntimeDevice) => void;
    runtimeDevices: RuntimeDevices;
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

export default function SummaryPanel({
    summary,
    isLoading,
    selectedModel,
    onModelChange,
    selectedDevice,
    onDeviceChange,
    runtimeDevices,
    onSummarize,
    canSummarize,
}: Props) {
    const selectedModelDescription = SUMMARY_MODELS.find(
        (model) => model.value === selectedModel,
    )?.description;

    return (
        <div className="flex flex-col gap-4">
            <div className="flex flex-wrap items-start justify-between gap-4">
                <div>
                    <h3 className="text-sm font-semibold text-zinc-900">Summary</h3>
                    <p className="mt-1 text-sm text-zinc-500">
                        Generate a concise summary from this recording transcript.
                    </p>
                </div>

                <div className="flex flex-wrap items-center justify-end gap-3">
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

            {selectedModelDescription && (
                <p className="break-all text-xs text-zinc-500">{selectedModelDescription}</p>
            )}

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
