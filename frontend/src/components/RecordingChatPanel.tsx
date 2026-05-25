"use client";

import { FormEvent, useState } from "react";
import {
    BYOKProvider,
    chatWithRecording,
    RecordingChatMessage,
} from "@/src/api/sonaApi";
import { BYOK_PROVIDERS } from "@/src/utils/constants";

interface Props {
    recordingId: string;
    canChat: boolean;
}

export default function RecordingChatPanel({ recordingId, canChat }: Props) {
    const [messages, setMessages] = useState<RecordingChatMessage[]>([]);
    const [question, setQuestion] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [provider, setProvider] = useState<BYOKProvider>("openai");
    const [apiKey, setApiKey] = useState("");
    const [model, setModel] = useState("gpt-4o-mini");
    const [baseUrl, setBaseUrl] = useState("");
    const [error, setError] = useState("");

    const canSend =
        canChat &&
        !isLoading &&
        Boolean(question.trim()) &&
        Boolean(apiKey.trim()) &&
        Boolean(model.trim()) &&
        (provider !== "custom" || Boolean(baseUrl.trim()));

    const handleProviderChange = (nextProvider: BYOKProvider) => {
        setProvider(nextProvider);

        const defaultModel = BYOK_PROVIDERS.find(
            (item) => item.value === nextProvider,
        )?.defaultModel;
        if (defaultModel !== undefined) {
            setModel(defaultModel);
        }
    };

    const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        if (!canSend) return;

        const nextQuestion = question.trim();
        const userMessage: RecordingChatMessage = {
            role: "user",
            content: nextQuestion,
        };
        const nextMessages = [...messages, userMessage];

        setMessages(nextMessages);
        setQuestion("");
        setError("");
        setIsLoading(true);

        try {
            const answer = await chatWithRecording(recordingId, {
                question: nextQuestion,
                history: messages.slice(-8),
                byok: {
                    provider,
                    apiKey: apiKey.trim(),
                    model: model.trim(),
                    baseUrl: baseUrl.trim() || undefined,
                },
            });

            setMessages([
                ...nextMessages,
                {
                    role: "assistant",
                    content: answer,
                },
            ]);
        } catch (err) {
            setError(err instanceof Error ? err.message : "Failed to chat with recording");
        } finally {
            setIsLoading(false);
        }
    };

    return (
        <div className="flex flex-col gap-4">
            <div>
                <h3 className="text-sm font-semibold text-zinc-900">Chat</h3>
                <p className="mt-1 text-sm text-zinc-500">
                    Ask questions about this recording transcript.
                </p>
            </div>

            <div className="flex flex-wrap items-end gap-3">
                <label className="flex flex-col gap-1">
                    <span className="text-xs font-medium text-zinc-500">Provider</span>
                    <select
                        value={provider}
                        onChange={(event) =>
                            handleProviderChange(event.target.value as BYOKProvider)
                        }
                        disabled={isLoading}
                        className="min-h-10 rounded-md border border-zinc-300 bg-white px-3 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                    >
                        {BYOK_PROVIDERS.map((item) => (
                            <option key={item.value} value={item.value}>
                                {item.label}
                            </option>
                        ))}
                    </select>
                </label>

                <label className="flex flex-col gap-1">
                    <span className="text-xs font-medium text-zinc-500">API Key</span>
                    <input
                        type="password"
                        value={apiKey}
                        onChange={(event) => setApiKey(event.target.value)}
                        disabled={isLoading}
                        placeholder="sk-..."
                        className="min-h-10 w-44 rounded-md border border-zinc-300 px-3 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                    />
                </label>

                <label className="flex flex-col gap-1">
                    <span className="text-xs font-medium text-zinc-500">Model</span>
                    <input
                        type="text"
                        value={model}
                        onChange={(event) => setModel(event.target.value)}
                        disabled={isLoading}
                        className="min-h-10 w-44 rounded-md border border-zinc-300 px-3 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                    />
                </label>

                {provider === "custom" && (
                    <label className="flex flex-col gap-1">
                        <span className="text-xs font-medium text-zinc-500">Base URL</span>
                        <input
                            type="text"
                            value={baseUrl}
                            onChange={(event) => setBaseUrl(event.target.value)}
                            disabled={isLoading}
                            placeholder="https://.../v1"
                            className="min-h-10 w-56 rounded-md border border-zinc-300 px-3 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                        />
                    </label>
                )}
            </div>

            {error && (
                <div className="rounded-md border border-red-200 bg-red-50 px-3 py-2 text-sm text-red-800">
                    {error}
                </div>
            )}

            <div className="flex min-h-72 flex-col gap-3 rounded-md border border-zinc-200 bg-zinc-50 p-4">
                {messages.length === 0 && !isLoading && (
                    <p className="text-sm text-zinc-500">
                        {canChat
                            ? "Ask a question about this recording."
                            : "Transcript is required before chatting."}
                    </p>
                )}

                {messages.map((message, index) => (
                    <div
                        key={`${message.role}-${index}`}
                        className={`max-w-[85%] whitespace-pre-wrap rounded-md px-3 py-2 text-sm leading-relaxed ${message.role === "user"
                            ? "ml-auto bg-zinc-950 text-white"
                            : "mr-auto bg-white text-zinc-800 ring-1 ring-zinc-200"
                            }`}
                    >
                        {message.content}
                    </div>
                ))}

                {isLoading && (
                    <div className="mr-auto rounded-md bg-white px-3 py-2 text-sm text-zinc-500 ring-1 ring-zinc-200">
                        Thinking...
                    </div>
                )}
            </div>

            <form onSubmit={handleSubmit} className="flex gap-2">
                <input
                    type="text"
                    value={question}
                    onChange={(event) => setQuestion(event.target.value)}
                    disabled={isLoading || !canChat}
                    placeholder={
                        canChat
                            ? "Ask about this recording..."
                            : "Transcript is required before chatting"
                    }
                    className="min-h-11 min-w-0 flex-1 rounded-md border border-zinc-300 px-3 text-sm outline-none focus:border-zinc-900 disabled:cursor-not-allowed disabled:opacity-50"
                />
                <button
                    type="submit"
                    disabled={!canSend}
                    className="min-h-11 rounded-md bg-zinc-950 px-4 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-40"
                >
                    Send
                </button>
            </form>
        </div>
    );
}
