"use client";

import { FormEvent, useState } from "react";
import {
    BYOKSummarySettings,
    chatWithRecording,
    RecordingChatMessage,
} from "@/src/api/sonaApi";
import { BYOK_PROVIDERS } from "@/src/utils/constants";

interface Props {
    recordingId: string;
    canChat: boolean;
    byokSettings?: BYOKSummarySettings;
    isBYOKConfigured: boolean;
    onOpenSettings: () => void;
}

export default function RecordingChatPanel({
    recordingId,
    canChat,
    byokSettings,
    isBYOKConfigured,
    onOpenSettings,
}: Props) {
    const [messages, setMessages] = useState<RecordingChatMessage[]>([]);
    const [question, setQuestion] = useState("");
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState("");

    const canSend =
        canChat &&
        !isLoading &&
        Boolean(question.trim()) &&
        isBYOKConfigured &&
        Boolean(byokSettings);

    const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
        event.preventDefault();
        if (!canSend || !byokSettings) return;

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
                byok: byokSettings,
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

            <div className="flex flex-wrap items-center gap-3">
                <div className="rounded-md border border-zinc-200 bg-zinc-50 px-3 py-2 text-sm text-zinc-700">
                    {isBYOKConfigured && byokSettings
                        ? `${providerLabel(byokSettings.provider)} / ${byokSettings.model}`
                        : "API settings required"}
                </div>
                <button
                    type="button"
                    onClick={onOpenSettings}
                    disabled={isLoading}
                    className="min-h-10 rounded-md border border-zinc-300 px-3 text-sm font-medium text-zinc-700 hover:border-zinc-400 hover:text-zinc-950 disabled:cursor-not-allowed disabled:opacity-50"
                >
                    Settings
                </button>
            </div>

            {!isBYOKConfigured && (
                <div className="rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-sm text-amber-900">
                    Add API settings before chatting with this recording.
                </div>
            )}

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

function providerLabel(provider: BYOKSummarySettings["provider"]): string {
    return BYOK_PROVIDERS.find((item) => item.value === provider)?.label ?? provider;
}
