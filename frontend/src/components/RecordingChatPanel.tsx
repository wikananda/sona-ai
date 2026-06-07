"use client";

import { FormEvent, useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
    BYOKSummarySettings,
    chatWithRecording,
    RecordingChatMessage,
} from "@/src/api/sonaApi";
import { BYOK_PROVIDERS } from "@/src/utils/constants";
import { remarkHtmlLineBreaks } from "@/src/utils/markdownPlugins";

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
                        className={`max-w-[85%] rounded-md px-3 py-2 text-sm leading-relaxed ${message.role === "user"
                            ? "ml-auto bg-zinc-950 text-white"
                            : "mr-auto w-fit bg-white text-zinc-800 ring-1 ring-zinc-200"
                            }`}
                    >
                        {message.role === "assistant" ? (
                            <AssistantMarkdown content={message.content} />
                        ) : (
                            <span className="whitespace-pre-wrap">{message.content}</span>
                        )}
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

function AssistantMarkdown({ content }: { content: string }) {
    return (
        <div className="max-w-none text-zinc-800">
            <ReactMarkdown
                remarkPlugins={[remarkGfm, remarkHtmlLineBreaks]}
                components={{
                    h1: ({ children }) => (
                        <h1 className="mb-2 mt-3 text-base font-semibold text-zinc-950 first:mt-0">
                            {children}
                        </h1>
                    ),
                    h2: ({ children }) => (
                        <h2 className="mb-2 mt-3 text-sm font-semibold text-zinc-950 first:mt-0">
                            {children}
                        </h2>
                    ),
                    h3: ({ children }) => (
                        <h3 className="mb-1.5 mt-3 text-sm font-semibold text-zinc-950 first:mt-0">
                            {children}
                        </h3>
                    ),
                    p: ({ children }) => (
                        <p className="mb-2 last:mb-0">
                            {children}
                        </p>
                    ),
                    ul: ({ children }) => (
                        <ul className="mb-2 list-disc space-y-1 pl-5 last:mb-0">
                            {children}
                        </ul>
                    ),
                    ol: ({ children }) => (
                        <ol className="mb-2 list-decimal space-y-1 pl-5 last:mb-0">
                            {children}
                        </ol>
                    ),
                    li: ({ children }) => (
                        <li className="pl-0.5">
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
                        <code className="rounded bg-zinc-100 px-1 py-0.5 text-[0.85em] text-zinc-900">
                            {children}
                        </code>
                    ),
                    table: ({ children }) => (
                        <div className="my-2 overflow-x-auto rounded-md border border-zinc-200">
                            <table className="min-w-full divide-y divide-zinc-200 text-sm">
                                {children}
                            </table>
                        </div>
                    ),
                    thead: ({ children }) => (
                        <thead className="bg-zinc-50">
                            {children}
                        </thead>
                    ),
                    tbody: ({ children }) => (
                        <tbody className="divide-y divide-zinc-100 bg-white">
                            {children}
                        </tbody>
                    ),
                    tr: ({ children }) => (
                        <tr>
                            {children}
                        </tr>
                    ),
                    th: ({ children }) => (
                        <th className="px-3 py-2 text-left align-top text-xs font-semibold uppercase tracking-wide text-zinc-600">
                            {children}
                        </th>
                    ),
                    td: ({ children }) => (
                        <td className="whitespace-pre-line px-3 py-2 align-top leading-relaxed text-zinc-700">
                            {children}
                        </td>
                    ),
                }}
            >
                {content}
            </ReactMarkdown>
        </div>
    );
}
