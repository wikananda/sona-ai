"use client";

interface Props {
    summary: string;
    isLoading: boolean;
}

export default function SummaryPanel({ summary, isLoading }: Props) {
    if (!summary) return null;

    return (
        <div className="w-full max-w-2xl bg-white p-6 rounded-xl shadow-lg border border-zinc-100">
            <h3 className="font-semibold text-zinc-900 border-b pb-2 text-lg">Summary:</h3>
            {isLoading ? (
                <div className="flex flex-row items-center justify-center">
                    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-zinc-500"></div>
                    <span className="ml-2 text-zinc-600">Generating summary...</span>
                </div>
            ) : (
                <p className="text-zinc-700 leading-relaxed">{summary}</p>
            )}
        </div>
    )
}