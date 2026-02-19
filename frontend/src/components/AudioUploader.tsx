"use client";

import { useState } from "react";

interface SpeakerSegment {
    speaker: string;
    text: string;
    start: number;
    end: number;
}

export default function AudioUploader() {
    const [file, setFile] = useState<File | null>(null);
    const [transcript, setTranscript] = useState<SpeakerSegment[]>([]);
    const [loading, setLoading] = useState(false);

    const handleUpload = async () => {
        if (!file) return;

        const formData = new FormData();
        formData.append("file", file);

        setLoading(true);
        try {
            const response = await fetch("http://localhost:8000/transcribe", {
                method: "POST",
                body: formData,
            });

            const data = await response.json();
            // Data is now an array of segments
            setTranscript(data.transcript);
        } catch (error) {
            console.error("Error transcribing file:", error);
        }
        setLoading(false);
    };

    return (
        <div className="flex flex-col gap-4 w-full max-w-2xl bg-white p-6 rounded-xl shadow-lg border border-zinc-100">
            <h2 className="text-xl font-bold text-zinc-800">Audio Transcription</h2>

            <div className="flex flex-col gap-2">
                <label className="text-sm font-medium text-zinc-600">Choose Audio File</label>
                <input
                    type="file"
                    accept="audio/*"
                    className="block w-full text-sm text-slate-500
                        file:mr-4 file:py-2 file:px-4
                        file:rounded-full file:border-0
                        file:text-sm file:font-semibold
                        file:bg-zinc-100 file:text-zinc-700
                        hover:file:bg-zinc-300 cursor-pointer"
                    onChange={(e) => {
                        if (e.target.files) {
                            setFile(e.target.files[0]);
                        }
                    }}
                />
            </div>

            <button
                onClick={handleUpload}
                disabled={loading || !file}
                className="bg-black text-white py-3 rounded-lg font-medium transition-all active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-zinc-500 hover:cursor-pointer"
            >
                {loading ? "Transcribing with AI..." : "Upload & Transcribe"}
            </button>

            {transcript.length > 0 && (
                <div className="mt-4 flex flex-col gap-3">
                    <h3 className="font-semibold text-zinc-900 border-b pb-2 text-lg">Conversation:</h3>
                    <div className="max-h-[400px] overflow-y-auto flex flex-col gap-1">
                        {transcript.map((segment, index) => (
                            <div key={index} className="flex flex-row items-center gap-8 p-2">
                                <span className="text-xs font-bold text-blue-600 uppercase tracking-wider">
                                    {segment.speaker}
                                </span>
                                <p className="text-zinc-700 leading-relaxed">{segment.text}</p>
                            </div>
                        ))}
                    </div>
                </div>
            )}
        </div>
    );
}