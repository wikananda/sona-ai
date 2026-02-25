"use client";

import { useState } from "react";

interface Language {
    label: string;
    code: string;
    flag: string;
}

interface Props {
    file: File | null;
    onFileChange: (file: File | null) => void;
    selectedLang: Language;
    onLangChange: (lang: Language) => void;
    minSpeakers: number | "";
    maxSpeakers: number | "";
    autoDetect: boolean;
    onMinSpeakersChange: (value: number | "") => void;
    onMaxSpeakersChange: (value: number | "") => void;
    onAutoDetectChange: (value: boolean) => void;
    onSubmit: () => void;
    isLoading: boolean;
}

const LANGUAGES: Language[] = [
    { label: "Auto detect", code: "None", flag: "🔍" },
    { label: "English", code: "en", flag: "🇺🇸" },
    { label: "Indonesian", code: "id", flag: "🇮🇩" },
];

export default function AudioUploader({
    file, onFileChange,
    selectedLang, onLangChange,
    minSpeakers, maxSpeakers, autoDetect,
    onMinSpeakersChange, onMaxSpeakersChange, onAutoDetectChange,
    onSubmit, isLoading,
}: Props) {
    // State for Language Menu
    const [isLangMenuOpen, setIsLangMenuOpen] = useState(false);

    return (
        <div className="flex flex-col gap-4 w-full max-w-full bg-white p-6 rounded-xl shadow-lg border border-zinc-100">
            <h2 className="text-xl font-bold text-zinc-800">Audio Transcription</h2>

            {/* Audio File Upload */}
            <div className="flex flex-col gap-2">
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
                            onFileChange(e.target.files[0]);
                        }
                    }}
                />
            </div>

            {/* Language Selection */}
            <div className="flex flex-row items-center gap-4 relative">
                <label className="text-sm font-medium text-zinc-600">Language</label>
                <div className="relative">
                    <button
                        onClick={() => setIsLangMenuOpen(!isLangMenuOpen)}
                        type="button"
                        className="inline-flex w-48 justify-between items-center rounded-lg border border-gray-300 bg-white px-4 py-2 text-sm font-medium text-gray-700 hover:cursor-pointer hover:bg-gray-100 focus:outline-none"
                    >
                        <div className="flex items-center gap-2">
                            <span className="text-base">{selectedLang.flag}</span>
                            <span>{selectedLang.label}</span>
                        </div>
                        <svg className={`ml-2 h-4 w-4 transition-transform ${isLangMenuOpen ? 'rotate-180' : ''}`} xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24" stroke="currentColor">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M19 9l-7 7-7-7" />
                        </svg>
                    </button>

                    {isLangMenuOpen && (
                        <div className="absolute left-0 z-20 mt-2 w-48 origin-top-left text-sm font-medium text-gray-700 rounded-lg bg-white shadow-xl ring-1 ring-gray-300 ring-opacity-5">
                            <div className="py-1">
                                {LANGUAGES.map((lang) => (
                                    <button
                                        key={lang.code}
                                        onClick={() => {
                                            onLangChange(lang);
                                            setIsLangMenuOpen(false);
                                        }}
                                        className="flex items-center gap-2 w-full px-4 py-2 text-left text-sm hover:bg-zinc-100 hover:rounded-md hover:cursor-pointer transition-colors"
                                    >
                                        <span className="text-base">{lang.flag}</span>
                                        <span>{lang.label}</span>
                                    </button>
                                ))}
                            </div>
                        </div>
                    )}
                </div>
            </div>

            {/* Speaker Detection settings */}
            <div className="flex flex-col gap-4 text-sm text-gray-700">
                <div className="flex flex-row gap-4">
                    <div className="flex-1 flex flex-col gap-1">
                        <label htmlFor="minSpeakers" className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Min Speakers</label>
                        <input
                            type="number"
                            id="minSpeakers"
                            min="1"
                            disabled={autoDetect}
                            value={minSpeakers}
                            onChange={(e) => onMinSpeakersChange(e.target.value === "" ? "" : Math.max(1, parseInt(e.target.value)))}
                            placeholder="Optional"
                            className="w-full px-3 py-2 bg-white border border-zinc-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-zinc-200 transition-all disabled:cursor-not-allowed disabled:opacity-50"
                        />
                    </div>
                    <div className="flex-1 flex flex-col gap-1">
                        <label htmlFor="maxSpeakers" className="text-xs font-semibold text-zinc-400 uppercase tracking-wider">Max Speakers</label>
                        <input
                            type="number"
                            id="maxSpeakers"
                            min="1"
                            disabled={autoDetect}
                            value={maxSpeakers}
                            onChange={(e) => onMaxSpeakersChange(e.target.value === "" ? "" : Math.max(1, parseInt(e.target.value)))}
                            placeholder="Optional"
                            className="w-full px-3 py-2 bg-white border border-zinc-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-zinc-200 transition-all disabled:cursor-not-allowed disabled:opacity-50"
                        />
                    </div>
                </div>
                <div className="flex items-center gap-2">
                    <input
                        type="checkbox"
                        id="autoDetect"
                        checked={autoDetect}
                        onChange={(e) => onAutoDetectChange(e.target.checked)}
                        className="w-4 h-4 rounded border-zinc-300 text-black focus:ring-black"
                    />
                    <label htmlFor="autoDetect" className="text-sm text-zinc-500">Auto-detect number of speakers</label>
                </div>
            </div>

            {/* Upload Button */}
            <button
                onClick={onSubmit}
                disabled={isLoading || !file}
                className="bg-black text-white py-3 rounded-lg font-medium transition-all active:scale-95 disabled:opacity-50 disabled:cursor-not-allowed hover:bg-zinc-500 hover:cursor-pointer"
            >
                {isLoading ? "Transcribing..." : "Upload & Transcribe"}
            </button>
        </div>
    );
}