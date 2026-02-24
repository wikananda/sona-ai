"use client";

import { useState } from "react";
import AudioUploader from "@/src/components/AudioUploader";
import TranscriptPanel from "@/src/components/TranscriptPanel";
import SummaryPanel from "@/src/components/SummaryPanel";
import { transcribeAudio, summarizeTranscript, SpeakerSegment } from "@/src/api/sonaApi";
import sanitizeTranscript from "@/src/utils/sanitizeTranscript";

const DEFAULT_LANG = {
  label: "English",
  code: "en",
  flag: "🇺🇸",
}

export default function Home() {
  // Upload form state
  const [file, setFile] = useState<File | null>(null);
  const [selectedLang, setSelectedLang] = useState(DEFAULT_LANG);
  const [minSpeakers, setMinSpeakers] = useState<number | "">("");
  const [maxSpeakers, setMaxSpeakers] = useState<number | "">("");
  const [autoDetect, setAutoDetect] = useState(true);

  // Result state
  const [transcript, setTranscript] = useState<SpeakerSegment[]>([]);
  const [summary, setSummary] = useState<string>("");

  // Loading states
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [isSummarizing, setIsSummarizing] = useState(false);

  const handleTranscribe = async () => {
    if (!file) return;

    setIsTranscribing(true);
    setTranscript([]);

    try {
      const segments = await transcribeAudio({
        file,
        language: selectedLang.code,
        minSpeakers: minSpeakers,
        maxSpeakers: maxSpeakers,
      });
      setTranscript(segments);
    } catch (error) {
      console.error("Transcription failed:", error);
    } finally {
      setIsTranscribing(false);
    }
  };

  const handleSummarize = async () => {
    if (!transcript.length) return;

    setIsSummarizing(true);
    setSummary("");

    try {
      const summary = await summarizeTranscript({
        text: sanitizeTranscript(transcript),
      })
      setSummary(summary);
    } catch (error) {
      console.error("Summarization failed:", error);
    } finally {
      setIsSummarizing(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center">
      <div className="h-48 flex items-center">
        <h1 className="text-4xl font-bold">Sona AI</h1>
      </div>

      <div className="flex flex-col gap-8 w-full max-w-2xl mb-8">
        <AudioUploader
          file={file}
          onFileChange={setFile}
          selectedLang={selectedLang}
          onLangChange={setSelectedLang}
          minSpeakers={minSpeakers}
          maxSpeakers={maxSpeakers}
          autoDetect={autoDetect}
          onMinSpeakersChange={setMinSpeakers}
          onMaxSpeakersChange={setMaxSpeakers}
          onAutoDetectChange={setAutoDetect}
          onSubmit={handleTranscribe}
          isLoading={isTranscribing}
        />

        {transcript.length > 0 && (
          <button
            onClick={handleSummarize}
            disabled={isSummarizing}
            className="bg-blue-600 text-white px-4 py-2 rounded-lg hover:bg-blue-700 hover:cursor-pointer disabled:bg-blue-200 disabled:cursor-not-allowed"
          >
            {isSummarizing ? "Summarizing..." : "Summarize"}
          </button>
        )}

        <TranscriptPanel segments={transcript} />
        <SummaryPanel summary={summary} isLoading={isSummarizing} />
      </div>
    </main>
  )
}