import { SpeakerSegment } from "@/src/api/sonaApi";

interface Props {
    segments: SpeakerSegment[];
}

export default function TranscriptionPanel({ segments }: Props) {
    if (segments.length === 0) return null;

    return (
        <div className="mt-4 flex flex-col gap-3 max-w-full bg-white p-6 rounded-xl shadow-lg border border-zinc-100">
            <h3 className="font-semibold text-zinc-900 border-b pb-2 text-lg">Conversation:</h3>
            <div className="max-h-[400px] overflow-y-auto flex flex-col gap-1">
                {segments.map((segment, index) => (
                    <div key={index} className="flex flex-row items-center gap-8 p-2">
                        <span className="text-xs font-bold text-blue-600 uppercase tracking-wider">
                            {segment.speaker}
                        </span>
                        <p className="text-zinc-700 leading-relaxed">{segment.text}</p>
                    </div>
                ))}
            </div>
        </div>
    )
}