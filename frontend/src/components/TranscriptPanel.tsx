import { SpeakerSegment } from "@/src/api/sonaApi";

interface Props {
    segments: SpeakerSegment[];
}

export default function TranscriptPanel({ segments }: Props) {
    if (segments.length === 0) return null;

    return (
        <div className="flex max-w-full flex-col">
            <div className="flex max-h-[520px] flex-col gap-1 overflow-y-auto pr-2">
                {segments.map((segment, index) => (
                    <div
                        key={index}
                        className="grid grid-cols-[112px_1fr] items-start gap-4 border-b border-zinc-100 py-3 last:border-b-0"
                    >
                        <span className="text-sm font-bold leading-relaxed text-zinc-700">
                            {segment.speaker}
                        </span>
                        <p className="text-zinc-700 leading-relaxed">{segment.text}</p>
                    </div>
                ))}
            </div>
        </div>
    )
}
