import { SpeakerSegment } from "@/src/api/sonaApi";

interface Props {
    segments: SpeakerSegment[];
}

export default function TranscriptPanel({ segments }: Props) {
    if (segments.length === 0) return null;

    return (
        <div className="flex max-w-full flex-col gap-3">
            <h3 className="border-b border-zinc-200 pb-2 text-sm font-semibold text-zinc-900">
                Conversation
            </h3>
            <div className="flex max-h-[520px] flex-col gap-1 overflow-y-auto pr-2">
                {segments.map((segment, index) => (
                    <div
                        key={index}
                        className="grid grid-cols-[96px_1fr] gap-4 border-b border-zinc-100 py-3 last:border-b-0"
                    >
                        <span className="text-xs font-semibold uppercase text-blue-600">
                            {segment.speaker}
                        </span>
                        <p className="text-zinc-700 leading-relaxed">{segment.text}</p>
                    </div>
                ))}
            </div>
        </div>
    )
}
