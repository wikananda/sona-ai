import { SpeakerSegment } from "@/src/api/sonaApi";

export default function sanitizeTranscript(segments: SpeakerSegment[]): string {
    return segments
        .map((segment) => {
            if (!segment.speaker) return segment.text;
            return `${segment.speaker}: ${segment.text}`;
        })
        .join("\n");
}
