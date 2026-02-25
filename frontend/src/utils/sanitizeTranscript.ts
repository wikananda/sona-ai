import { SpeakerSegment } from "@/src/api/sonaApi";

export default function sanitizeTranscript(segments: SpeakerSegment[]): string {
    return segments.map((segment) => `${segment.speaker}: ${segment.text}`).join("\n");
}