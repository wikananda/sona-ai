export interface SpeakerSegment {
    speaker: string;
    text: string;
    start: number;
    end: number;
}

export interface TranscribeParams {
    file: File;
    language?: string;
    minSpeakers?: number | "";
    maxSpeakers?: number | "";
}

const BASE_URL = "http://localhost:8000";

export async function transcribeAudio(params: TranscribeParams): Promise<SpeakerSegment[]> {
    const formData = new FormData();
    formData.append("file", params.file);

    const url = new URL(`${BASE_URL}/transcribe`);
    if (params.language && params.language !== "None") {
        url.searchParams.append("language", params.language);
    }
    if (params.minSpeakers !== "") {
        url.searchParams.append("min_speakers", String(params.minSpeakers));
    }
    if (params.maxSpeakers !== "") {
        url.searchParams.append("max_speakers", String(params.maxSpeakers));
    }

    const response = await fetch(url.toString(), {
        method: "POST",
        body: formData,
    })

    if (!response.ok) {
        throw new Error(`API error! status: ${response.status}`);
    }

    const data = await response.json();
    return data.transcript;
}