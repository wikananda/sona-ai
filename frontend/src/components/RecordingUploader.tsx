"use client";

import { useState } from "react";
import { RuntimeDevice, RuntimeDevices, TranscriptionModel } from "@/src/api/sonaApi";
import AudioFileDropzone from "@/src/components/AudioFileDropzone";
import RecordingSettingsForm from "@/src/components/RecordingSettingsForm";

interface Props {
    onUpload: (params: {
        files: File[];
        language?: string;
        model: TranscriptionModel;
        device: RuntimeDevice;
        minSpeakers?: number | "";
        maxSpeakers?: number | "";
        extractSpeakers?: boolean;
    }) => Promise<boolean>;
    isUploading: boolean;
    runtimeDevices: RuntimeDevices;
}

export default function RecordingUploader({ onUpload, isUploading, runtimeDevices }: Props) {
    const [file, setFile] = useState<File | null>(null);

    return (
        <div className="flex flex-col gap-4">
            <AudioFileDropzone
                file={file}
                onFileChange={setFile}
                disabled={isUploading}
                title="Drop one audio file here"
                helperText="Choose from computer or drag audio into this area."
            />

            {file && (
                <RecordingSettingsForm
                    file={file}
                    onUpload={async (params) => {
                        const started = await onUpload(params);
                        if (started) {
                            setFile(null);
                        }
                        return started;
                    }}
                    isUploading={isUploading}
                    runtimeDevices={runtimeDevices}
                    onClear={() => setFile(null)}
                />
            )}
        </div>
    );
}
