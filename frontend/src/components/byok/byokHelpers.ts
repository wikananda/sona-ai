import { ModelJob, RuntimeModel } from "@/src/api/sonaApi";
import { BYOKConnection, BYOKModelPreset } from "@/src/hooks/useBYOKSettings";

export function trimConnection(connection: BYOKConnection): BYOKConnection {
    return {
        ...connection,
        name: connection.name.trim(),
        apiKey: connection.apiKey.trim(),
        baseUrl: connection.provider === "custom" ? connection.baseUrl.trim() : "",
    };
}

export function trimModelPreset(preset: BYOKModelPreset): BYOKModelPreset {
    return {
        ...preset,
        connectionId: preset.connectionId.trim(),
        model: preset.model.trim(),
        name: preset.name.trim(),
    };
}

export function isActiveJob(job: ModelJob): boolean {
    return job.status === "queued" || job.status === "running";
}

export function modelStatusLabel(status: RuntimeModel["status"]): string {
    if (status === "installed") return "Installed";
    if (status === "running") return "Working";
    if (status === "failed") return "Failed";
    return "Missing";
}

export function statusClassName(status: RuntimeModel["status"]): string {
    if (status === "installed") return "bg-emerald-100 text-emerald-800";
    if (status === "running") return "bg-blue-100 text-blue-800";
    if (status === "failed") return "bg-red-100 text-red-800";
    return "bg-zinc-100 text-zinc-600";
}

export function modelJobLabel(job?: ModelJob): string {
    if (!job) return "Working";
    if (job.action === "uninstall") return "Uninstalling";
    if (job.action === "redownload") return "Re-downloading";
    return "Downloading";
}

export function modelJobStageLabel(stage: ModelJob["stage"]): string {
    if (stage === "preparing") return "Preparing";
    if (stage === "downloading") return "Downloading";
    if (stage === "removing") return "Removing";
    if (stage === "verifying") return "Verifying";
    if (stage === "done") return "Done";
    if (stage === "failed") return "Failed";
    return "Queued";
}

export function modelTypeLabel(type: string): string {
    if (type === "transcription") return "Transcription";
    if (type === "alignment") return "Timestamp aligner";
    if (type === "diarization") return "Speaker extraction";
    return type.charAt(0).toUpperCase() + type.slice(1);
}
