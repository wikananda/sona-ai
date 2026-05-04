import { RecordingStatus } from "@/src/api/sonaApi";

const LABELS: Record<RecordingStatus, string> = {
    pending: "Pending",
    processing: "Processing",
    done: "Done",
    failed: "Failed",
};

const STYLES: Record<RecordingStatus, string> = {
    pending: "bg-zinc-100 text-zinc-700",
    processing: "bg-amber-100 text-amber-800",
    done: "bg-emerald-100 text-emerald-800",
    failed: "bg-red-100 text-red-800",
};

interface Props {
    status: RecordingStatus;
}

export default function RecordingStatusBadge({ status }: Props) {
    return (
        <span className={`inline-flex rounded-full px-2 py-1 text-xs font-medium ${STYLES[status]}`}>
            {LABELS[status]}
        </span>
    );
}
