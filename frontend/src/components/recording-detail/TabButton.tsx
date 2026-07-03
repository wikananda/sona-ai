"use client";

export default function TabButton({
    label,
    isActive,
    onClick,
}: {
    label: string;
    isActive: boolean;
    onClick: () => void;
}) {
    return (
        <button
            type="button"
            onClick={onClick}
            className={`min-h-11 border-b-2 px-4 text-sm font-medium transition-colors ${isActive
                ? "border-zinc-950 text-zinc-950"
                : "border-transparent text-zinc-500 hover:text-zinc-950 hover:cursor-pointer"
                }`}
        >
            {label}
        </button>
    );
}
