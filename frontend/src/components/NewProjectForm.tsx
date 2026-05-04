"use client";

import { FormEvent, useState } from "react";

interface Props {
    onCreate: (params: { name: string; description?: string }) => Promise<void>;
    onCancel: () => void;
    isCreating: boolean;
}

export default function NewProjectForm({ onCreate, onCancel, isCreating }: Props) {
    const [name, setName] = useState("");
    const [description, setDescription] = useState("");

    const handleSubmit = async (event: FormEvent) => {
        event.preventDefault();
        if (!name.trim()) return;

        await onCreate({
            name: name.trim(),
            description: description.trim() || undefined,
        });
        setName("");
        setDescription("");
    };

    return (
        <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            <div className="flex flex-col gap-3">
                <input
                    value={name}
                    onChange={(event) => setName(event.target.value)}
                    placeholder="Project name"
                    className="min-h-11 rounded-md border border-zinc-300 px-3 text-sm outline-none focus:border-zinc-900"
                />
                <textarea
                    value={description}
                    onChange={(event) => setDescription(event.target.value)}
                    placeholder="Description"
                    rows={4}
                    className="resize-none rounded-md border border-zinc-300 px-3 py-3 text-sm outline-none focus:border-zinc-900"
                />
            </div>

            <div className="flex justify-end gap-3">
                <button
                    type="button"
                    onClick={onCancel}
                    disabled={isCreating}
                    className="min-h-10 rounded-md border border-zinc-300 px-4 text-sm font-medium text-zinc-700 disabled:cursor-not-allowed disabled:opacity-40"
                >
                    Cancel
                </button>
                <button
                    type="submit"
                    disabled={isCreating || !name.trim()}
                    className="min-h-10 rounded-md bg-zinc-950 px-4 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-40"
                >
                    {isCreating ? "Creating" : "Create"}
                </button>
            </div>
        </form>
    );
}
