"use client";

import { FormEvent, useState } from "react";

interface Props {
    onCreate: (params: { name: string; description?: string }) => Promise<void>;
    isCreating: boolean;
}

export default function NewProjectForm({ onCreate, isCreating }: Props) {
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
        <form onSubmit={handleSubmit} className="flex flex-col gap-3 rounded-lg border border-zinc-200 bg-white p-4">
            <div className="grid gap-3 md:grid-cols-[1fr_1.5fr_auto]">
                <input
                    value={name}
                    onChange={(event) => setName(event.target.value)}
                    placeholder="Project name"
                    className="min-h-11 rounded-md border border-zinc-300 px-3 text-sm outline-none focus:border-zinc-900"
                />
                <input
                    value={description}
                    onChange={(event) => setDescription(event.target.value)}
                    placeholder="Description"
                    className="min-h-11 rounded-md border border-zinc-300 px-3 text-sm outline-none focus:border-zinc-900"
                />
                <button
                    type="submit"
                    disabled={isCreating || !name.trim()}
                    className="min-h-11 rounded-md bg-zinc-950 px-4 text-sm font-medium text-white disabled:cursor-not-allowed disabled:opacity-40"
                >
                    {isCreating ? "Creating" : "Create"}
                </button>
            </div>
        </form>
    );
}
