"use client";

import { RuntimeModel } from "@/src/api/sonaApi";
import Modal from "@/src/components/ui/Modal";
import {
    BYOKConnection,
    byokConnectionLabel,
    providerLabel,
} from "@/src/hooks/useBYOKSettings";
import { ConfirmableModelAction } from "@/src/components/byok/types";
import { modelTypeLabel } from "@/src/components/byok/byokHelpers";

export function ConfirmConnectionDeleteModal({
    connection,
    linkedPresetCount,
    onCancel,
    onConfirm,
}: {
    connection: BYOKConnection;
    linkedPresetCount: number;
    onCancel: () => void;
    onConfirm: () => void;
}) {
    return (
        <Modal zClassName="z-[60]">
            <div className="w-full max-w-lg rounded-lg bg-white shadow-xl">
                <div className="border-b border-zinc-200 px-5 py-4">
                    <h3 className="text-base font-semibold text-zinc-950">
                        Delete connection
                    </h3>
                    <p className="mt-1 text-sm text-zinc-500">
                        This removes the saved connection from this browser.
                    </p>
                </div>

                <div className="flex flex-col gap-3 px-5 py-4">
                    <div className="rounded-md border border-zinc-200 bg-zinc-50 px-3 py-3">
                        <p className="text-sm font-medium text-zinc-950">
                            {byokConnectionLabel(connection)}
                        </p>
                        <p className="mt-1 text-xs text-zinc-500">
                            {providerLabel(connection.provider)}
                        </p>
                    </div>

                    <div className="rounded-md border border-amber-200 bg-amber-50 px-3 py-3 text-sm text-amber-900">
                        Deleting this connection will also delete {linkedPresetCount} linked model preset{linkedPresetCount === 1 ? "" : "s"}.
                    </div>
                </div>

                <div className="flex justify-end gap-3 border-t border-zinc-200 px-5 py-4">
                    <button
                        type="button"
                        onClick={onCancel}
                        className="min-h-10 rounded-md border border-zinc-300 px-4 text-sm font-medium hover:cursor-pointer text-zinc-700 hover:border-zinc-400 hover:text-zinc-950"
                    >
                        Cancel
                    </button>
                    <button
                        type="button"
                        onClick={onConfirm}
                        className="min-h-10 rounded-md bg-red-700 px-4 text-sm font-medium hover:cursor-pointer text-white hover:bg-red-800"
                    >
                        Delete
                    </button>
                </div>
            </div>
        </Modal>
    );
}

export function ConfirmModelPresetDeleteModal({
    label,
    onCancel,
    onConfirm,
}: {
    label: string;
    onCancel: () => void;
    onConfirm: () => void;
}) {
    return (
        <Modal zClassName="z-[60]">
            <div className="w-full max-w-lg rounded-lg bg-white shadow-xl">
                <div className="border-b border-zinc-200 px-5 py-4">
                    <h3 className="text-base font-semibold text-zinc-950">
                        Delete model preset
                    </h3>
                    <p className="mt-1 text-sm text-zinc-500">
                        This removes the saved model preset from this browser.
                    </p>
                </div>

                <div className="px-5 py-4">
                    <div className="rounded-md border border-zinc-200 bg-zinc-50 px-3 py-3">
                        <p className="text-sm font-medium text-zinc-950">
                            {label}
                        </p>
                    </div>
                </div>

                <div className="flex justify-end gap-3 border-t border-zinc-200 px-5 py-4">
                    <button
                        type="button"
                        onClick={onCancel}
                        className="min-h-10 rounded-md border border-zinc-300 px-4 text-sm hover:cursor-pointer font-medium text-zinc-700 hover:border-zinc-400 hover:text-zinc-950"
                    >
                        Cancel
                    </button>
                    <button
                        type="button"
                        onClick={onConfirm}
                        className="min-h-10 rounded-md bg-red-700 px-4 text-sm hover:cursor-pointer font-medium text-white hover:bg-red-800"
                    >
                        Delete
                    </button>
                </div>
            </div>
        </Modal>
    );
}

export function ConfirmModelActionModal({
    model,
    action,
    onCancel,
    onConfirm,
}: {
    model: RuntimeModel;
    action: ConfirmableModelAction;
    onCancel: () => void;
    onConfirm: () => void;
}) {
    const isRedownload = action === "redownload";

    return (
        <Modal zClassName="z-[60]">
            <div className="w-full max-w-lg rounded-lg bg-white shadow-xl">
                <div className="border-b border-zinc-200 px-5 py-4">
                    <h3 className="text-base font-semibold text-zinc-950">
                        {isRedownload ? "Re-download model" : "Uninstall model"}
                    </h3>
                    <p className="mt-1 text-sm text-zinc-500">
                        {isRedownload
                            ? "This will remove the current cached files before downloading the model again."
                            : "This will remove the cached model files from local storage."}
                    </p>
                </div>

                <div className="flex flex-col gap-3 px-5 py-4">
                    <div className="rounded-md border border-zinc-200 bg-zinc-50 px-3 py-3">
                        <p className="text-sm font-medium text-zinc-950">
                            {model.label}
                        </p>
                        <p className="mt-1 text-xs text-zinc-500">
                            {modelTypeLabel(model.type)} model
                        </p>
                        <p className="mt-2 break-all text-xs text-zinc-500">
                            Cache: {model.cache_path}
                        </p>
                    </div>

                    <div className="rounded-md border border-amber-200 bg-amber-50 px-3 py-3 text-sm text-amber-900">
                        {isRedownload
                            ? "The model cache will be deleted first, then Sona will fetch a fresh copy."
                            : "The model cache will be deleted. The next use will require downloading the model again."}
                    </div>
                </div>

                <div className="flex justify-end gap-3 border-t border-zinc-200 px-5 py-4">
                    <button
                        type="button"
                        onClick={onCancel}
                        className="min-h-10 rounded-md border border-zinc-300 px-4 text-sm font-medium text-zinc-700 hover:cursor-pointer hover:border-zinc-400 hover:text-zinc-950"
                    >
                        Cancel
                    </button>
                    <button
                        type="button"
                        onClick={onConfirm}
                        className={`min-h-10 rounded-md px-4 text-sm font-medium text-white hover:cursor-pointer ${isRedownload
                            ? "bg-zinc-950 hover:bg-zinc-400"
                            : "bg-red-700 hover:bg-red-800"
                            }`}
                    >
                        {isRedownload ? "Re-download" : "Uninstall"}
                    </button>
                </div>
            </div>
        </Modal>
    );
}
