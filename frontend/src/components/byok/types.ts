import { ModelJobAction, RuntimeModel } from "@/src/api/sonaApi";
import { BYOKConnection, BYOKModelPreset } from "@/src/hooks/useBYOKSettings";

export type SettingsTab = "api" | "models";
export type ConfirmableModelAction = Extract<ModelJobAction, "uninstall" | "redownload">;

export interface PendingModelAction {
    model: RuntimeModel;
    action: ConfirmableModelAction;
}

export interface PendingConnectionDelete {
    connection: BYOKConnection;
    linkedPresetCount: number;
}

export interface PendingModelPresetDelete {
    preset: BYOKModelPreset;
    label: string;
}
