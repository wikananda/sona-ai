import { BYOKSummarySettings } from "@/src/api/sonaApi";
import {
    BYOKResolvedModelPreset,
    byokResolvedModelPresetToSettings,
} from "@/src/hooks/useBYOKSettings";

export interface BYOKPresetSelection {
    effectivePresetId: string;
    selectedPreset?: BYOKResolvedModelPreset;
    selectedSettings?: BYOKSummarySettings;
}

export function selectBYOKPreset(
    presets: BYOKResolvedModelPreset[],
    presetId: string,
): BYOKPresetSelection {
    const effectivePresetId =
        presets.find((preset) => preset.id === presetId)?.id ??
        presets[0]?.id ??
        "";
    const selectedPreset = presets.find((preset) => preset.id === effectivePresetId);

    return {
        effectivePresetId,
        selectedPreset,
        selectedSettings: selectedPreset
            ? byokResolvedModelPresetToSettings(selectedPreset)
            : undefined,
    };
}
