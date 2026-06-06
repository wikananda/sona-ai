"use client";

import { useEffect, useMemo, useState } from "react";
import { BYOKProvider, BYOKSummarySettings } from "@/src/api/sonaApi";
import { BYOK_PROVIDERS } from "@/src/utils/constants";

const STORAGE_KEY_V1 = "sona-ai.byok-settings.v1";
const STORAGE_KEY_V2 = "sona-ai.byok-settings.v2";

export interface BYOKProviderSettings {
    apiKey: string;
    model: string;
    baseUrl: string;
}

export interface BYOKSettingsState {
    rememberKeys: boolean;
    selectedProvider: BYOKProvider;
    providers: Record<BYOKProvider, BYOKProviderSettings>;
}

const PROVIDER_VALUES = BYOK_PROVIDERS.map((provider) => provider.value);

export function useBYOKSettings() {
    const [settings, setSettings] = useState<BYOKSettingsState>(() =>
        typeof window === "undefined" ? defaultBYOKSettings() : readStoredSettings(),
    );

    useEffect(() => persistSettings(settings), [settings]);

    const selectedSettings = settings.providers[settings.selectedProvider];
    const isSelectedProviderConfigured = isProviderConfigured(
        settings.selectedProvider,
        selectedSettings,
    );
    const selectedBYOKSettings = useMemo<BYOKSummarySettings | undefined>(() => {
        if (!isSelectedProviderConfigured) return undefined;

        return {
            provider: settings.selectedProvider,
            apiKey: selectedSettings.apiKey.trim(),
            model: selectedSettings.model.trim(),
            baseUrl: selectedSettings.baseUrl.trim() || undefined,
        };
    }, [isSelectedProviderConfigured, selectedSettings, settings.selectedProvider]);

    return {
        settings,
        setSettings,
        clearSavedKeys: () => setSettings((current) => clearAllKeys(current)),
        selectedSettings,
        selectedBYOKSettings,
        isSelectedProviderConfigured,
    };
}

export function defaultBYOKSettings(): BYOKSettingsState {
    return {
        rememberKeys: false,
        selectedProvider: "openai",
        providers: {
            openai: providerDefaults("openai"),
            groq: providerDefaults("groq"),
            openrouter: providerDefaults("openrouter"),
            custom: providerDefaults("custom"),
        },
    };
}

export function clearAllKeys(settings: BYOKSettingsState): BYOKSettingsState {
    return {
        ...settings,
        rememberKeys: false,
        providers: {
            openai: { ...settings.providers.openai, apiKey: "" },
            groq: { ...settings.providers.groq, apiKey: "" },
            openrouter: { ...settings.providers.openrouter, apiKey: "" },
            custom: { ...settings.providers.custom, apiKey: "" },
        },
    };
}

export function isProviderConfigured(
    provider: BYOKProvider,
    settings: BYOKProviderSettings,
): boolean {
    return Boolean(settings.apiKey.trim()) &&
        Boolean(settings.model.trim()) &&
        (provider !== "custom" || Boolean(settings.baseUrl.trim()));
}

function providerDefaults(provider: BYOKProvider): BYOKProviderSettings {
    return {
        apiKey: "",
        model: BYOK_PROVIDERS.find((item) => item.value === provider)?.defaultModel ?? "",
        baseUrl: "",
    };
}

function readStoredSettings(): BYOKSettingsState {
    try {
        const v2 = window.localStorage.getItem(STORAGE_KEY_V2);
        if (v2) return normalizeSettings(JSON.parse(v2), true);

        const v1 = window.localStorage.getItem(STORAGE_KEY_V1);
        if (v1) {
            window.localStorage.removeItem(STORAGE_KEY_V1);
            return normalizeSettings(JSON.parse(v1), false);
        }

        return defaultBYOKSettings();
    } catch {
        return defaultBYOKSettings();
    }
}

function persistSettings(settings: BYOKSettingsState) {
    if (typeof window === "undefined") return;

    if (!settings.rememberKeys) {
        window.localStorage.removeItem(STORAGE_KEY_V1);
        window.localStorage.removeItem(STORAGE_KEY_V2);
        return;
    }

    window.localStorage.removeItem(STORAGE_KEY_V1);
    window.localStorage.setItem(STORAGE_KEY_V2, JSON.stringify(settings));
}

function normalizeSettings(value: unknown, allowRememberKeys: boolean): BYOKSettingsState {
    const defaults = defaultBYOKSettings();
    if (!value || typeof value !== "object") return defaults;

    const candidate = value as Partial<BYOKSettingsState>;
    const rememberKeys = allowRememberKeys && candidate.rememberKeys === true;
    const selectedProvider = isBYOKProvider(candidate.selectedProvider)
        ? candidate.selectedProvider
        : defaults.selectedProvider;

    const providers = { ...defaults.providers };
    if (candidate.providers && typeof candidate.providers === "object") {
        for (const provider of PROVIDER_VALUES) {
            const stored = candidate.providers[provider];
            if (!stored || typeof stored !== "object") continue;

            providers[provider] = {
                apiKey: stringValue(stored.apiKey),
                model: stringValue(stored.model) || providers[provider].model,
                baseUrl: stringValue(stored.baseUrl),
            };
        }
    }

    return { rememberKeys, selectedProvider, providers };
}

function isBYOKProvider(value: unknown): value is BYOKProvider {
    return typeof value === "string" && PROVIDER_VALUES.includes(value as BYOKProvider);
}

function stringValue(value: unknown): string {
    return typeof value === "string" ? value : "";
}
