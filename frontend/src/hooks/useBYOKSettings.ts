"use client";

import { useEffect, useMemo, useState } from "react";
import { BYOKProvider, BYOKSummarySettings } from "@/src/api/sonaApi";
import { BYOK_PROVIDERS } from "@/src/utils/constants";

const STORAGE_KEY = "sona-ai.byok-settings.v1";

export interface BYOKProviderSettings {
    apiKey: string;
    model: string;
    baseUrl: string;
}

export interface BYOKSettingsState {
    selectedProvider: BYOKProvider;
    providers: Record<BYOKProvider, BYOKProviderSettings>;
}

const PROVIDER_VALUES = BYOK_PROVIDERS.map((provider) => provider.value);

export function useBYOKSettings() {
    const [settings, setSettings] = useState<BYOKSettingsState>(() =>
        typeof window === "undefined" ? defaultBYOKSettings() : readStoredSettings(),
    );

    useEffect(() => {
        window.localStorage.setItem(STORAGE_KEY, JSON.stringify(settings));
    }, [settings]);

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
        selectedSettings,
        selectedBYOKSettings,
        isSelectedProviderConfigured,
    };
}

export function defaultBYOKSettings(): BYOKSettingsState {
    return {
        selectedProvider: "openai",
        providers: {
            openai: providerDefaults("openai"),
            groq: providerDefaults("groq"),
            openrouter: providerDefaults("openrouter"),
            custom: providerDefaults("custom"),
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
        const raw = window.localStorage.getItem(STORAGE_KEY);
        if (!raw) return defaultBYOKSettings();
        return normalizeSettings(JSON.parse(raw));
    } catch {
        return defaultBYOKSettings();
    }
}

function normalizeSettings(value: unknown): BYOKSettingsState {
    const defaults = defaultBYOKSettings();
    if (!value || typeof value !== "object") return defaults;

    const candidate = value as Partial<BYOKSettingsState>;
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

    return { selectedProvider, providers };
}

function isBYOKProvider(value: unknown): value is BYOKProvider {
    return typeof value === "string" && PROVIDER_VALUES.includes(value as BYOKProvider);
}

function stringValue(value: unknown): string {
    return typeof value === "string" ? value : "";
}
