"use client";

import { useEffect, useMemo, useState } from "react";
import { BYOKProvider, BYOKSummarySettings } from "@/src/api/sonaApi";
import { BYOK_PROVIDERS } from "@/src/utils/constants";

const STORAGE_KEY_V1 = "sona-ai.byok-settings.v1";
const STORAGE_KEY_V2 = "sona-ai.byok-settings.v2";
const STORAGE_KEY_V3 = "sona-ai.byok-settings.v3";

export interface BYOKEntry {
    id: string;
    provider: BYOKProvider;
    apiKey: string;
    model: string;
    baseUrl: string;
}

export interface BYOKEntryDraft {
    provider: BYOKProvider;
    apiKey: string;
    model: string;
    baseUrl: string;
}

export interface BYOKSettingsState {
    rememberKeys: boolean;
    entries: BYOKEntry[];
}

const PROVIDER_VALUES = BYOK_PROVIDERS.map((provider) => provider.value);

export function useBYOKSettings() {
    const [settings, setSettings] = useState<BYOKSettingsState>(() =>
        typeof window === "undefined" ? defaultBYOKSettings() : readStoredSettings(),
    );

    useEffect(() => persistSettings(settings), [settings]);

    const validEntries = useMemo(
        () => settings.entries.filter(isBYOKEntryConfigured),
        [settings.entries],
    );

    return {
        settings,
        setSettings,
        clearSavedKeys: () => setSettings((current) => clearAllKeys(current)),
        validEntries,
    };
}

export function defaultBYOKSettings(): BYOKSettingsState {
    return {
        rememberKeys: false,
        entries: [],
    };
}

export function createBYOKEntry(draft?: Partial<BYOKEntryDraft>): BYOKEntry {
    return {
        id: createEntryId(),
        provider: draft?.provider ?? "openai",
        apiKey: draft?.apiKey ?? "",
        model: draft?.model ?? providerDefaultModel(draft?.provider ?? "openai"),
        baseUrl: draft?.baseUrl ?? "",
    };
}

export function clearAllKeys(settings: BYOKSettingsState): BYOKSettingsState {
    return {
        ...settings,
        rememberKeys: false,
        entries: settings.entries.map((entry) => ({
            ...entry,
            apiKey: "",
        })),
    };
}

export function isBYOKEntryConfigured(entry: BYOKEntry): boolean {
    return Boolean(entry.apiKey.trim()) &&
        Boolean(entry.model.trim()) &&
        (entry.provider !== "custom" || Boolean(entry.baseUrl.trim()));
}

export function byokEntryToSettings(entry: BYOKEntry): BYOKSummarySettings {
    return {
        provider: entry.provider,
        apiKey: entry.apiKey.trim(),
        model: entry.model.trim(),
        baseUrl: entry.baseUrl.trim() || undefined,
    };
}

export function byokEntryLabel(entry: Pick<BYOKEntry, "provider" | "model">): string {
    return `${providerLabel(entry.provider)} / ${entry.model || "No model"}`;
}

export function providerLabel(provider: BYOKProvider): string {
    return BYOK_PROVIDERS.find((item) => item.value === provider)?.label ?? provider;
}

export function providerDefaultModel(provider: BYOKProvider): string {
    return BYOK_PROVIDERS.find((item) => item.value === provider)?.defaultModel ?? "";
}

function readStoredSettings(): BYOKSettingsState {
    try {
        const v3 = window.localStorage.getItem(STORAGE_KEY_V3);
        if (v3) return normalizeEntrySettings(JSON.parse(v3), true);

        const v2 = window.localStorage.getItem(STORAGE_KEY_V2);
        if (v2) return migrateLegacySettings(JSON.parse(v2), true);

        const v1 = window.localStorage.getItem(STORAGE_KEY_V1);
        if (v1) {
            window.localStorage.removeItem(STORAGE_KEY_V1);
            return migrateLegacySettings(JSON.parse(v1), false);
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
        window.localStorage.removeItem(STORAGE_KEY_V3);
        return;
    }

    window.localStorage.removeItem(STORAGE_KEY_V1);
    window.localStorage.removeItem(STORAGE_KEY_V2);
    window.localStorage.setItem(STORAGE_KEY_V3, JSON.stringify(settings));
}

function normalizeEntrySettings(value: unknown, allowRememberKeys: boolean): BYOKSettingsState {
    if (!value || typeof value !== "object") return defaultBYOKSettings();

    const candidate = value as Partial<BYOKSettingsState>;
    const rememberKeys = allowRememberKeys && candidate.rememberKeys === true;
    const entries = Array.isArray(candidate.entries)
        ? candidate.entries.flatMap(normalizeEntry)
        : [];

    return { rememberKeys, entries };
}

function migrateLegacySettings(value: unknown, allowRememberKeys: boolean): BYOKSettingsState {
    if (!value || typeof value !== "object") return defaultBYOKSettings();

    const candidate = value as {
        rememberKeys?: unknown;
        providers?: Record<string, unknown>;
    };
    const rememberKeys = allowRememberKeys && candidate.rememberKeys === true;
    const entries: BYOKEntry[] = [];

    if (candidate.providers && typeof candidate.providers === "object") {
        for (const provider of PROVIDER_VALUES) {
            const stored = candidate.providers[provider];
            if (!stored || typeof stored !== "object") continue;

            const legacy = stored as Record<string, unknown>;
            const entry = createBYOKEntry({
                provider,
                apiKey: stringValue(legacy.apiKey),
                model: stringValue(legacy.model) || providerDefaultModel(provider),
                baseUrl: stringValue(legacy.baseUrl),
            });
            if (isBYOKEntryConfigured(entry)) {
                entries.push(entry);
            }
        }
    }

    return { rememberKeys, entries };
}

function normalizeEntry(value: unknown): BYOKEntry[] {
    if (!value || typeof value !== "object") return [];

    const candidate = value as Partial<BYOKEntry>;
    if (!isBYOKProvider(candidate.provider)) return [];

    return [
        {
            id: stringValue(candidate.id) || createEntryId(),
            provider: candidate.provider,
            apiKey: stringValue(candidate.apiKey),
            model: stringValue(candidate.model) || providerDefaultModel(candidate.provider),
            baseUrl: stringValue(candidate.baseUrl),
        },
    ];
}

function isBYOKProvider(value: unknown): value is BYOKProvider {
    return typeof value === "string" && PROVIDER_VALUES.includes(value as BYOKProvider);
}

function stringValue(value: unknown): string {
    return typeof value === "string" ? value : "";
}

function createEntryId(): string {
    if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
        return crypto.randomUUID();
    }
    return `byok-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}
