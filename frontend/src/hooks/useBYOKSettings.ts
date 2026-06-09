"use client";

import { useEffect, useMemo, useState } from "react";
import { BYOKProvider, BYOKSummarySettings } from "@/src/api/sonaApi";
import { BYOK_PROVIDERS } from "@/src/utils/constants";

const STORAGE_KEY_V1 = "sona-ai.byok-settings.v1";
const STORAGE_KEY_V2 = "sona-ai.byok-settings.v2";
const STORAGE_KEY_V3 = "sona-ai.byok-settings.v3";
const STORAGE_KEY_V4 = "sona-ai.byok-settings.v4";

export interface BYOKConnection {
    id: string;
    name: string;
    provider: BYOKProvider;
    apiKey: string;
    baseUrl: string;
}

export interface BYOKConnectionDraft {
    name: string;
    provider: BYOKProvider;
    apiKey: string;
    baseUrl: string;
}

export interface BYOKModelPreset {
    id: string;
    connectionId: string;
    model: string;
    name: string;
}

export interface BYOKModelPresetDraft {
    connectionId: string;
    model: string;
    name: string;
}

export interface BYOKResolvedModelPreset extends BYOKModelPreset {
    connection: BYOKConnection;
}

export interface BYOKSettingsState {
    rememberKeys: boolean;
    connections: BYOKConnection[];
    modelPresets: BYOKModelPreset[];
}

const PROVIDER_VALUES = BYOK_PROVIDERS.map((provider) => provider.value);

export function useBYOKSettings() {
    const [settings, setSettings] = useState<BYOKSettingsState>(() =>
        typeof window === "undefined" ? defaultBYOKSettings() : readStoredSettings(),
    );

    useEffect(() => persistSettings(settings), [settings]);

    const validConnections = useMemo(
        () => settings.connections.filter(isBYOKConnectionConfigured),
        [settings.connections],
    );

    const validModelPresets = useMemo(() => {
        const connectionMap = new Map(validConnections.map((connection) => [connection.id, connection]));

        return settings.modelPresets.flatMap((preset) => {
            if (!isBYOKModelPresetConfigured(preset)) {
                return [];
            }

            const connection = connectionMap.get(preset.connectionId);
            if (!connection) {
                return [];
            }

            return [{ ...preset, connection }];
        });
    }, [settings.modelPresets, validConnections]);

    return {
        settings,
        setSettings,
        clearSavedKeys: () => setSettings((current) => clearAllKeys(current)),
        validConnections,
        validModelPresets,
    };
}

export function defaultBYOKSettings(): BYOKSettingsState {
    return {
        rememberKeys: false,
        connections: [],
        modelPresets: [],
    };
}

export function createBYOKConnection(
    draft?: Partial<BYOKConnectionDraft>,
): BYOKConnection {
    const provider = draft?.provider ?? "openai";
    return {
        id: createId("byok-connection"),
        name: draft?.name ?? providerLabel(provider),
        provider,
        apiKey: draft?.apiKey ?? "",
        baseUrl: draft?.baseUrl ?? "",
    };
}

export function createBYOKModelPreset(
    draft?: Partial<BYOKModelPresetDraft>,
): BYOKModelPreset {
    return {
        id: createId("byok-model"),
        connectionId: draft?.connectionId ?? "",
        model: draft?.model ?? "",
        name: draft?.name ?? "",
    };
}

export function clearAllKeys(settings: BYOKSettingsState): BYOKSettingsState {
    return {
        ...settings,
        rememberKeys: false,
        connections: settings.connections.map((connection) => ({
            ...connection,
            apiKey: "",
        })),
    };
}

export function isBYOKConnectionConfigured(connection: BYOKConnection): boolean {
    return Boolean(connection.name.trim()) &&
        Boolean(connection.apiKey.trim()) &&
        (connection.provider !== "custom" || Boolean(connection.baseUrl.trim()));
}

export function isBYOKModelPresetConfigured(preset: BYOKModelPreset): boolean {
    return Boolean(preset.connectionId.trim()) && Boolean(preset.model.trim());
}

export function byokResolvedModelPresetToSettings(
    preset: BYOKResolvedModelPreset,
): BYOKSummarySettings {
    return {
        provider: preset.connection.provider,
        apiKey: preset.connection.apiKey.trim(),
        model: preset.model.trim(),
        baseUrl: preset.connection.baseUrl.trim() || undefined,
    };
}

export function byokConnectionLabel(connection: Pick<BYOKConnection, "name" | "provider">): string {
    return connection.name.trim() || providerLabel(connection.provider);
}

export function byokResolvedModelPresetLabel(
    preset: Pick<BYOKModelPreset, "name" | "model"> & {
        connection: Pick<BYOKConnection, "name" | "provider">;
    },
): string {
    return preset.name.trim() || `${byokConnectionLabel(preset.connection)} / ${preset.model || "No model"}`;
}

export function providerLabel(provider: BYOKProvider): string {
    return BYOK_PROVIDERS.find((item) => item.value === provider)?.label ?? provider;
}

export function providerDefaultModel(provider: BYOKProvider): string {
    return BYOK_PROVIDERS.find((item) => item.value === provider)?.defaultModel ?? "";
}

function readStoredSettings(): BYOKSettingsState {
    try {
        const v4 = window.localStorage.getItem(STORAGE_KEY_V4);
        if (v4) return normalizeSettingsV4(JSON.parse(v4), true);

        const v3 = window.localStorage.getItem(STORAGE_KEY_V3);
        if (v3) return migrateV3Entries(JSON.parse(v3), true);

        const v2 = window.localStorage.getItem(STORAGE_KEY_V2);
        if (v2) return migrateLegacyProviderSettings(JSON.parse(v2), true);

        const v1 = window.localStorage.getItem(STORAGE_KEY_V1);
        if (v1) {
            window.localStorage.removeItem(STORAGE_KEY_V1);
            return migrateLegacyProviderSettings(JSON.parse(v1), false);
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
        window.localStorage.removeItem(STORAGE_KEY_V4);
        return;
    }

    window.localStorage.removeItem(STORAGE_KEY_V1);
    window.localStorage.removeItem(STORAGE_KEY_V2);
    window.localStorage.removeItem(STORAGE_KEY_V3);
    window.localStorage.setItem(STORAGE_KEY_V4, JSON.stringify(settings));
}

function normalizeSettingsV4(value: unknown, allowRememberKeys: boolean): BYOKSettingsState {
    if (!value || typeof value !== "object") return defaultBYOKSettings();

    const candidate = value as Partial<BYOKSettingsState>;
    const rememberKeys = allowRememberKeys && candidate.rememberKeys === true;
    const connections = Array.isArray(candidate.connections)
        ? candidate.connections.flatMap(normalizeConnection)
        : [];
    const modelPresets = Array.isArray(candidate.modelPresets)
        ? candidate.modelPresets.flatMap(normalizeModelPreset)
        : [];

    return {
        rememberKeys,
        connections,
        modelPresets,
    };
}

function migrateV3Entries(value: unknown, allowRememberKeys: boolean): BYOKSettingsState {
    if (!value || typeof value !== "object") return defaultBYOKSettings();

    const candidate = value as {
        rememberKeys?: unknown;
        entries?: unknown[];
    };
    const rememberKeys = allowRememberKeys && candidate.rememberKeys === true;
    const connections: BYOKConnection[] = [];
    const modelPresets: BYOKModelPreset[] = [];
    const providerCounts = new Map<BYOKProvider, number>();

    if (Array.isArray(candidate.entries)) {
        for (const rawEntry of candidate.entries) {
            const entry = normalizeLegacyEntry(rawEntry);
            if (!entry) continue;

            const providerIndex = (providerCounts.get(entry.provider) ?? 0) + 1;
            providerCounts.set(entry.provider, providerIndex);
            const connection = createBYOKConnection({
                name: legacyConnectionName(entry.provider, providerIndex, entry.baseUrl),
                provider: entry.provider,
                apiKey: entry.apiKey,
                baseUrl: entry.baseUrl,
            });
            const preset = createBYOKModelPreset({
                connectionId: connection.id,
                model: entry.model || providerDefaultModel(entry.provider),
                name: "",
            });

            connections.push(connection);
            modelPresets.push(preset);
        }
    }

    return { rememberKeys, connections, modelPresets };
}

function migrateLegacyProviderSettings(value: unknown, allowRememberKeys: boolean): BYOKSettingsState {
    if (!value || typeof value !== "object") return defaultBYOKSettings();

    const candidate = value as {
        rememberKeys?: unknown;
        providers?: Record<string, unknown>;
    };
    const rememberKeys = allowRememberKeys && candidate.rememberKeys === true;
    const connections: BYOKConnection[] = [];
    const modelPresets: BYOKModelPreset[] = [];
    let providerIndex = 0;

    if (candidate.providers && typeof candidate.providers === "object") {
        for (const provider of PROVIDER_VALUES) {
            const stored = candidate.providers[provider];
            if (!stored || typeof stored !== "object") continue;

            providerIndex += 1;
            const legacy = stored as Record<string, unknown>;
            const connection = createBYOKConnection({
                name: legacyConnectionName(provider, providerIndex, stringValue(legacy.baseUrl)),
                provider,
                apiKey: stringValue(legacy.apiKey),
                baseUrl: stringValue(legacy.baseUrl),
            });
            const preset = createBYOKModelPreset({
                connectionId: connection.id,
                model: stringValue(legacy.model) || providerDefaultModel(provider),
                name: "",
            });

            if (isBYOKConnectionConfigured(connection) && isBYOKModelPresetConfigured(preset)) {
                connections.push(connection);
                modelPresets.push(preset);
            }
        }
    }

    return { rememberKeys, connections, modelPresets };
}

function normalizeConnection(value: unknown): BYOKConnection[] {
    if (!value || typeof value !== "object") return [];

    const candidate = value as Partial<BYOKConnection>;
    if (!isBYOKProvider(candidate.provider)) return [];

    return [{
        id: stringValue(candidate.id) || createId("byok-connection"),
        name: stringValue(candidate.name) || providerLabel(candidate.provider),
        provider: candidate.provider,
        apiKey: stringValue(candidate.apiKey),
        baseUrl: stringValue(candidate.baseUrl),
    }];
}

function normalizeModelPreset(value: unknown): BYOKModelPreset[] {
    if (!value || typeof value !== "object") return [];

    const candidate = value as Partial<BYOKModelPreset>;
    return [{
        id: stringValue(candidate.id) || createId("byok-model"),
        connectionId: stringValue(candidate.connectionId),
        model: stringValue(candidate.model),
        name: stringValue(candidate.name),
    }];
}

function normalizeLegacyEntry(value: unknown): {
    provider: BYOKProvider;
    apiKey: string;
    model: string;
    baseUrl: string;
} | null {
    if (!value || typeof value !== "object") return null;

    const candidate = value as Record<string, unknown>;
    if (!isBYOKProvider(candidate.provider)) return null;

    return {
        provider: candidate.provider,
        apiKey: stringValue(candidate.apiKey),
        model: stringValue(candidate.model),
        baseUrl: stringValue(candidate.baseUrl),
    };
}

function legacyConnectionName(
    provider: BYOKProvider,
    index: number,
    baseUrl: string,
): string {
    if (provider === "custom" && baseUrl.trim()) {
        return `Custom ${index}`;
    }
    return `${providerLabel(provider)} ${index}`;
}

function isBYOKProvider(value: unknown): value is BYOKProvider {
    return typeof value === "string" && PROVIDER_VALUES.includes(value as BYOKProvider);
}

function stringValue(value: unknown): string {
    return typeof value === "string" ? value : "";
}

function createId(prefix: string): string {
    if (typeof crypto !== "undefined" && "randomUUID" in crypto) {
        return `${prefix}-${crypto.randomUUID()}`;
    }
    return `${prefix}-${Date.now()}-${Math.random().toString(36).slice(2)}`;
}
