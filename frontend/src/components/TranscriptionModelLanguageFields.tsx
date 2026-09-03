"use client";

import { useMemo, useState } from "react";
import { TranscriptionModel } from "@/src/api/sonaApi";
import SearchableSelect, {
    SearchableSelectOption,
} from "@/src/components/ui/SearchableSelect";
import {
    compatibleModelForLanguage,
    isModelLanguageCompatible,
    transcriptionLanguageName,
    transcriptionModelLanguageNote,
    TRANSCRIPTION_LANGUAGES,
    TRANSCRIPTION_MODELS,
    unsupportedLanguageReason,
} from "@/src/utils/transcriptionSettings";

interface Props {
    language: string;
    model: TranscriptionModel;
    onLanguageChange: (language: string) => void;
    onModelChange: (model: TranscriptionModel) => void;
    disabled?: boolean;
    className?: string;
}

const LANGUAGE_OPTIONS: readonly SearchableSelectOption<string>[] = TRANSCRIPTION_LANGUAGES.map(
    (language) => {
        const compatibleModels = TRANSCRIPTION_MODELS.filter((model) => (
            isModelLanguageCompatible(model.value, language.value)
        )).length;
        const translatedName = language.englishName
            ? `${language.englishName} · `
            : "";
        return {
            value: language.value,
            label: language.label,
            description: language.value === "auto"
                ? "Let the selected model detect the language"
                : `${translatedName}${language.value.toUpperCase()} · ${compatibleModels} ${compatibleModels === 1 ? "model" : "models"}`,
            searchText: [
                language.englishName,
                ...(language.aliases ?? []),
                language.value,
            ].filter(Boolean).join(" "),
        };
    },
);

export default function TranscriptionModelLanguageFields({
    language,
    model,
    onLanguageChange,
    onModelChange,
    disabled = false,
    className = "flex flex-col gap-3",
}: Props) {
    const [announcement, setAnnouncement] = useState("");
    const selectedLanguage = transcriptionLanguageName(language);
    const languageOptions = useMemo(() => {
        if (LANGUAGE_OPTIONS.some((option) => option.value === language)) {
            return LANGUAGE_OPTIONS;
        }
        return [
            {
                value: language,
                label: selectedLanguage,
                description: `Saved language code · ${language}`,
                searchText: language,
            },
            ...LANGUAGE_OPTIONS,
        ];
    }, [language, selectedLanguage]);
    const modelOptions = useMemo<readonly SearchableSelectOption<TranscriptionModel>[]>(() => (
        TRANSCRIPTION_MODELS.map((item) => {
            const compatible = isModelLanguageCompatible(item.value, language);
            return {
                value: item.value,
                label: item.label,
                description: compatible
                    ? language === "auto"
                        ? `${item.description} · Auto-detect supported`
                        : `${item.description} · Supports ${selectedLanguage}`
                    : item.description,
                disabled: !compatible,
                disabledReason: compatible
                    ? undefined
                    : `${item.description} · ${unsupportedLanguageReason(item.value, language)}`,
                searchText: `${item.value} ${item.description}`,
            };
        })
    ), [language, selectedLanguage]);
    const unavailableModels = TRANSCRIPTION_MODELS.filter((item) => (
        !isModelLanguageCompatible(item.value, language)
    ));
    const modelNote = transcriptionModelLanguageNote(model, language);
    const compatibilitySummary = unavailableModels.length === 0
        ? "All transcription models support this language choice."
        : `${formatModelList(unavailableModels.map((item) => item.label))} ${
            unavailableModels.length === 1 ? "is" : "are"
        } unavailable for ${selectedLanguage}.`;

    const handleLanguageChange = (nextLanguage: string) => {
        const nextModel = compatibleModelForLanguage(model, nextLanguage);
        onLanguageChange(nextLanguage);
        if (nextModel !== model) {
            onModelChange(nextModel);
            const previousModel = TRANSCRIPTION_MODELS.find((item) => item.value === model)?.label
                ?? model;
            const fallbackModel = TRANSCRIPTION_MODELS.find((item) => item.value === nextModel)?.label
                ?? nextModel;
            setAnnouncement(
                `${fallbackModel} selected because ${previousModel} does not support ${transcriptionLanguageName(nextLanguage)}.`,
            );
        } else {
            setAnnouncement("");
        }
    };

    return (
        <div className={className}>
            <SearchableSelect
                label="Language"
                value={language}
                options={languageOptions}
                onChange={handleLanguageChange}
                disabled={disabled}
                searchPlaceholder="Search languages or codes"
                helpText={`${TRANSCRIPTION_LANGUAGES.length - 1} languages across the available models.`}
            />

            <SearchableSelect
                label="Model"
                value={model}
                options={modelOptions}
                onChange={onModelChange}
                disabled={disabled}
                searchable={false}
                helpText={modelNote}
            />

            <p className="text-xs text-zinc-500">{compatibilitySummary}</p>
            <p
                role="status"
                aria-live="polite"
                className={announcement
                    ? "rounded-md border border-amber-200 bg-amber-50 px-3 py-2 text-xs text-amber-900"
                    : "sr-only"}
            >
                {announcement}
            </p>
        </div>
    );
}

function formatModelList(models: string[]): string {
    if (models.length <= 1) return models[0] ?? "";
    if (models.length === 2) return `${models[0]} and ${models[1]}`;
    return `${models.slice(0, -1).join(", ")}, and ${models.at(-1)}`;
}
