import assert from "node:assert/strict";
import test from "node:test";

import {
    compatibleTranscriptionModel,
    modelSupportsLanguage,
    NEMOTRON_LANGUAGE_CODES,
    PARAKEET_LANGUAGE_CODES,
    transcriptionLanguageName,
    TRANSCRIPTION_LANGUAGE_DATA,
    unsupportedLanguageReason,
    WHISPER_LANGUAGE_CODES,
} from "../src/utils/transcriptionCapabilities.mjs";

const WHISPER_MODELS = ["faster-whisper-large-v3", "faster-whisper-turbo"];

test("catalog contains every unique model language plus auto detect", () => {
    const values = TRANSCRIPTION_LANGUAGE_DATA.map((language) => language.value);
    const expected = new Set([
        "auto",
        ...WHISPER_LANGUAGE_CODES,
        ...PARAKEET_LANGUAGE_CODES,
        ...NEMOTRON_LANGUAGE_CODES,
    ]);

    assert.equal(WHISPER_LANGUAGE_CODES.length, 100);
    assert.equal(PARAKEET_LANGUAGE_CODES.length, 25);
    assert.equal(NEMOTRON_LANGUAGE_CODES.length, 28);
    assert.equal(TRANSCRIPTION_LANGUAGE_DATA.length, 102);
    assert.equal(new Set(values).size, values.length);
    assert.deepEqual(new Set(values), expected);
});

test("every declared language is accepted by exactly its declared models", () => {
    for (const language of TRANSCRIPTION_LANGUAGE_DATA) {
        for (const model of WHISPER_MODELS) {
            assert.equal(
                modelSupportsLanguage(model, language.value),
                language.value === "auto" || WHISPER_LANGUAGE_CODES.includes(language.value),
                `${model}/${language.value}`,
            );
        }
        assert.equal(
            modelSupportsLanguage("parakeet", language.value),
            language.value === "auto" || PARAKEET_LANGUAGE_CODES.includes(language.value),
            `parakeet/${language.value}`,
        );
        assert.equal(
            modelSupportsLanguage("nemotron-3.5", language.value),
            language.value === "auto" || NEMOTRON_LANGUAGE_CODES.includes(language.value),
            `nemotron/${language.value}`,
        );
    }
});

test("Bahasa Indonesia disables NVIDIA models and falls back to Whisper", () => {
    assert.equal(transcriptionLanguageName("id"), "Bahasa Indonesia");
    assert.equal(modelSupportsLanguage("parakeet", "id"), false);
    assert.equal(modelSupportsLanguage("nemotron-3.5", "id"), false);
    assert.equal(modelSupportsLanguage("faster-whisper-large-v3", "id"), true);
    assert.equal(modelSupportsLanguage("faster-whisper-turbo", "id"), true);
    assert.equal(
        compatibleTranscriptionModel("parakeet", "id"),
        "faster-whisper-turbo",
    );
    assert.equal(
        unsupportedLanguageReason("nemotron-3.5", "id"),
        "Not supported for Bahasa Indonesia",
    );
});

test("normalizes regional locale values without conflating Norwegian variants", () => {
    assert.equal(modelSupportsLanguage("parakeet", "PT_br"), true);
    assert.equal(modelSupportsLanguage("nemotron-3.5", "en-US"), true);
    assert.equal(modelSupportsLanguage("nemotron-3.5", "nb-NO"), true);
    assert.equal(modelSupportsLanguage("faster-whisper-turbo", "nb-NO"), false);
    assert.equal(modelSupportsLanguage("faster-whisper-turbo", "no-NO"), true);
});

test("Nemotron adaptation-only languages stay disabled until fine-tuned", () => {
    for (const language of ["el", "he", "lt", "lv", "mt", "sl", "th", "nn"]) {
        assert.equal(modelSupportsLanguage("nemotron-3.5", language), false);
    }
    assert.equal(modelSupportsLanguage("parakeet", "el"), true);
});
