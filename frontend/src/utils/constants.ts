import { LocalLLMModel } from "@/src/api/sonaApi";

export const LOCAL_LLM_MODELS: { value: LocalLLMModel; label: string; description: string }[] = [
    {
        value: "qwen",
        label: "Qwen",
        description: "khazarai/Qwen3-4B-Qwen3.6-plus-Reasoning-Distilled-GGUF",
    },
    {
        value: "llama",
        label: "Llama",
        description: "meta-llama/Llama-3.2-3B-Instruct",
    },
    {
        value: "gemma",
        label: "Gemma",
        description: "google/gemma-4-E2B-it",
    },
];

export const BYOK_PROVIDERS = [
    { value: "openai", label: "OpenAI", defaultModel: "gpt-4o-mini" },
    { value: "groq", label: "Groq", defaultModel: "llama-3.1-8b-instant" },
    { value: "openrouter", label: "OpenRouter", defaultModel: "openai/gpt-4o-mini" },
    { value: "custom", label: "Custom", defaultModel: "" },
] as const;