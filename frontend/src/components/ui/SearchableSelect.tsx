"use client";

import {
    KeyboardEvent,
    useEffect,
    useId,
    useMemo,
    useRef,
    useState,
} from "react";

export interface SearchableSelectOption<Value extends string> {
    value: Value;
    label: string;
    description?: string;
    searchText?: string;
    disabled?: boolean;
    disabledReason?: string;
}

interface Props<Value extends string> {
    label: string;
    value: Value;
    options: readonly SearchableSelectOption<Value>[];
    onChange: (value: Value) => void;
    disabled?: boolean;
    searchable?: boolean;
    searchPlaceholder?: string;
    helpText?: string;
}

function normalizeSearchText(value: string): string {
    return value
        .normalize("NFD")
        .replace(/[\u0300-\u036f]/g, "")
        .toLocaleLowerCase()
        .trim();
}

export default function SearchableSelect<Value extends string>({
    label,
    value,
    options,
    onChange,
    disabled = false,
    searchable = true,
    searchPlaceholder = "Search",
    helpText,
}: Props<Value>) {
    const fieldId = useId();
    const rootRef = useRef<HTMLDivElement>(null);
    const triggerRef = useRef<HTMLButtonElement>(null);
    const [open, setOpen] = useState(false);
    const [query, setQuery] = useState("");
    const [activeIndex, setActiveIndex] = useState(0);
    const isOpen = open && !disabled;
    const selectedOption = options.find((option) => option.value === value);
    const filteredOptions = useMemo(() => {
        const normalizedQuery = normalizeSearchText(query);
        if (!normalizedQuery) return options;
        return options.filter((option) => normalizeSearchText([
            option.label,
            option.description,
            option.searchText,
            option.value,
        ].filter(Boolean).join(" ")).includes(normalizedQuery));
    }, [options, query]);
    const safeActiveIndex = filteredOptions.length === 0
        ? -1
        : Math.min(activeIndex, filteredOptions.length - 1);
    const labelId = `${fieldId}-label`;
    const helpId = `${fieldId}-help`;
    const listboxId = `${fieldId}-listbox`;
    const activeOptionId = safeActiveIndex >= 0
        ? `${fieldId}-option-${safeActiveIndex}`
        : undefined;

    useEffect(() => {
        if (!isOpen) return;
        const closeOnOutsidePointer = (event: PointerEvent) => {
            if (!rootRef.current?.contains(event.target as Node)) setOpen(false);
        };
        document.addEventListener("pointerdown", closeOnOutsidePointer);
        return () => document.removeEventListener("pointerdown", closeOnOutsidePointer);
    }, [isOpen]);

    const openMenu = (index = 0) => {
        if (disabled) return;
        setQuery("");
        setActiveIndex(Math.max(index, 0));
        setOpen(true);
    };

    const closeMenu = (restoreFocus = false) => {
        setOpen(false);
        setQuery("");
        if (restoreFocus) window.requestAnimationFrame(() => triggerRef.current?.focus());
    };

    const selectOption = (option: SearchableSelectOption<Value>) => {
        if (option.disabled) return;
        onChange(option.value);
        closeMenu(true);
    };

    const moveActive = (direction: 1 | -1) => {
        if (filteredOptions.length === 0) return;
        const nextIndex = safeActiveIndex < 0
            ? 0
            : (safeActiveIndex + direction + filteredOptions.length) % filteredOptions.length;
        setActiveIndex(nextIndex);
        window.requestAnimationFrame(() => {
            document.getElementById(`${fieldId}-option-${nextIndex}`)?.scrollIntoView({
                block: "nearest",
            });
        });
    };

    const handleListNavigation = (event: KeyboardEvent<HTMLElement>) => {
        if (event.key === "ArrowDown") {
            event.preventDefault();
            moveActive(1);
        } else if (event.key === "ArrowUp") {
            event.preventDefault();
            moveActive(-1);
        } else if (event.key === "Home") {
            event.preventDefault();
            setActiveIndex(0);
        } else if (event.key === "End") {
            event.preventDefault();
            setActiveIndex(Math.max(filteredOptions.length - 1, 0));
        } else if (event.key === "Enter" && safeActiveIndex >= 0) {
            event.preventDefault();
            selectOption(filteredOptions[safeActiveIndex]);
        } else if (event.key === "Escape") {
            event.preventDefault();
            closeMenu(true);
        }
    };

    return (
        <div
            ref={rootRef}
            className="relative flex flex-col gap-1"
            onBlur={(event) => {
                if (!event.currentTarget.contains(event.relatedTarget)) closeMenu();
            }}
        >
            <span id={labelId} className="text-xs font-medium text-zinc-500">
                {label}
            </span>
            <button
                ref={triggerRef}
                type="button"
                aria-labelledby={labelId}
                aria-describedby={helpText ? helpId : undefined}
                aria-haspopup="listbox"
                aria-expanded={isOpen}
                aria-controls={isOpen ? listboxId : undefined}
                aria-activedescendant={!searchable && isOpen ? activeOptionId : undefined}
                disabled={disabled}
                onClick={() => (isOpen ? closeMenu() : openMenu(
                    Math.max(options.findIndex((option) => option.value === value), 0),
                ))}
                onKeyDown={(event) => {
                    if (!isOpen && ["ArrowDown", "ArrowUp"].includes(event.key)) {
                        event.preventDefault();
                        openMenu(Math.max(options.findIndex((option) => option.value === value), 0));
                        return;
                    }
                    if (isOpen && !searchable) handleListNavigation(event);
                }}
                className="flex min-h-11 w-full items-center justify-between gap-3 rounded-md border border-zinc-300 bg-white px-3 py-2 text-left text-sm outline-none transition-colors hover:cursor-pointer hover:border-zinc-400 focus:border-zinc-900 focus:ring-2 focus:ring-zinc-900/10 disabled:cursor-not-allowed disabled:opacity-50"
            >
                <span className="min-w-0">
                    <span className="block truncate font-medium text-zinc-900">
                        {selectedOption?.label ?? value}
                    </span>
                    {selectedOption?.description && (
                        <span className="block truncate text-xs text-zinc-500">
                            {selectedOption.description}
                        </span>
                    )}
                </span>
                <svg
                    aria-hidden="true"
                    viewBox="0 0 20 20"
                    fill="none"
                    className={`h-4 w-4 shrink-0 text-zinc-500 transition-transform ${isOpen ? "rotate-180" : ""}`}
                >
                    <path d="m5 7.5 5 5 5-5" stroke="currentColor" strokeWidth="1.7" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
            </button>

            {helpText && (
                <span id={helpId} className="text-xs text-zinc-500">
                    {helpText}
                </span>
            )}

            {isOpen && (
                <div className="absolute left-0 right-0 top-full z-[70] mt-1 overflow-hidden rounded-lg border border-zinc-200 bg-white shadow-xl">
                    {searchable && (
                        <div className="border-b border-zinc-200 p-2">
                            <div className="flex items-center gap-2 rounded-md bg-zinc-100 px-3 focus-within:ring-2 focus-within:ring-zinc-900/15">
                                <svg aria-hidden="true" viewBox="0 0 20 20" fill="none" className="h-4 w-4 text-zinc-500">
                                    <circle cx="8.5" cy="8.5" r="5" stroke="currentColor" strokeWidth="1.6" />
                                    <path d="m12.3 12.3 3.5 3.5" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" />
                                </svg>
                                <input
                                    autoFocus
                                    type="search"
                                    value={query}
                                    onChange={(event) => {
                                        setQuery(event.target.value);
                                        setActiveIndex(0);
                                    }}
                                    onKeyDown={handleListNavigation}
                                    role="combobox"
                                    aria-label={searchPlaceholder}
                                    aria-autocomplete="list"
                                    aria-expanded="true"
                                    aria-controls={listboxId}
                                    aria-activedescendant={activeOptionId}
                                    placeholder={searchPlaceholder}
                                    className="min-h-10 w-full bg-transparent text-sm outline-none placeholder:text-zinc-500"
                                />
                            </div>
                        </div>
                    )}

                    <div
                        id={listboxId}
                        role="listbox"
                        aria-labelledby={labelId}
                        className="max-h-80 overflow-y-auto p-1.5"
                    >
                        {filteredOptions.length === 0 ? (
                            <p role="status" className="px-3 py-5 text-center text-sm text-zinc-500">
                                No matching options
                            </p>
                        ) : filteredOptions.map((option, index) => {
                            const selected = option.value === value;
                            const active = index === safeActiveIndex;
                            return (
                                <button
                                    id={`${fieldId}-option-${index}`}
                                    key={option.value}
                                    type="button"
                                    role="option"
                                    aria-selected={selected}
                                    aria-disabled={option.disabled || undefined}
                                    disabled={option.disabled}
                                    tabIndex={-1}
                                    onMouseEnter={() => setActiveIndex(index)}
                                    onClick={() => selectOption(option)}
                                    className={`flex min-h-11 w-full items-center justify-between gap-3 rounded-md px-3 py-2 text-left outline-none ${
                                        option.disabled
                                            ? "cursor-not-allowed bg-zinc-50 text-zinc-400"
                                            : active
                                                ? "cursor-pointer bg-zinc-100 text-zinc-950"
                                                : "cursor-pointer text-zinc-800 hover:bg-zinc-50"
                                    }`}
                                >
                                    <span className="min-w-0">
                                        <span className="block text-sm font-medium">{option.label}</span>
                                        <span className={`block text-xs ${option.disabled ? "text-zinc-400" : "text-zinc-500"}`}>
                                            {option.disabledReason ?? option.description}
                                        </span>
                                    </span>
                                    {selected && (
                                        <svg aria-hidden="true" viewBox="0 0 20 20" fill="none" className="h-4 w-4 shrink-0">
                                            <path d="m4.5 10.5 3.2 3.2 7.8-8" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" />
                                        </svg>
                                    )}
                                </button>
                            );
                        })}
                    </div>
                </div>
            )}
        </div>
    );
}
