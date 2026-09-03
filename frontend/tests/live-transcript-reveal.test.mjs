import assert from "node:assert/strict";
import test from "node:test";

import {
    commonTokenPrefix,
    nextVisibleTokenCount,
    tokenizeTranscript,
} from "../src/utils/liveTranscriptReveal.mjs";

test("tokenizes transcript words without losing spacing", () => {
    assert.deepEqual(
        tokenizeTranscript("Hello, world from Sona"),
        ["Hello, ", "world ", "from ", "Sona"],
    );
});

test("finds the stable prefix when a provisional revision changes", () => {
    const previous = tokenizeTranscript("we should ship on Thursday morning");
    const corrected = tokenizeTranscript("we should ship on Friday morning");

    assert.equal(commonTokenPrefix(previous, corrected), 4);
});

test("reveals normal updates one word at a time", () => {
    assert.equal(nextVisibleTokenCount(3, 7), 4);
    assert.equal(nextVisibleTokenCount(7, 7), 7);
});

test("catches up quickly when an engine emits a large batch", () => {
    assert.equal(nextVisibleTokenCount(0, 30), 4);
    assert.equal(nextVisibleTokenCount(12, 30), 14);
});
