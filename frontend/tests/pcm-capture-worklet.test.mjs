import assert from "node:assert/strict";
import test from "node:test";

import { StreamingPcm16Resampler } from "../public/pcm-capture-worklet.mjs";

function collectFrames(resampler, channels, chunkSize) {
    const frames = [];
    for (let offset = 0; offset < channels[0].length; offset += chunkSize) {
        frames.push(...resampler.push(
            channels.map((channel) => channel.subarray(offset, offset + chunkSize)),
        ));
    }
    frames.push(...resampler.flush());
    const output = new Int16Array(frames.reduce((sum, frame) => sum + frame.length, 0));
    let offset = 0;
    for (const frame of frames) {
        output.set(frame, offset);
        offset += frame.length;
    }
    return output;
}

test("resamples ten seconds at 44.1 kHz without sample-count drift", () => {
    const input = new Float32Array(44100 * 10);
    for (let index = 0; index < input.length; index += 1) {
        input[index] = Math.sin(2 * Math.PI * 440 * index / 44100) * 0.5;
    }

    const output = collectFrames(new StreamingPcm16Resampler(44100), [input], 128);

    assert.equal(output.length, 16000 * 10);
    assert.ok(output.some((sample) => sample !== 0));
});

test("keeps identical phase across arbitrary 48 kHz input chunks", () => {
    const input = Float32Array.from(
        { length: 48000 + 37 },
        (_, index) => Math.sin(index / 17),
    );

    const whole = collectFrames(new StreamingPcm16Resampler(48000), [input], input.length);
    const chunked = collectFrames(new StreamingPcm16Resampler(48000), [input], 128);

    assert.deepEqual(chunked, whole);
});

test("downmixes stereo and clips to signed int16", () => {
    const left = Float32Array.from([2, 1, -2, -1]);
    const right = Float32Array.from([2, -1, -2, 1]);

    const output = collectFrames(
        new StreamingPcm16Resampler(16000, 16000, 32),
        [left, right],
        4,
    );

    assert.deepEqual(Array.from(output), [32767, 0, -32768, 0]);
});

test("flush emits a short residual frame", () => {
    const input = new Float32Array(4800).fill(0.25);
    const resampler = new StreamingPcm16Resampler(48000, 16000, 3200);

    const beforeFlush = resampler.push([input]);
    const afterFlush = resampler.flush();

    assert.equal(beforeFlush.length, 0);
    assert.equal(afterFlush.length, 1);
    assert.equal(afterFlush[0].length, 1600);
});
