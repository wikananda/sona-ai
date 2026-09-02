const DEFAULT_TARGET_RATE = 16000;
const DEFAULT_FRAME_SAMPLES = 3200;

export class StreamingPcm16Resampler {
    constructor(inputRate, targetRate = DEFAULT_TARGET_RATE, frameSamples = DEFAULT_FRAME_SAMPLES) {
        if (!Number.isFinite(inputRate) || inputRate <= 0) {
            throw new Error("inputRate must be positive");
        }
        if (!Number.isFinite(targetRate) || targetRate <= 0 || targetRate > inputRate) {
            throw new Error("targetRate must be positive and no greater than inputRate");
        }
        if (!Number.isInteger(frameSamples) || frameSamples <= 0) {
            throw new Error("frameSamples must be a positive integer");
        }

        this.inputRate = inputRate;
        this.targetRate = targetRate;
        this.frameSamples = frameSamples;
        this.step = inputRate / targetRate;
        this.position = 0;
        this.source = new Float32Array(0);
        this.pending = [];
        this.totalInputSamples = 0;
        this.totalOutputSamples = 0;
    }

    push(channels) {
        if (!channels?.length || channels[0].length === 0) return [];
        const length = channels[0].length;
        if (channels.some((channel) => channel.length !== length)) {
            throw new Error("All audio channels must contain the same sample count");
        }

        const mono = new Float32Array(length);
        for (let sampleIndex = 0; sampleIndex < length; sampleIndex += 1) {
            let mixed = 0;
            for (let channelIndex = 0; channelIndex < channels.length; channelIndex += 1) {
                mixed += channels[channelIndex][sampleIndex];
            }
            mono[sampleIndex] = mixed / channels.length;
        }
        this.totalInputSamples += length;

        const combined = new Float32Array(this.source.length + mono.length);
        combined.set(this.source);
        combined.set(mono, this.source.length);
        const emitted = [];
        while (this.position < combined.length - 1) {
            const leftIndex = Math.floor(this.position);
            const fraction = this.position - leftIndex;
            const sample = combined[leftIndex] +
                ((combined[leftIndex + 1] - combined[leftIndex]) * fraction);
            this.#append(sample, emitted);
            this.position += this.step;
        }

        const consumed = Math.min(Math.floor(this.position), Math.max(combined.length - 1, 0));
        this.source = combined.slice(consumed);
        this.position -= consumed;
        return emitted;
    }

    flush() {
        const emitted = [];
        const targetTotal = Math.round(
            this.totalInputSamples * this.targetRate / this.inputRate,
        );
        const lastSample = this.source.length > 0
            ? this.source[this.source.length - 1]
            : 0;
        while (this.totalOutputSamples < targetTotal) {
            const leftIndex = Math.min(Math.floor(this.position), Math.max(this.source.length - 1, 0));
            const rightIndex = Math.min(leftIndex + 1, Math.max(this.source.length - 1, 0));
            const fraction = this.position - Math.floor(this.position);
            const left = this.source[leftIndex] ?? lastSample;
            const right = this.source[rightIndex] ?? lastSample;
            this.#append(left + ((right - left) * fraction), emitted);
            this.position += this.step;
        }
        if (this.pending.length > 0) {
            emitted.push(Int16Array.from(this.pending));
            this.pending = [];
        }
        this.source = new Float32Array(0);
        return emitted;
    }

    #append(sample, emitted) {
        const clamped = Math.max(-1, Math.min(1, sample));
        this.pending.push(Math.round(clamped < 0 ? clamped * 32768 : clamped * 32767));
        this.totalOutputSamples += 1;
        if (this.pending.length === this.frameSamples) {
            emitted.push(Int16Array.from(this.pending));
            this.pending = [];
        }
    }
}

const WorkletBase = globalThis.AudioWorkletProcessor ?? class {
    constructor() {
        this.port = { onmessage: null, postMessage() {} };
    }
};

class SonaPcmCaptureProcessor extends WorkletBase {
    constructor(options) {
        super();
        const inputRate = options?.processorOptions?.inputSampleRate ?? globalThis.sampleRate;
        this.resampler = new StreamingPcm16Resampler(inputRate);
        this.port.onmessage = (event) => {
            if (event.data?.type !== "flush") return;
            this.#postFrames(this.resampler.flush());
            this.port.postMessage({ type: "flushed" });
        };
    }

    process(inputs) {
        const channels = inputs[0];
        if (channels?.length) {
            this.#postFrames(this.resampler.push(channels));
        }
        return true;
    }

    #postFrames(frames) {
        for (const frame of frames) {
            this.port.postMessage({ type: "audio", buffer: frame.buffer }, [frame.buffer]);
        }
    }
}

if (typeof globalThis.registerProcessor === "function") {
    globalThis.registerProcessor("sona-pcm-capture", SonaPcmCaptureProcessor);
}
