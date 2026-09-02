const FLUSH_TIMEOUT_MS = 3000;

export interface LivePcmCapture {
    finish: () => Promise<void>;
    abort: () => Promise<void>;
}

export async function startLivePcmCapture(
    stream: MediaStream,
    onFrame: (frame: ArrayBuffer) => void,
): Promise<LivePcmCapture> {
    if (typeof AudioWorkletNode === "undefined") {
        throw new Error("Realtime audio streaming is not supported in this browser.");
    }

    const context = new AudioContext();
    let closed = false;
    try {
        await context.audioWorklet.addModule("/pcm-capture-worklet.mjs");
        const source = context.createMediaStreamSource(stream);
        const worklet = new AudioWorkletNode(context, "sona-pcm-capture", {
            numberOfInputs: 1,
            numberOfOutputs: 1,
            outputChannelCount: [1],
            processorOptions: { inputSampleRate: context.sampleRate },
        });
        const silentOutput = context.createGain();
        silentOutput.gain.value = 0;
        source.connect(worklet);
        worklet.connect(silentOutput);
        silentOutput.connect(context.destination);

        worklet.port.addEventListener("message", (event: MessageEvent) => {
            if (event.data?.type === "audio" && event.data.buffer instanceof ArrayBuffer) {
                onFrame(event.data.buffer);
            }
        });
        worklet.port.start();
        await context.resume();

        const cleanup = async () => {
            if (closed) return;
            closed = true;
            source.disconnect();
            worklet.disconnect();
            silentOutput.disconnect();
            await context.close().catch(() => undefined);
        };

        return {
            finish: async () => {
                if (closed) return;
                await new Promise<void>((resolve) => {
                    const timeout = window.setTimeout(resolve, FLUSH_TIMEOUT_MS);
                    const handleMessage = (event: MessageEvent) => {
                        if (event.data?.type !== "flushed") return;
                        window.clearTimeout(timeout);
                        worklet.port.removeEventListener("message", handleMessage);
                        resolve();
                    };
                    worklet.port.addEventListener("message", handleMessage);
                    worklet.port.postMessage({ type: "flush" });
                });
                await cleanup();
            },
            abort: cleanup,
        };
    } catch (error) {
        await context.close().catch(() => undefined);
        throw error;
    }
}
