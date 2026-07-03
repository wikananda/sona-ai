import { useEffect, useRef } from "react";

export function usePolling(
    fn: () => void | Promise<void>,
    intervalMs: number,
    enabled: boolean,
) {
    const fnRef = useRef(fn);

    useEffect(() => {
        fnRef.current = fn;
    }, [fn]);

    useEffect(() => {
        if (!enabled) return;

        const intervalId = window.setInterval(() => {
            void fnRef.current();
        }, intervalMs);

        return () => window.clearInterval(intervalId);
    }, [enabled, intervalMs]);
}
