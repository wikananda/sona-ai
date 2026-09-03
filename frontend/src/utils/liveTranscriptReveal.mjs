/**
 * Split transcript text into display tokens while preserving the whitespace
 * following each word.
 *
 * @param {string} text
 * @returns {string[]}
 */
export function tokenizeTranscript(text) {
    return text.match(/\S+\s*/g) ?? [];
}

/**
 * Return the number of unchanged tokens at the start of two revisions.
 *
 * @param {string[]} previous
 * @param {string[]} next
 * @returns {number}
 */
export function commonTokenPrefix(previous, next) {
    const length = Math.min(previous.length, next.length);
    let index = 0;
    while (index < length && previous[index] === next[index]) {
        index += 1;
    }
    return index;
}

/**
 * Reveal slowly near the live edge and catch up quickly after a large batch.
 *
 * @param {number} visible
 * @param {number} total
 * @returns {number}
 */
export function nextVisibleTokenCount(visible, total) {
    const remaining = Math.max(total - visible, 0);
    const step = remaining > 18 ? 4 : remaining > 8 ? 2 : 1;
    return Math.min(visible + step, total);
}
