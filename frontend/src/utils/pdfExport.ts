import type { SpeakerSegment } from "@/src/api/sonaApi";

interface ExportTranscriptPdfParams {
    recordingName: string;
    segments: SpeakerSegment[];
}

interface ExportSummaryPdfParams {
    recordingName: string;
    summary: string;
}

type JsPdfDocument = InstanceType<typeof import("jspdf").jsPDF>;

const PAGE_MARGIN = 56;
const LINE_HEIGHT = 15;

export async function exportTranscriptPdf({
    recordingName,
    segments,
}: ExportTranscriptPdfParams): Promise<void> {
    const { jsPDF } = await import("jspdf");
    const doc = new jsPDF({ unit: "pt", format: "a4" });
    const writer = createPdfWriter(doc);

    writer.addDocumentHeader("Transcript", recordingName);

    segments.forEach((segment) => {
        const labelParts = [
            segment.speaker?.trim(),
            `${formatTimestamp(segment.start)} - ${formatTimestamp(segment.end)}`,
        ].filter(Boolean);

        writer.addText(labelParts.join("  |  "), {
            fontSize: 9,
            fontStyle: "bold",
            textColor: [82, 82, 91],
            before: 8,
            after: 2,
        });
        writer.addText(segment.text.trim(), {
            fontSize: 11,
            fontStyle: "normal",
            textColor: [39, 39, 42],
        });
    });

    doc.save(buildFilename(recordingName, "transcript"));
}

export async function exportSummaryPdf({
    recordingName,
    summary,
}: ExportSummaryPdfParams): Promise<void> {
    const { jsPDF } = await import("jspdf");
    const doc = new jsPDF({ unit: "pt", format: "a4" });
    const writer = createPdfWriter(doc);

    writer.addDocumentHeader("Summary", recordingName);
    writeMarkdownSummary(writer, summary);

    doc.save(buildFilename(recordingName, "summary"));
}

function createPdfWriter(doc: JsPdfDocument) {
    const pageWidth = doc.internal.pageSize.getWidth();
    const pageHeight = doc.internal.pageSize.getHeight();
    const contentWidth = pageWidth - PAGE_MARGIN * 2;
    let cursorY = PAGE_MARGIN;

    const addPageIfNeeded = (height: number) => {
        if (cursorY + height <= pageHeight - PAGE_MARGIN) return;

        doc.addPage();
        cursorY = PAGE_MARGIN;
    };

    const addText = (
        text: string,
        options: {
            fontSize?: number;
            fontStyle?: "normal" | "bold";
            textColor?: [number, number, number];
            before?: number;
            after?: number;
            indent?: number;
        } = {},
    ) => {
        const safeText = text.trim();
        if (!safeText) {
            cursorY += options.after ?? LINE_HEIGHT;
            return;
        }

        const fontSize = options.fontSize ?? 11;
        const fontStyle = options.fontStyle ?? "normal";
        const lineHeight = Math.max(fontSize + 4, LINE_HEIGHT);
        const indent = options.indent ?? 0;
        const before = options.before ?? 0;
        const after = options.after ?? 6;
        const maxLineWidth = Math.max(24, contentWidth - indent);

        doc.setFont("helvetica", fontStyle);
        doc.setFontSize(fontSize);
        doc.setTextColor(...(options.textColor ?? [39, 39, 42]));

        const wrappedText = doc.splitTextToSize(safeText, maxLineWidth);
        const wrappedLines = Array.isArray(wrappedText) ? wrappedText : [wrappedText];

        addPageIfNeeded(before + lineHeight);
        cursorY += before;
        wrappedLines.forEach((line) => {
            addPageIfNeeded(lineHeight);
            doc.text(line, PAGE_MARGIN + indent, cursorY);
            cursorY += lineHeight;
        });
        cursorY += after;
    };

    const addDocumentHeader = (title: string, recordingName: string) => {
        doc.setProperties({
            title: `${title} - ${recordingName}`,
            subject: `${title} export`,
            creator: "Sona AI",
        });

        addText(title, {
            fontSize: 20,
            fontStyle: "bold",
            textColor: [9, 9, 11],
            after: 4,
        });
        addText(recordingName, {
            fontSize: 12,
            fontStyle: "bold",
            textColor: [63, 63, 70],
            after: 2,
        });
        addText(`Generated ${new Date().toLocaleString()}`, {
            fontSize: 9,
            textColor: [113, 113, 122],
            after: 16,
        });
    };

    return {
        addText,
        addDocumentHeader,
    };
}

function writeMarkdownSummary(
    writer: ReturnType<typeof createPdfWriter>,
    summary: string,
) {
    summary.split("\n").forEach((line) => {
        const trimmedLine = line.trim();
        if (!trimmedLine) {
            writer.addText("", { after: 5 });
            return;
        }

        const headingMatch = trimmedLine.match(/^(#{1,3})\s+(.+)$/);
        if (headingMatch) {
            const level = headingMatch[1].length;
            writer.addText(cleanMarkdownInline(headingMatch[2]), {
                fontSize: level === 1 ? 16 : level === 2 ? 14 : 12,
                fontStyle: "bold",
                textColor: [9, 9, 11],
                before: level === 1 ? 10 : 8,
                after: 4,
            });
            return;
        }

        const boldHeadingMatch = trimmedLine.match(/^\*\*(.+)\*\*$/);
        if (boldHeadingMatch) {
            writer.addText(cleanMarkdownInline(boldHeadingMatch[1]), {
                fontSize: 13,
                fontStyle: "bold",
                textColor: [9, 9, 11],
                before: 8,
                after: 4,
            });
            return;
        }

        const unorderedListMatch = trimmedLine.match(/^[-*]\s+(.+)$/);
        if (unorderedListMatch) {
            writer.addText(`• ${cleanMarkdownInline(unorderedListMatch[1])}`, {
                fontSize: 11,
                indent: 12,
                after: 3,
            });
            return;
        }

        const orderedListMatch = trimmedLine.match(/^(\d+\.)\s+(.+)$/);
        if (orderedListMatch) {
            writer.addText(
                `${orderedListMatch[1]} ${cleanMarkdownInline(orderedListMatch[2])}`,
                {
                    fontSize: 11,
                    indent: 12,
                    after: 3,
                },
            );
            return;
        }

        writer.addText(cleanMarkdownInline(trimmedLine), {
            fontSize: 11,
            after: 6,
        });
    });
}

function cleanMarkdownInline(text: string): string {
    return text
        .replace(/\*\*(.*?)\*\*/g, "$1")
        .replace(/__(.*?)__/g, "$1")
        .replace(/\*(.*?)\*/g, "$1")
        .replace(/_(.*?)_/g, "$1")
        .replace(/`([^`]+)`/g, "$1")
        .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
        .trim();
}

function formatTimestamp(seconds: number): string {
    const totalSeconds = Math.max(0, Math.floor(seconds));
    const minutes = Math.floor(totalSeconds / 60);
    const remainingSeconds = totalSeconds % 60;

    return `${minutes}:${String(remainingSeconds).padStart(2, "0")}`;
}

function buildFilename(recordingName: string, type: "summary" | "transcript"): string {
    const baseName = recordingName
        .replace(/\.[a-z0-9]+$/i, "")
        .replace(/[^a-z0-9]+/gi, "-")
        .replace(/^-+|-+$/g, "")
        .toLowerCase();

    return `${baseName || "recording"}-${type}.pdf`;
}
