import type { SpeakerSegment } from "@/src/api/sonaApi";
import { formatClockTime } from "@/src/utils/formatters";

interface ExportTranscriptPdfParams {
    recordingName: string;
    segments: SpeakerSegment[];
}

interface ExportSummaryPdfParams {
    recordingName: string;
    summary: string;
}

type JsPdfDocument = InstanceType<typeof import("jspdf").jsPDF>;
type PdfFontStyle = "normal" | "bold";
type PdfColor = [number, number, number];

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
            `${formatClockTime(segment.start)} - ${formatClockTime(segment.end)}`,
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

    const bottomY = pageHeight - PAGE_MARGIN;

    const addPage = () => {
        doc.addPage();
        cursorY = PAGE_MARGIN;
    };

    const addPageIfNeeded = (height: number) => {
        if (cursorY + height <= bottomY) return;

        addPage();
    };

    const measureLines = (
        text: string,
        maxLineWidth: number,
        fontSize: number,
        fontStyle: PdfFontStyle,
    ): string[] => {
        doc.setFont("helvetica", fontStyle);
        doc.setFontSize(fontSize);

        const wrappedText = doc.splitTextToSize(text, maxLineWidth);
        return Array.isArray(wrappedText) ? wrappedText : [wrappedText];
    };

    const addText = (
        text: string,
        options: {
            fontSize?: number;
            fontStyle?: PdfFontStyle;
            textColor?: PdfColor;
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

        const wrappedLines = measureLines(
            safeText,
            maxLineWidth,
            fontSize,
            fontStyle,
        );

        addPageIfNeeded(before + lineHeight);
        cursorY += before;
        wrappedLines.forEach((line) => {
            addPageIfNeeded(lineHeight);
            doc.text(line, PAGE_MARGIN + indent, cursorY);
            cursorY += lineHeight;
        });
        cursorY += after;
    };

    const addTable = (headers: string[], rows: string[][]) => {
        const columnCount = headers.length;
        if (columnCount === 0) return;

        cursorY += 4;

        const columnWidths = buildColumnWidths(columnCount, contentWidth);
        const paddingX = 6;
        const paddingY = 7;
        const tableLineHeight = 12;
        const headerFontSize = 9;
        const bodyFontSize = 9;
        const maxCellWidths = columnWidths.map((width) => Math.max(20, width - paddingX * 2));

        const renderRowChunk = (
            cellLineSets: string[][],
            startLine: number,
            lineCount: number,
            isHeader: boolean,
        ) => {
            const rowHeight = Math.max(26, lineCount * tableLineHeight + paddingY * 2);

            if (cursorY + rowHeight > bottomY) {
                addPage();
            }

            let cursorX = PAGE_MARGIN;
            columnWidths.forEach((columnWidth, columnIndex) => {
                if (isHeader) {
                    doc.setFillColor(244, 244, 245);
                } else {
                    doc.setFillColor(255, 255, 255);
                }
                doc.setDrawColor(212, 212, 216);
                doc.setLineWidth(0.5);
                doc.rect(cursorX, cursorY, columnWidth, rowHeight, "FD");

                doc.setFont("helvetica", isHeader ? "bold" : "normal");
                doc.setFontSize(isHeader ? headerFontSize : bodyFontSize);
                const textColor: PdfColor = isHeader ? [63, 63, 70] : [39, 39, 42];
                doc.setTextColor(...textColor);

                const visibleLines = cellLineSets[columnIndex].slice(
                    startLine,
                    startLine + lineCount,
                );
                visibleLines.forEach((line, lineIndex) => {
                    doc.text(
                        line,
                        cursorX + paddingX,
                        cursorY + paddingY + (lineIndex + 1) * tableLineHeight - 2,
                    );
                });

                cursorX += columnWidth;
            });

            cursorY += rowHeight;
        };

        const renderHeader = () => {
            const headerLineSets = headers.map((header, index) => (
                measureLines(
                    cleanMarkdownInline(header),
                    maxCellWidths[index],
                    headerFontSize,
                    "bold",
                )
            ));
            const headerLineCount = Math.max(
                1,
                ...headerLineSets.map((lines) => lines.length),
            );
            renderRowChunk(headerLineSets, 0, headerLineCount, true);
        };

        renderHeader();

        rows.forEach((row) => {
            const cellLineSets = headers.map((_, index) => (
                wrapTableCell(row[index] ?? "", maxCellWidths[index], bodyFontSize)
            ));
            const totalLineCount = Math.max(
                1,
                ...cellLineSets.map((lines) => lines.length),
            );
            let startLine = 0;

            while (startLine < totalLineCount) {
                const availableLineCount = Math.floor(
                    Math.max(0, bottomY - cursorY - paddingY * 2) / tableLineHeight,
                );
                if (availableLineCount < 1) {
                    addPage();
                    renderHeader();
                    continue;
                }

                const lineCount = Math.min(totalLineCount - startLine, availableLineCount);
                renderRowChunk(cellLineSets, startLine, lineCount, false);
                startLine += lineCount;

                if (startLine < totalLineCount) {
                    addPage();
                    renderHeader();
                }
            }
        });

        cursorY += 8;
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
        addTable,
        addDocumentHeader,
    };

    function wrapTableCell(text: string, maxLineWidth: number, fontSize: number): string[] {
        const parts = text.split(/<br\s*\/?>/i);
        const lines = parts.flatMap((part) => {
            const cleanedPart = cleanMarkdownInline(part)
                .replace(/^\s*[-*]\s+/, "• ")
                .trim();
            if (!cleanedPart) return [];

            return measureLines(cleanedPart, maxLineWidth, fontSize, "normal");
        });

        return lines.length ? lines : [""];
    }
}

function writeMarkdownSummary(
    writer: ReturnType<typeof createPdfWriter>,
    summary: string,
) {
    const lines = summary.split("\n");

    for (let index = 0; index < lines.length; index += 1) {
        const tableBlock = parseMarkdownTable(lines, index);
        if (tableBlock) {
            writer.addTable(tableBlock.headers, tableBlock.rows);
            index = tableBlock.endIndex;
            continue;
        }

        const line = lines[index];
        const trimmedLine = line.trim();
        if (!trimmedLine) {
            writer.addText("", { after: 5 });
            continue;
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
            continue;
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
            continue;
        }

        const unorderedListMatch = trimmedLine.match(/^[-*]\s+(.+)$/);
        if (unorderedListMatch) {
            writer.addText(`• ${cleanMarkdownInline(unorderedListMatch[1])}`, {
                fontSize: 11,
                indent: 12,
                after: 3,
            });
            continue;
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
            continue;
        }

        writer.addText(cleanMarkdownInline(trimmedLine), {
            fontSize: 11,
            after: 6,
        });
    }
}

function parseMarkdownTable(
    lines: string[],
    startIndex: number,
): { headers: string[]; rows: string[][]; endIndex: number } | null {
    const headerLine = lines[startIndex]?.trim();
    const separatorLine = lines[startIndex + 1]?.trim();
    if (!headerLine || !separatorLine) return null;
    if (!isMarkdownTableRow(headerLine) || !isMarkdownTableSeparator(separatorLine)) {
        return null;
    }

    const headers = splitMarkdownTableRow(headerLine);
    const separatorCells = splitMarkdownTableRow(separatorLine);
    if (headers.length < 2 || separatorCells.length !== headers.length) return null;

    const rows: string[][] = [];
    let index = startIndex + 2;
    while (index < lines.length && isMarkdownTableRow(lines[index])) {
        const row = splitMarkdownTableRow(lines[index]);
        if (row.length === headers.length) {
            rows.push(row);
        }
        index += 1;
    }

    return {
        headers,
        rows,
        endIndex: index - 1,
    };
}

function isMarkdownTableRow(line: string): boolean {
    return line.trim().includes("|");
}

function isMarkdownTableSeparator(line: string): boolean {
    const cells = splitMarkdownTableRow(line);
    return (
        cells.length >= 2 &&
        cells.every((cell) => /^:?-{3,}:?$/.test(cell.replace(/\s/g, "")))
    );
}

function splitMarkdownTableRow(line: string): string[] {
    return line
        .trim()
        .replace(/^\|/, "")
        .replace(/\|$/, "")
        .split("|")
        .map((cell) => cell.trim());
}

function buildColumnWidths(columnCount: number, contentWidth: number): number[] {
    if (columnCount === 2) {
        return [contentWidth * 0.3, contentWidth * 0.7];
    }

    const columnWidth = contentWidth / columnCount;
    return Array.from({ length: columnCount }, () => columnWidth);
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

function buildFilename(recordingName: string, type: "summary" | "transcript"): string {
    const baseName = recordingName
        .replace(/\.[a-z0-9]+$/i, "")
        .replace(/[^a-z0-9]+/gi, "-")
        .replace(/^-+|-+$/g, "")
        .toLowerCase();

    return `${baseName || "recording"}-${type}.pdf`;
}
