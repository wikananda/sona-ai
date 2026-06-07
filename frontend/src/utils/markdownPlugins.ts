type MarkdownNode = {
    type?: string;
    value?: string;
    children?: MarkdownNode[];
};

export function remarkHtmlLineBreaks() {
    return function transform(tree: MarkdownNode) {
        replaceHtmlBreaks(tree);
    };
}

function replaceHtmlBreaks(node: MarkdownNode) {
    if (!node.children) return;

    node.children = node.children.map((child) => {
        if (
            child.type === "html" &&
            typeof child.value === "string" &&
            /^<br\s*\/?>$/i.test(child.value.trim())
        ) {
            return { type: "break" };
        }

        replaceHtmlBreaks(child);
        return child;
    });
}
