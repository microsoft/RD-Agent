<template>
  <div class="markdown-body" v-html="renderedHtml"></div>
</template>

<script setup>
import { ref, onMounted, watch } from "vue";
import "katex/dist/katex.min.css";

const props = defineProps({
  content: {
    type: String,
    required: true,
  },
});

const renderedHtml = ref("");
let md = null;
let katexEngine = null;

const normalizeMathBlockInnerContent = (content) => {
  if (!content || typeof content !== "string") {
    return content;
  }

  return content
    .replace(/\r\n?/g, "\n")
    .replace(/\\text\{([\s\S]*?)\}/g, (match, inner) => {
      return `\\text{${inner.replace(/\s*\n\s*/g, " ").trim()}}`;
    })
    .replace(/\s*\n+\s*/g, " ")
    .trim();
};

const wrapBareLatexBlocks = (content) => {
  if (!content || !content.includes("\\begin{")) {
    return content;
  }

  return content.replace(
    /(^|\n)(\s*)(\\begin\{([a-zA-Z*]+)\}[\s\S]*?\\end\{\4\})(?=\s*(?:\n|$))/g,
    (match, lineStart, indent, block) => {
      const trimmedBlock = block.trim();

      if (
        trimmedBlock.startsWith("$$") ||
        trimmedBlock.startsWith("\\[") ||
        trimmedBlock.startsWith("\\(")
      ) {
        return match;
      }

      return `${lineStart}${indent}$$\n${trimmedBlock}\n$$`;
    }
  );
};

const normalizeMathDelimiters = (content) => {
  if (!content || typeof content !== "string") {
    return content;
  }

  return content
    .replace(/\$\$([\s\S]+?)\$\$/g, (match, inner) => {
      return `$$\n${normalizeMathBlockInnerContent(inner)}\n$$`;
    })
    .replace(/\\\[([\s\S]+?)\\\]/g, (match, inner) => {
      return `\\[${normalizeMathBlockInnerContent(inner)}\\]`;
    });
};

const preprocessMathContent = (content) => {
  return normalizeMathDelimiters(wrapBareLatexBlocks(content));
};

const extractStandaloneMath = (content) => {
  if (!content || typeof content !== "string") {
    return null;
  }

  const trimmedContent = preprocessMathContent(content).trim();
  const dollarBlockMatch = trimmedContent.match(/^\$\$([\s\S]+)\$\$$/);

  if (dollarBlockMatch) {
    return {
      displayMode: true,
      formula: normalizeMathBlockInnerContent(dollarBlockMatch[1]),
    };
  }

  const bracketBlockMatch = trimmedContent.match(/^\\\[([\s\S]+)\\\]$/);

  if (bracketBlockMatch) {
    return {
      displayMode: true,
      formula: normalizeMathBlockInnerContent(bracketBlockMatch[1]),
    };
  }

  const inlineMatch = trimmedContent.match(/^\\\(([\s\S]+)\\\)$/);

  if (inlineMatch) {
    return {
      displayMode: false,
      formula: normalizeMathBlockInnerContent(inlineMatch[1]),
    };
  }

  if (/^\\begin\{([a-zA-Z*]+)\}[\s\S]*\\end\{\1\}$/.test(trimmedContent)) {
    return {
      displayMode: true,
      formula: normalizeMathBlockInnerContent(trimmedContent),
    };
  }

  return null;
};

const renderStandaloneMath = (content) => {
  if (!katexEngine) {
    return null;
  }

  const standaloneMath = extractStandaloneMath(content);

  if (!standaloneMath || !standaloneMath.formula) {
    return null;
  }

  try {
    return katexEngine.renderToString(standaloneMath.formula, {
      displayMode: standaloneMath.displayMode,
      throwOnError: false,
      strict: "ignore",
      macros: {
        "\\RR": "\\mathbb{R}",
      },
    });
  } catch (_) {
    return null;
  }
};

const renderContent = (content) => {
  if (!md) {
    return "";
  }

  const standaloneMathHtml = renderStandaloneMath(content);

  if (standaloneMathHtml) {
    return standaloneMathHtml;
  }

  return md.render(preprocessMathContent(content));
};

onMounted(async () => {
  try {
    const [{ default: markdownit }, hljsModule, katex, texmathModule] =
      await Promise.all([
        import("markdown-it"),
        import("highlight.js"),
        import("katex"),
        import("markdown-it-texmath"),
      ]);

    const hljs = hljsModule.default || hljsModule;
    const texmath = texmathModule.default || texmathModule;
    katexEngine = katex.default || katex;

    md = markdownit({
      highlight: function (str, lang) {
        if (lang && hljs.getLanguage(lang)) {
          try {
            const highlighted = hljs.highlight(str, { language: lang }).value;
            return `<pre><code class="hljs language-${lang}">${highlighted}</code></pre>`;
          } catch (_) {}
        }
        return `<pre><code class="hljs">${md.utils.escapeHtml(
          str
        )}</code></pre>`;
      },
    });

    // 修复列表和段落渲染逻辑
    md.renderer.rules.list_item_open = () => "<li>";
    md.renderer.rules.list_item_close = () => "</li>";
    md.renderer.rules.paragraph_open = (tokens, idx) => {
      const parentToken = tokens[idx - 1];
      return parentToken && parentToken.type === "list_item_open" ? "" : "<p>";
    };
    md.renderer.rules.paragraph_close = (tokens, idx) => {
      const parentToken = tokens[idx - 1];
      return parentToken && parentToken.type === "list_item_close"
        ? ""
        : "</p>";
    };

    md.use(texmath, {
      engine: katexEngine,
      delimiters: ["dollars", "brackets"],
      katexOptions: {
        throwOnError: false,
        strict: "ignore",
        macros: {
          "\\RR": "\\mathbb{R}",
        },
      },
    });

    renderedHtml.value = renderContent(props.content);
  } catch (e) {
    console.error("MarkdownPreview 初始化失败:", e);
  }
});

watch(
  () => props.content,
  (newVal) => {
    if (md) {
      renderedHtml.value = renderContent(newVal);
    }
  }
);
</script>

<style scoped>
.markdown-body {
  font-family: "Microsoft YaHei";
  font-size: 1em;
  line-height: 180%;
  background-color: #fff;
  max-height: unset;
  padding: 0;
}

.markdown-body li {
  list-style: unset;
}
</style>
