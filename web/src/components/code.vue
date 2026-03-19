<template>
  <div class="code">
    <div class="code-content">
      <SvgIcon
        @click="fullScreen"
        class="expand-icon"
        color="#2b2b2b"
        name="fullscreen"
      ></SvgIcon>
      <SvgIcon
        @click="copy"
        class="copy-icon"
        color="#2b2b2b"
        name="copy"
      ></SvgIcon>
      <div
        class="md-code"
        :class="{
          'full-dev': fullScreenFlag && developer,
          'full-no-dev': fullScreenFlag && !developer,
          'no-full-dev': !fullScreenFlag && developer,
          'no-full-no-dev': !fullScreenFlag && !developer,
        }"
      >
        <pre
          class="code-display language-python"
        ><code v-html="highlightedCode"></code>
        </pre>
      </div>
    </div>
  </div>
</template>
<script setup>
import { ref, watch, onMounted, defineProps, defineEmits, nextTick } from "vue";
// main.js 或 Vue 组件内部
import "prismjs";
import "prismjs/components/prism-python.min.js"; // 导入Python的语言支持

import { ElMessage } from "element-plus";

const props = defineProps({
  markdown: String,
  developer: Boolean,
  fullscreen: Boolean,
});

const emit = defineEmits(["fullScreen"]);
const markdown = ref(props.markdown);
const developer = ref(props.developer);
const highlightedCode = ref("");
const fullScreenFlag = ref(props.fullscreen);
watch(
  () => [props.markdown, props.developer, props.fullscreen],
  (newValue, oldValue) => {
    markdown.value = newValue[0];
    developer.value = newValue[1];
    fullScreenFlag.value = newValue[2];
    highlightCode();
  }
);

const highlightCode = () => {
  // 使用 PrismJS 对 Python 代码进行高亮
  highlightedCode.value = Prism.highlight(
    markdown.value,
    Prism.languages.python,
    "python"
  );
};

const copy = () => {
  navigator.clipboard.writeText(markdown.value);
  ElMessage({
    message: "Copy Success.",
    type: "success",
    plain: true,
  });
};

const fullScreen = () => {
  fullScreenFlag.value = !fullScreenFlag.value;
  emit("fullScreen", fullScreenFlag.value);
};

onMounted(() => {
  // const codeBlock = document.querySelector("pre code");
  // hljs.highlightElement(codeBlock);
  highlightCode();
});
</script>

<style lang="scss">
.code {
  width: 100%;
  .code-content {
    border-radius: 11px;
    background: var(--bg-white);
    padding: 1.35em 0 0 0.9em;
    box-sizing: border-box;
    overflow-y: hidden;
    position: relative;

    .expand-icon {
      position: absolute;
      right: 3.6em;
      top: 0.45em;
      cursor: pointer;
      opacity: 0.5;
      width: 1.24em;
      &:hover {
        opacity: 0.8;
      }
    }

    .copy-icon {
      position: absolute;
      right: 1.35em;
      top: 0.45em;
      cursor: pointer;
      opacity: 0.5;
      width: 1.24em;
      &:hover {
        opacity: 0.8;
      }
    }
    .md-code {
      height: calc(100vh - 28.35em);
      max-width: 100%;
      font-size: 0.9em;
      line-height: 140%;
      overflow: auto;
      // &:hover {
      //   overflow: auto;
      // }
      &::-webkit-scrollbar-thumb {
        background-color: #fff;
      }
      &:hover {
        &::-webkit-scrollbar-thumb {
          background-color: #e4e7ff;
        }
      }

      &.full-dev {
        height: calc(100vh - 26.13em);
      }
      &.full-no-dev {
        height: calc(100vh - 19.98em);
      }
      &.no-full-dev {
        height: calc(100vh - 32.8em);
      }
      &.no-full-no-dev {
        height: calc(100vh - 26.1em);
      }
      pre {
        background: transparent;
        border: 0px;
        display: inline;
        font-size: 0.9em;
        margin: 0px;
        overflow: auto;
        padding: 0px;
        white-space: pre;
        word-break: normal;
        overflow-wrap: normal;
        font-family: "consolas", monospace;
        &::-webkit-scrollbar-thumb {
          background-color: #fff;
        }
        &:hover {
          &::-webkit-scrollbar-thumb {
            background-color: #e4e7ff;
          }
        }
      }
      code {
        font-family: "consolas", monospace;
      }
    }
  }
}
</style>
