<template>
  <div class="markdown-body" v-html="renderedHtml"></div>
</template>
<script setup>
import { ref, watch, computed, onMounted, defineProps } from "vue";
import { marked } from "marked";
import "github-markdown-css/github-markdown.css";

const props = defineProps({
  markdown: String,
});
const markdown = ref(props.markdown);
watch(
  () => props.markdown,
  (newValue, oldValue) => {
    markdown.value = newValue;
  }
);

// 通过 computed 属性来动态计算渲染后的 HTML 内容
const renderedHtml = computed(() => {
  return marked(markdown.value); // 使用 marked 转换为 HTML
});
</script>

<style lang="scss">
.markdown-body {
  padding: 0 1.35em;
  background-color: var(--bg-white-blue-color);
  border-radius: 8px;
  font-family: "Segoe UI";
  max-height: 8.505em;
  overflow: auto;
  &::-webkit-scrollbar-thumb {
    background-color: #fff;
  }
  &:hover {
    &::-webkit-scrollbar-thumb {
      background-color: #e4e7ff;
    }
  }
  table {
    // margin: 0 auto;
    width: 100%;
    display: inline-table;
    tr {
      background-color: var(--bg-white-blue-color);
    }
  }
}
</style>
