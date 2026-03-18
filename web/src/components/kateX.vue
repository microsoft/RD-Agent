<template>
  <div class="math-box" ref="katexContainer"></div>
</template>
<script setup>
import { ref, watch, onMounted, defineProps } from "vue";
import "katex/dist/katex.min.css";
import katex from "katex";

const props = defineProps({
  formula: String,
});
const katexContainer = ref(null);
watch(
  () => props.formula,
  (newValue, oldValue) => {
    katex.render(newValue, katexContainer.value, {
      throwOnError: false, // 避免在公式错误时抛出异常
    });
  }
);

onMounted(() => {
  if (katexContainer.value) {
    katex.render(props.formula, katexContainer.value, {
      throwOnError: false, // 避免在公式错误时抛出异常
    });
  }
});
</script>

<style scoped lang="scss">
.math-box {
  display: flex;
  justify-content: center;
}
</style>
