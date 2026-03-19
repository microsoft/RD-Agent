<template>
  <div class="step-box">
    <div
      v-for="(item, index) in stepList"
      :key="index"
      class="step-card"
      :class="{
        'default-color': currentIndex < index,
        'current-color': currentIndex == index,
        'active-color': currentIndex > index,
      }"
    >
      <SvgIcon v-if="index != 0" class="step-icon" name="right-arrow"></SvgIcon>
      <span class="step-label"
        ><i>{{ index + 1 }}.</i>{{ item }}</span
      >
    </div>
  </div>
</template>
<script setup>
import { ref, watch, defineProps, nextTick } from "vue";
const props = defineProps({
  activeIndex: Number,
});
const currentIndex = ref(props.activeIndex);
const stepList = ["Start", "Input Information", "Summary"];
</script>

<style scoped lang="scss">
.step-box {
  display: flex;
  justify-content: center;
  .step-card {
    display: flex;
    align-items: center;
  }
  .step-label {
    display: flex;
    padding: 0.65em 1em;
    justify-content: center;
    align-items: center;
    text-align: center;
    font-size: 1.25em;
    font-weight: 600;
    line-height: 200%;
    border-radius: 999px;
    color: var(--step-default-color);
    border: 2px solid var(--step-default-color);
    i {
      margin-right: 0.75em;
      font-style: normal;
    }
  }
  .step-icon {
    width: 2.15em;
    height: 1.25em;
    margin: 0 1.25em;
  }
  .default-color {
    .step-label {
      border-color: var(--step-default-color);
      color: var(--step-default-color);
    }
    .step-icon {
      color: var(--step-default-color) !important;
    }
  }
  .current-color {
    .step-label {
      color: var(--step-current-color);
      border-color: var(--step-current-color);
    }
    .step-icon {
      color: var(--step-current-color) !important;
    }
  }
  .active-color {
    .step-label {
      color: var(--step-active-color);
      border-color: var(--step-active-color);
    }
    .step-icon {
      color: var(--step-active-color) !important;
    }
  }
}
</style>
