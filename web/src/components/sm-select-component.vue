<template>
  <div class="select-box">
    <div class="select-div gradient-border" @click.stop="changePopover">
      <div>
        <div class="checked-item" v-if="scenarioChecked">
          <span
            v-if="showStatus"
            :class="{
              success: scenarioChecked.decision,
              fail: !scenarioChecked.decision,
            }"
          ></span>
          <span :class="{ omit: showStatus }">{{ scenarioChecked.name }}</span>
        </div>
        <div class="checked-item checked-placeholder" v-else>
          <span>{{ placeholder }}</span>
        </div>
      </div>
      <span class="down-arrow"></span>
    </div>
    <div
      class="select-drop-panel gradient-border"
      :style="{
        '--height':
          optionCount <= 4
            ? optionCount * 3.15 * 16 +
              Math.max(optionCount - 2, 0) * 2 +
              'px'
            : '16em',
      }"
      v-show="showPopover"
    >
      <div class="select-drop-list">
        <div
          class="select-drop-item"
          @click="choiceScenario(item, index)"
          v-for="(item, index) in scenarioList"
          :key="index"
          :style="{ 'border-color': item.color }"
          :class="{ active: scenarioCheckedIndex == index }"
        >
          <span
            v-if="showStatus"
            :class="{
              success: item.decision,
              fail: !item.decision,
            }"
          ></span>
          <span :class="{ omit: showStatus }">{{ item.name }}</span>
        </div>
      </div>
    </div>
  </div>
</template>
<script setup>
import {
  computed,
  ref,
  watch,
  onMounted,
  defineProps,
  defineEmits,
  nextTick,
  onUnmounted,
} from "vue";
const props = defineProps({
  scenarioList: Array,
  scenarioIndex: Number,
  showStatus: Boolean,
  placeholder: {
    type: String,
    default: "",
  },
});

const emit = defineEmits(["scenarioCheckedItem"]);
const scenarioCheckedIndex = ref(props.scenarioIndex);
const scenarioList = ref(props.scenarioList);
const scenarioChecked = ref(null);
const optionCount = computed(() => scenarioList.value?.length || 0);
if (scenarioList.value) {
  scenarioChecked.value = scenarioList.value[scenarioCheckedIndex.value];
}
const showStatus = ref(props.showStatus);
const placeholder = ref(props.placeholder);
watch(
  () => [props.scenarioList, props.scenarioIndex, props.showStatus, props.placeholder],
  (newValue, oldValue) => {
    scenarioList.value = newValue[0];
    scenarioCheckedIndex.value = newValue[1];
    showStatus.value = newValue[2];
    placeholder.value = newValue[3];
    if (scenarioList.value) {
      scenarioChecked.value = scenarioList.value[scenarioCheckedIndex.value];
    }
  }
);

const showPopover = ref(false);
const changePopover = () => {
  if (showPopover.value) {
    showPopover.value = false;
  } else {
    showPopover.value = true;
  }
};
const choiceScenario = (item, index) => {
  scenarioCheckedIndex.value = index;
  scenarioChecked.value = item;
  showPopover.value = false;
  emit("scenarioCheckedItem", {
    scenarioCheckedIndex: scenarioCheckedIndex.value,
    scenarioChecked: scenarioChecked.value,
  });
};
const globalClickHandler = () => {
  showPopover.value = false;
};
onMounted(() => {
  document.addEventListener("click", globalClickHandler);
});

// 在组件被卸载前移除全局点击事件监听
onUnmounted(() => {
  document.removeEventListener("click", globalClickHandler);
});
</script>

<style scoped lang="scss">
.select-box {
  position: relative;
  .select-div {
    display: flex;
    height: 2.7em;
    justify-content: space-between;
    align-items: center;
    border-radius: 9px;
    --border-radius: 11px;
    --border-width: 2px;
    cursor: pointer;
    .down-arrow {
      width: 1.35em;
      height: 1.35em;
      background: url(@/assets/images/down-arrow.svg) no-repeat;
      background-size: contain;
      position: absolute;
      right: 1.35em;
    }
    .checked-item {
      box-sizing: border-box;
      padding: 0.5625em 1.98em 0.5625em;
      display: flex;
      align-items: center;
      .select-item-icon {
        margin-right: 0.9em;
      }
      span {
        color: var(--text-color);
        font-size: 1.0125em;
        line-height: 200%;
        margin-top: -2px;
      }
    }
  }
  .select-drop-panel {
    --height: 16em;
    width: 100%;
    height: calc(var(--height) + 4px);
    max-height: calc(20em + 4px);
    position: absolute;
    left: 0;
    top: 2.745em;
    cursor: pointer;
    background-color: var(--bg-white);
    border-radius: 11px;
    z-index: 99;
    overflow: hidden;
    // box-shadow: 8px 11px 30px 0px var(--wg-shadow-color);
    .select-drop-list {
      width: calc(100% - 4px);
      height: var(--height);
      max-height: 20em;
      overflow-y: auto;
      position: absolute;
      left: 2px;
      top: 2px;
      z-index: 1;
      background-color: var(--bg-white);
      border-radius: 11px;
    }
    .select-drop-item {
      padding: 0.5625em 1.98em 0.5625em;
      border-bottom: 2px solid #2e65ff;
      display: flex;
      align-items: center;
      height: 3.15em;
      box-sizing: border-box;

      &:last-child {

      &.checked-placeholder {
        span {
          color: #868ca5;
        }
      }
        border-bottom: none;
      }
      .select-item-icon {
        margin-right: 0.9em;
      }
      span {
        color: var(--text-color);
        font-size: 1.0125em;
        line-height: 200%;
        margin-top: -2px;
      }
      &:hover,
      &.active {
        background-color: var(--card-bg-hover-color);
      }
    }
  }
  .success {
    display: inline-block;
    width: 1.125em;
    height: 1.125em;
    background: url(@/assets/playground-images/process-checked.svg) no-repeat;
    background-size: contain;
    vertical-align: middle;
    margin-right: 0.45em;
  }
  .fail {
    display: inline-block;
    width: 1.125em;
    height: 1.125em;
    background: url(@/assets/playground-images/process-fail-checked.svg)
      no-repeat;
    background-size: contain;
    vertical-align: middle;
    margin-right: 0.45em;
  }
  .omit {
    display: inline-block;
    width: 410px;
    white-space: nowrap; /* 不换行 */
    overflow: hidden; /* 超出部分隐藏 */
    text-overflow: ellipsis; /* 显示省略号 */
  }
}
</style>
