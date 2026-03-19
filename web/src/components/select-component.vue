<template>
  <div>
    <div class="select-box">
      <div class="select-div gradient-border" @click.stop="changePopover">
        <div>
          <div class="checked-item" v-if="scenarioChecked">
            <SvgIcon
              class="select-item-icon"
              :name="scenarioChecked.icon"
            ></SvgIcon>
            <span>{{
              scenarioChecked.checkedName
                ? scenarioChecked.checkedName
                : scenarioChecked.name
            }}</span>
          </div>
        </div>
        <span class="down-arrow" :class="{ active: showPopover }"></span>
      </div>
      <div
        class="select-drop-panel gradient-border"
        v-show="showPopover"
        :style="{
          '--height':
            scenarioList.length <= 4
              ? scenarioList.length * 3.375 * 16 +
                (scenarioList.length - 1) +
                'px'
              : '16em',
        }"
      >
        <div class="select-drop-list">
          <div
            class="select-drop-item"
            @click.stop="choiceScenario(item, index)"
            v-for="(item, index) in scenarioList"
            :key="index"
            :style="{ 'border-color': item.color }"
          >
            <div
              class="drop-item-one"
              :class="{ active: scenarioCheckedIndex == index && !item.child }"
            >
              <SvgIcon
                v-if="item.icon"
                class="select-item-icon"
                :name="item.icon"
              ></SvgIcon>
              <span>{{ item.name }}</span>
              <span
                class="down-arrow"
                :class="{ active: showChild }"
                v-if="item.child"
              ></span>
            </div>
            <div v-if="item.child && showChild">
              <div
                class="drop-child-item"
                @click.stop="choiceScenario(item, index, child, index2)"
                v-for="(child, index2) in item.child"
                :key="child.name"
                :style="{ 'border-color': item.color }"
                :class="{ active: scenarioChildCheckedIndex == index2 }"
              >
                <span>{{ child.name }}</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>
<script setup>
import {
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
});
const emit = defineEmits(["scenarioCheckedItem"]);
const scenarioList = ref(props.scenarioList);
const scenarioCheckedIndex = ref(props.scenarioIndex);
const scenarioChildCheckedIndex = ref(-1);
const scenarioChecked = ref(null);
const showChild = ref(false);

const showPopover = ref(false);

watch(
  () => [props.scenarioList, props.scenarioIndex],
  (newValue, oldValue) => {
    scenarioList.value = newValue[0];
    scenarioCheckedIndex.value = newValue[1];

    if (scenarioList.value && scenarioCheckedIndex.value >= 0) {
      scenarioChecked.value = scenarioList.value[scenarioCheckedIndex.value];
    } else {
      scenarioChecked.value = null;
    }
  },
  {
    deep: true,
    immediate: true,
  }
);

const changePopover = () => {
  if (showPopover.value) {
    showPopover.value = false;
  } else {
    showPopover.value = true;
  }
};
const choiceScenario = (item, index, child, index2) => {
  if (item.child && !child) {
    showChild.value = !showChild.value;
    return;
  }

  scenarioCheckedIndex.value = index;
  scenarioChecked.value = item;

  scenarioChildCheckedIndex.value = -1;
  if (child) {
    scenarioChildCheckedIndex.value = index2;
    scenarioChecked.value.checkedName = child.name;
  }
  showPopover.value = false;
  emit("scenarioCheckedItem", {
    scenarioCheckedIndex: scenarioCheckedIndex.value,
    scenarioChecked: scenarioChecked.value,
  });
};
const globalClickHandler = (e) => {
  e.stopPropagation();
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
  margin-top: 1.62em;
  margin-bottom: 2.52em;
  position: relative;
  .select-div {
    display: flex;
    height: 3.375em;
    justify-content: space-between;
    align-items: center;
    --border-radius: 11px;
    --border-width: 2px;
    cursor: pointer;

    .checked-item {
      padding: 0.5625em 1.98em 0.5625em;
      display: flex;
      align-items: center;
      .select-item-icon {
        margin-right: 0.9em;
      }
      span {
        color: var(--text-color);
        font-size: 1.17em;
        line-height: 200%;
        margin-top: -2px;
      }
    }
  }
  .select-drop-panel {
    --height: 16em;
    --border-width: 2px;
    --border-radius: 11px;
    width: 100%;
    height: calc(var(--height) + 4px);
    position: absolute;
    left: 0;
    top: 3.375em;
    cursor: pointer;
    background-color: var(--bg-white);
    border-radius: 13px;
    z-index: 99;
    overflow: hidden;
    box-shadow: 8px 11px 30px 0px var(--wg-shadow-color);
    .select-drop-list {
      width: calc(100% - 4px);
      height: var(--height);
      position: absolute;
      left: 2px;
      top: 2px;
      z-index: 1;
      background-color: var(--bg-white);
      border-radius: 11px;
      overflow: auto;
      &::-webkit-scrollbar-thumb {
        background-color: #fff;
      }
      &:hover {
        &::-webkit-scrollbar-thumb {
          background-color: #e4e7ff;
        }
      }
    }
    .select-drop-item {
      border-bottom: 2px solid #2e65ff;

      .drop-item-one {
        padding: 0.5625em 1.98em 0.5625em;
        display: flex;
        align-items: center;
        &:hover,
        &.active {
          background-color: var(--card-bg-hover-color);
        }
      }

      .drop-child-item {
        padding: 0.5625em 1.98em 0.5625em;
        padding-left: 4.3em;
        display: flex;
        align-items: center;
        border-top: 2px solid #2e65ff;
        &:hover,
        &.active {
          background-color: var(--card-bg-hover-color);
        }
      }

      &:last-child {
        border-bottom: none;
      }
      .select-item-icon {
        margin-right: 0.9em;
      }
      span {
        color: var(--text-color);
        font-size: 1.17em;
        line-height: 200%;
        margin-top: -2px;
      }
    }
  }
  .down-arrow {
    width: 20px;
    height: 20px;
    background: url(@/assets/images/down-arrow.svg) no-repeat;
    background-size: contain;
    position: absolute;
    right: 20px;

    &.active {
      transform: rotate(180deg);
    }
  }
}
</style>
