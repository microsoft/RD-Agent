<template>
  <div class="loop-box">
    <div class="loop-box-header">
      <div
        class="trace-name-text"
        v-if="traceName"
        :title="`Trace name: ${traceName}`"
      >
        <span class="trace-name-value">{{ traceName }}</span>
      </div>
      <span class="loop-title">Loops</span>
    </div>
    <div class="loop-box-list" ref="loops">
      <div class="loop-length">
        <div class="loop-item" v-for="index in loopNumber" :key="index">
          <div
            class="loop-item-content"
            @click="clickLoop(index, true)"
            v-if="!isCompleted(index) && loadingIndex == index"
          >
            <div class="loop-item-icon">
              <img
                src="@/assets/playground-images/loop-loading.gif"
                alt="loading"
              />
            </div>
            <div
              class="loop-item-label"
              :class="{ active: currentIndex == index }"
            >
              <span>{{ index < 10 ? "0" + index : index }}</span> Loop
            </div>
          </div>
          <div
            class="loop-item-content"
            @click="clickLoop(index, false)"
            v-if="isCompleted(index)"
          >
            <div class="loop-item-icon">
              <img
                v-if="statusList[index - 1]"
                src="@/assets/playground-images/loop-Sucess.svg"
                alt="loading"
              />
              <img
                v-else
                src="@/assets/playground-images/loop-error.svg"
                alt="loading"
              />
            </div>
            <div
              class="loop-item-label"
              :class="{ active: currentIndex == index }"
            >
              <span>{{ index < 10 ? "0" + index : index }}</span> Loop
            </div>
          </div>
          <div
            class="loop-item-content"
            v-if="!isCompleted(index) && loadingIndex !== index"
          >
            <div class="loop-item-icon">
              <img
                src="@/assets/playground-images/loop-default.svg"
                alt="loading"
              />
            </div>
            <div
              class="loop-item-label"
              :class="{ active: currentIndex == index }"
            >
              <span>{{ index < 10 ? "0" + index : index }}</span> Loop
            </div>
          </div>
        </div>
        <div class="default-line"></div>
        <div class="line" :style="{ height: height + '%' }"></div>
      </div>
    </div>
    <div class="loop-box-btn">
      <button
        :class="{
          active: !isDone,
          disable: isDone,
        }"
        :disabled="stopFlag || isDone"
        @click="stopClick"
      >
        Stop
      </button>
      <div class="auto-skip-toggle" v-if="editLoop">
        <label class="toggle-label">
          <span class="toggle-text">Auto Skip Interaction</span>
          <span class="toggle-switch">
            <input type="checkbox" v-model="autoSkip" @change="emitAutoSkip" />
            <span class="toggle-slider"></span>
          </span>
        </label>
      </div>
    </div>
  </div>
</template>
<script setup>
import { ref, watch, onMounted, defineProps, defineEmits, nextTick } from "vue";
const props = defineProps({
  loadingIndex: Number,
  loopNumber: Number,
  editLoop: Boolean,
  currentData: Array,
  updateEnd: Boolean,
  traceName: String,
});
const loadingIndex = ref(props.loadingIndex);
const loopNumber = ref(props.loopNumber);
const editLoop = ref(props.editLoop);
const currentData = ref(props.currentData);
const traceName = ref(props.traceName);
const statusList = ref([]);
const emit = defineEmits(["addLoop", "clickIndex", "clickStop", "toggleAutoSkip"]);
const loops = ref(null);
const stopFlag = ref(false);
const isDone = ref(props.updateEnd);
const autoSkip = ref(false);

const currentIndex = ref(loadingIndex.value);

const isCompleted = (index) => {
  // return loadingIndex.value > index;
  return currentData.value.length >= index;
};

const scrollTo = () => {
  const el = loops.value;
  if (el) {
    if (loopNumber.value - loadingIndex.value < 3) {
      el.scrollTo({
        top: 4.66 * 16 * (loadingIndex.value + 6),
        behavior: "smooth",
      });
    } else {
      el.scrollTo({
        top: 4.66 * 16 * (loadingIndex.value - 6),
        behavior: "smooth",
      });
    }
  }
};
const height = ref(0);
const getHeight = () => {
  if (loadingIndex.value >= loopNumber.value) {
    height.value = 100;
  } else {
    height.value =
      ((2.975 + 4.075 * (loadingIndex.value - 1)) /
        (4.075 * loopNumber.value)) *
      100;
  }
};
const updateData = () => {
  statusList.value = currentData.value.map((item) => {
    return item.feedbackHypothesis ? item.feedbackHypothesis.decision : false;
  });
};

watch(
  () => [
    props.loadingIndex,
    props.loopNumber,
    props.currentData,
    props.updateEnd,
    props.traceName,
  ],
  (newValue, oldValue) => {
    loadingIndex.value = newValue[0];
    loopNumber.value = newValue[1];
    currentData.value = newValue[2];
    isDone.value = newValue[3];
    traceName.value = newValue[4];
    if (!isDone.value) {
      stopFlag.value = false;
    }
    updateData();
    currentIndex.value = loadingIndex.value;
    nextTick(() => {
      getHeight();
      scrollTo();
    });
  }
);

const addLoop = () => {
  emit("addLoop", true);
};
const stopClick = () => {
  if (isDone.value && !stopFlag.value) {
    return;
  }
  stopFlag.value = true;
  emit("clickStop", stopFlag.value);
};

const emitAutoSkip = () => {
  emit("toggleAutoSkip", autoSkip.value);
};

const clickLoop = (index, flag) => {
  currentIndex.value = index;
  emit("clickIndex", {
    index: index,
    loading: flag,
  });
};

onMounted(() => {
  getHeight();
});
</script>

<style scoped lang="scss">
.loop-box {
  width: 17.5em;
  width: 15.75em;
  height: 100%;
  min-height: 0;
  display: flex;
  flex-direction: column;
  box-sizing: border-box;
  padding-top: 0.5em;
  .loop-box-header {
    width: 100%;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.1em;
    padding: 0 0.9em;
    box-sizing: border-box;

    .loop-title {
      color: var(--text-color);
      font-size: 1.4em;
      font-size: 1.26em;
      font-weight: 700;
      line-height: 1.5;
      letter-spacing: 0.01em;
      text-align: center;
    }
  }

  .trace-name-text {
    width: fit-content;
    max-width: calc(100% - 1.8em);
    margin: 0;
    display: flex;
    align-items: center;
    justify-content: center;
    min-height: 0;

    .trace-name-value {
      font-size: 1.06875em;
      font-weight: 700;
      line-height: 200%;
      text-shadow: 8px 11px 30px var(--wg-shadow-color);
      background: linear-gradient(90deg, #2667ff 0%, #9d41ff 100%);
      background-clip: text;
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      word-break: break-word;
      overflow: hidden;
      text-overflow: ellipsis;
      white-space: nowrap;
      max-width: 100%;
      text-align: center;
    }
  }

  .loop-box-list {
    flex: 1;
    min-height: 0;
    height: auto;
    padding: 0 3em 0 4.3em;
    padding: 0 2.7em 0 3.87em;
    margin-top: 0.9em;
    overflow: auto;
    &::-webkit-scrollbar-thumb {
      background-color: #fff;
    }
    &:hover {
      &::-webkit-scrollbar-thumb {
        background-color: #e4e7ff;
      }
    }
    &.no-btn {
      height: calc(100vh - 17em);
      height: calc(100vh - 15.3em);
    }
    .loop-length {
      position: relative;
      .line {
        position: absolute;
        height: 0;
        width: 3px;
        background: linear-gradient(to bottom, #2667ff 0%, #9d41ff 100%);
        background: linear-gradient(to bottom, #ffffff 0%, #ffffff 100%);
        opacity: 0.3;
        opacity: 1;
        top: 0;
        left: 0.8125em;
        left: 0.73125em;
        transition: 0.75s ease;
      }
      .default-line {
        position: absolute;
        height: 100%;
        width: 3px;
        background-color: #c5d2e6;
        bottom: 0;
        left: 0.8125em;
        left: 0.73125em;
      }
    }
    .loop-item {
      padding: 1.1em 0;
      padding: 0.99em 0;
      position: relative;
      z-index: 1;
      .loop-item-content {
        display: flex;
        align-items: center;
        justify-content: flex-start;
        .loop-item-icon {
          display: flex;
          flex-direction: column;
          align-items: center;
          // .line {
          //   height: 2.625em;
          //   width: 4px;
          //   background-color: #c5d2e6;
          //   &.active {
          //     background: linear-gradient(to bottom, #2667ff 0%, #9d41ff 100%),
          //       #fefefe;
          //   }
          // }
          img {
            width: 1.875em;
            height: 1.875em;
            width: 1.6875em;
            height: 1.6875em;
          }
        }
        .loop-item-label {
          width: 4.5em;
          color: var(--text-color);
          font-size: 1.125em;
          font-size: 1.0125em;
          font-weight: 700;
          line-height: 160%;
          margin-left: 1.08em;
          padding: 0.3em 0.4em 0.3em 0.6em;
          padding: 0.27em 0.36em 0.27em 0.54em;
          cursor: pointer;

          &:hover,
          &.active {
            border-radius: 999px;
            background: var(--card-bg-hover-color);
          }
          span {
            margin-right: 5px;
          }
        }
      }
    }
  }
  .loop-box-btn {
    // padding-left: 2.5em;
    display: flex;
    flex-direction: column;
    align-items: center;
    flex-shrink: 0;
    position: relative;
    z-index: 2;
    padding: 0.9em 0 0.2em;
    background: #fff;
    button {
      width: 9.1em;
      height: 3em;
      padding: 0.56em 1.1em;
      padding: 0.504em 0.99em;
      border-radius: 24px;
      background: var(---bg-white);
      // box-shadow: 0px 0px 6px 0px rgba(0, 0, 0, 0.25);
      margin-top: 1.1em;
      margin-top: 0.99em;
      border: none;
      color: var(--text-color);
      text-align: center;
      font-family: "Microsoft YaHei";
      font-size: 1.125em;
      font-size: 1.0125em;
      text-transform: capitalize;
      cursor: pointer;

      &.disable {
        background: #d9d9d9;
        color: var(--text-white-color);
        pointer-events: none;
      }
      &.active-black {
        background: var(--text-color);
        // box-shadow: 0px 0px 6px 0px rgba(0, 0, 0, 0.25);
        color: var(--text-white-color);
        &:hover {
          box-shadow: 0px 0px 10px 0px rgba(142, 62, 255, 0.74);
        }
      }
      &.active {
        border-radius: 999px;
        background: linear-gradient(90deg, #2667ff 0%, #9d41ff 100%), #fefefe;
        box-shadow: 0px 0px 6px 0px rgba(0, 0, 0, 0.25);
        color: var(--text-white-color);
        &:hover {
          background: linear-gradient(
              0deg,
              rgba(255, 255, 255, 0.1) 0%,
              rgba(255, 255, 255, 0.1) 100%
            ),
            linear-gradient(90deg, #2667ff 0%, #9d41ff 100%);
          box-shadow: 0px 0px 10px 0px rgba(142, 62, 255, 0.68);
        }
      }
    }
  }
  .loop-box-btn {
    display: flex;
    flex-direction: column;
    gap: 0.75em;
  }
  .auto-skip-toggle {
    display: flex;
    justify-content: center;
    padding: 0 1.2em;
  }
  .toggle-label {
    display: flex;
    align-items: center;
    gap: 0.8em;
    font-size: 1em;
    color: var(--text-color);
    cursor: pointer;
    user-select: none;
  }
  .toggle-text {
    font-weight: 600;
  }
  .toggle-switch {
    position: relative;
    display: inline-block;
    width: 2.8em;
    height: 1.6em;
  }
  .toggle-switch input {
    opacity: 0;
    width: 0;
    height: 0;
  }
  .toggle-slider {
    position: absolute;
    cursor: pointer;
    top: 0;
    left: 0;
    right: 0;
    bottom: 0;
    background-color: #d6dbe7;
    transition: 0.2s ease;
    border-radius: 999px;
    box-shadow: inset 0 0 0 2px #c5d2e6;
  }
  .toggle-slider:before {
    position: absolute;
    content: "";
    height: 1.2em;
    width: 1.2em;
    left: 0.2em;
    top: 0.2em;
    background-color: #fff;
    transition: 0.2s ease;
    border-radius: 50%;
    box-shadow: 0 2px 6px rgba(0, 0, 0, 0.15);
  }
  .toggle-switch input:checked + .toggle-slider {
    background: linear-gradient(90deg, #2667ff 0%, #9d41ff 100%);
    box-shadow: none;
  }
  .toggle-switch input:checked + .toggle-slider:before {
    transform: translateX(1.2em);
  }
}
</style>
