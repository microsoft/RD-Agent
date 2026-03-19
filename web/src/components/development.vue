<template>
  <div class="research-component" v-if="evolvingCodes.length != 0">
    <div
      class="content-box sm-7-size"
      :style="{ width: fullScreenFlag ? '100%' : 'calc(70% - 1.89em)' }"
    >
      <div v-show="!fullScreenFlag">
        <h2>
          Evolving process
          <img
            v-if="allData && allData.length == 0 && !updateEnd"
            src="@/assets/playground-images/loading-tab.gif"
            alt="loading"
          />
        </h2>
        <div class="process">
          <span
            class="down-arrow"
            :class="{ rotate: showProcessFlag }"
            @click="showAllProcess"
          ></span>
          <div class="process-content">
            <ul ref="panel">
              <li
                v-for="(item, index) in allData"
                :key="'p_' + index"
                :class="{ active: currentLoop == index }"
                @click="updateLoop(index, 0)"
              >
                <span style="margin-right: 1.5em"
                  >Round
                  {{ String(allData.length - index).padStart(2, "0") }}</span
                >
                <span
                  v-for="(child, n) in item"
                  :key="'c_' + n"
                  class="span"
                  :class="{
                    active: currentLoop == index && scenarioCheckedIndex == n,
                  }"
                  @click.stop="updateLoop(index, n)"
                >
                  <el-tooltip
                    effect="dark"
                    popper-class="process-popper"
                    :content="child.name"
                    placement="bottom"
                  >
                    <span
                      :class="{
                        success: child.decision,
                        fail: !child.decision,
                        checked:
                          currentLoop == index && scenarioCheckedIndex == n,
                      }"
                    ></span>
                  </el-tooltip>
                </span>
              </li>
            </ul>
          </div>
        </div>
      </div>
      <div v-if="scenarioChecked" style="width: 100%">
        <h2 style="margin: 1em 0 0; font-size: 1.125em">Implementation</h2>
        <div>
          <el-tabs
            v-model="activeName"
            class="demo-tabs"
            @tab-click="handleClick"
          >
            <el-tab-pane
              v-for="item in codeNavList"
              :key="item"
              :label="item"
              :name="item"
            >
              <codeComponent
                :markdown="scenarioChecked.workspace[activeName]"
                :developer="developer"
                :fullscreen="fullScreenFlag"
                @fullScreen="fullScreen"
              ></codeComponent>
            </el-tab-pane>
          </el-tabs>
        </div>
      </div>
    </div>
    <div class="content-box sm-3-size" v-show="!fullScreenFlag">
      <h2 style="font-size: 1.25em; margin-bottom: 0.8em">Tasks</h2>
      <el-tooltip
        effect="dark"
        raw-content
        :content="
          `<div style='width: 500px;font-size: 14px;padding: 0.5em 0.5em 0.7em;line-height:160%; '>` +
          modelTaskDesc +
          '</div>'
        "
        placement="left"
      >
        <selectComponent
          :scenarioList="currentLoopData"
          :scenarioIndex="scenarioCheckedIndex"
          :showStatus="true"
          @scenarioCheckedItem="scenarioCheckedItem"
        ></selectComponent>
      </el-tooltip>
      <h2 style="margin: 1.2em 0 0; font-size: 1.125em">Feedback</h2>
      <div
        class="code-nav"
        :style="{
          width: developer
            ? 'calc(calc(100vw - 19.35em) * 0.3)'
            : 'calc(calc(100vw - 5.49em) * 0.3)',
        }"
      >
        <el-tabs
          v-model="feedbackName"
          class="demo-tabs"
          @tab-click="handleClick"
        >
          <el-tab-pane
            v-for="item in feedbackList"
            :key="item.abridgeName"
            :label="item.abridgeName"
            :name="item.abridgeName"
          >
            <div class="deduction" v-if="item.content">
              <div
                class="deduction-content"
                :style="{
                  height: developer
                    ? 'calc(100vh - 28.58em)'
                    : 'calc(100vh - 27.5em)',
                }"
              >
                <p>
                  {{ item.content }}
                </p>
              </div>
            </div>
          </el-tab-pane>
        </el-tabs>
      </div>
    </div>
  </div>
  <div class="research-component" v-else-if="updateEnd">
    <p>No code generated due to some errors happened in previous steps.</p>
  </div>
</template>
<script setup>
import { ref, watch, onMounted, computed, defineProps, nextTick } from "vue";
import selectComponent from "../components/sm-select-component.vue";
import codeComponent from "../components/code.vue";
import { marked } from "marked"; // 用于解析Markdown
import hljs from "highlight.js"; // 用于代码高亮
import "highlight.js/styles/1c-light.css"; // 引入你想要的代码高亮样式

const props = defineProps({
  evolvingCodes: Array,
  evolvingFeedbacks: Array,
  updateEnd: Boolean,
  developer: Boolean,
  currentData: Object,
});

const fullScreenFlag = ref(false);
const evolvingCodes = ref(props.evolvingCodes);
const evolvingFeedbacks = ref(props.evolvingFeedbacks);
const updateEnd = ref(props.updateEnd);
const developer = ref(props.developer);
const currentData = ref(props.currentData);
const allData = ref(null);
const currentLoop = ref(0);
const currentLoopData = ref([]);
const modelTaskDescObj = ref(null);
const modelTaskDesc = ref("");
const modelTask = ref(null);
const scenarioChecked = ref(null);
const scenarioCheckedIndex = ref(0);
const feedbackList = ref(null);
const codeNavList = ref(["ad_data.py"]);
const activeName = ref("ad_data.py");

const feedbackName = ref("");

const handleClick = (tab, event) => {
  console.log(tab, event);
};

const getLoopdata = (codes, feedbacks) => {
  const data = [];
  const tempObj = {};
  for (let i = 0; i < codes.length; i++) {
    for (let j = 0; j < codes[i].content.length; j++) {
      if (tempObj[codes[i].content[j].evo_id]) {
        tempObj[codes[i].content[j].evo_id].push({
          name: codes[i].content[j].target_task_name,
          workspace: codes[i].content[j].workspace,
          decision: feedbacks[i].content[j].final_decision,
          feedback: [
            {
              name: "Execution Feedback🖥️",
              abridgeName: "Execution",
              content: feedbacks[i].content[j].execution,
            },
            {
              name: "Code Feedback📄",
              abridgeName: "Code",
              content: feedbacks[i].content[j].code,
            },
            {
              name: "Return Checking",
              abridgeName: "Return Checking",
              content: feedbacks[i].content[j].return_checking,
            },
          ],
        });
      } else {
        tempObj[codes[i].content[j].evo_id] = [
          {
            name: codes[i].content[j].target_task_name,
            workspace: codes[i].content[j].workspace,
            decision: feedbacks[i].content[j].final_decision,
            feedback: [
              {
                name: "Execution Feedback🖥️",
                abridgeName: "Execution",
                content: feedbacks[i].content[j].execution,
              },
              {
                name: "Code Feedback📄",
                abridgeName: "Code",
                content: feedbacks[i].content[j].code,
              },
              {
                name: "Return Checking",
                abridgeName: "Return Checking",
                content: feedbacks[i].content[j].return_checking,
              },
            ],
          },
        ];
      }
    }
  }
  let sortedKeys = Object.keys(tempObj).sort((a, b) => b - a);

  // 根据倒序的键名获取值
  let result = sortedKeys.map((key) => tempObj[key]);
  return result;
};

const updatData = () => {
  if (evolvingCodes.value.length > 0 && evolvingFeedbacks.value.length > 0) {
    allData.value = getLoopdata(evolvingCodes.value, evolvingFeedbacks.value);
    modelTaskDescObj.value = currentData.value.researcTasks.reduce(
      (acc, currentValue, index) => {
        acc[currentValue.name] = currentValue.description;
        return acc;
      },
      {}
    );
    currentLoop.value = 0;
    currentLoopData.value = allData.value[currentLoop.value];
    scenarioCheckedIndex.value = 0;
    scenarioChecked.value = currentLoopData.value[scenarioCheckedIndex.value];
    modelTaskDesc.value = modelTaskDescObj.value[scenarioChecked.value.name];
    codeNavList.value = Object.keys(scenarioChecked.value.workspace);
    activeName.value = codeNavList.value[0];
    feedbackList.value = scenarioChecked.value.feedback.filter((item) => {
      return item.content;
    });
    feedbackName.value = feedbackList.value[0].abridgeName;
  }
};

const updateLoop = (index, n) => {
  currentLoop.value = index;
  currentLoopData.value = allData.value[currentLoop.value];
  scenarioCheckedIndex.value = n;
  scenarioChecked.value = currentLoopData.value[scenarioCheckedIndex.value];
  codeNavList.value = Object.keys(scenarioChecked.value.workspace);
  activeName.value = codeNavList.value[0];
  feedbackList.value = scenarioChecked.value.feedback.filter((item) => {
    return item.content;
  });
  feedbackName.value = feedbackList.value[0].abridgeName;
};

const scenarioCheckedItem = (data) => {
  scenarioCheckedIndex.value = data.scenarioCheckedIndex;
  scenarioChecked.value = data.scenarioChecked;
  codeNavList.value = Object.keys(scenarioChecked.value.workspace);
  activeName.value = codeNavList.value[0];
  modelTaskDesc.value = modelTaskDescObj.value[scenarioChecked.value.name];
  feedbackList.value = scenarioChecked.value.feedback.filter((item) => {
    return item.content;
  });
  feedbackName.value = feedbackList.value[0].abridgeName;
};

watch(
  () => [props.currentData, props.updateEnd, props.developer],
  (newValue, oldValue) => {
    currentData.value = newValue[0];
    evolvingCodes.value = currentData.value.evolvingCodes;
    evolvingFeedbacks.value = currentData.value.evolvingFeedbacks;
    updateEnd.value = newValue[1];
    developer.value = newValue[2];

    updatData();
  },
  {
    deep: true,
    immediate: true,
  }
);

const panel = ref(null);
const showProcessFlag = ref(false);
const showAllProcess = () => {
  showProcessFlag.value = !showProcessFlag.value;
  if (!showProcessFlag.value) {
    panel.value.style.maxHeight = "2.925em";
  } else {
    panel.value.style.maxHeight = panel.value.scrollHeight + "px";
  }
};

const fullScreen = (flag) => {
  fullScreenFlag.value = flag;
};

onMounted(() => {
  updatData();
});
</script>

<style scoped lang="scss">
.research-component {
  width: 100%;
  height: 100%;
  display: flex;
  gap: 1.89em;
  .content-box {
    // width: 50%;
    height: 100%;
    color: var(--text-color);

    &.sm-7-size {
      width: calc(70% - 1.89em);
    }
    &.sm-3-size {
      width: 30%;
    }
    h2 {
      font-size: 1.26em;
      font-weight: 700;
      line-height: 200%;
      margin-bottom: 0.45em;
      // display: flex;
      // align-items: center;
      // justify-content: flex-start;
      padding-right: 0.18em;
      position: relative;
      img {
        width: 2.25em;
        height: 2.25em;
        margin-left: 0.45em;
        position: absolute;
        top: -0.18em;
      }
    }

    .process {
      background: var(--bg-white);
      border-radius: 11px;
      position: relative;
      .down-arrow {
        width: 0.9em;
        height: 0.9em;
        background: url(@/assets/images/down-arrow.svg) no-repeat;
        background-size: contain;
        cursor: pointer;
        transition: transform 0.3s ease-out;
        position: absolute;
        right: 0.9em;
        top: 1.035em;

        &.rotate {
          transform: rotate(-180deg);
          transition: transform 0.3s ease-out;
        }
      }
      .process-content {
        ul {
          overflow: hidden;
          max-height: 2.925em;
          transition: max-height 0.3s ease-out;
          li {
            display: flex;
            align-items: center;
            justify-content: flex-start;
            padding: 0.72em 1.9125em;
            height: 2.925em;
            box-sizing: border-box;
            margin-bottom: 0.1em;
            gap: 1.08em;

            &:last-child {
              margin: 0;
            }

            &:hover,
            &.active {
              border-radius: 11px;
              background: var(--card-bg-hover-color);
            }
            span {
              display: inline-block;
            }
            .span {
              padding: 0.18em 0.225em;
              border-radius: 4px;
              &:hover,
              &.active {
                background: rgba(178, 159, 255, 0.4);
              }
            }
            .success {
              width: 1.125em;
              height: 1.125em;
              background: url(@/assets/playground-images/process-success.svg)
                no-repeat;
              background-size: contain;
              vertical-align: middle;

              &.checked {
                width: 1.125em;
                height: 1.125em;
                background: url(@/assets/playground-images/process-checked.svg)
                  no-repeat;
                background-size: contain;
                vertical-align: middle;
              }
            }
            .fail {
              width: 1.125em;
              height: 1.125em;
              background: url(@/assets/playground-images/process-fail.svg)
                no-repeat;
              background-size: contain;
              vertical-align: middle;

              &.checked {
                width: 1.125em;
                height: 1.125em;
                background: url(@/assets/playground-images/process-fail-checked.svg)
                  no-repeat;
                background-size: contain;
                vertical-align: middle;
              }
            }
          }
        }
      }
    }
    .deduction {
      border-radius: 11px;
      background: var(--bg-white);
      box-sizing: border-box;
      overflow-y: hidden;
      .deduction-content {
        height: calc(100vh - 26.955em);
        padding: 1.35em 1.6875em 0.9em;
        box-sizing: border-box;
        overflow-y: auto;
        &::-webkit-scrollbar-thumb {
          background-color: #fff;
        }
        &:hover {
          &::-webkit-scrollbar-thumb {
            background-color: #e4e7ff;
          }
        }
        h3 {
          font-size: 1.1475em;
          font-weight: 700;
          line-height: 200%;
          margin-bottom: 0.45em;
          margin-top: 0.9em;
          &:first-child {
            margin-top: 0;
          }
        }
        p {
          font-family: "Microsoft YaHei";
          font-size: 0.9em;
          line-height: 180%;
        }
      }
    }
  }
}
:deep(.el-tabs__item.is-active),
:deep(.el-tabs__item:hover) {
  background: linear-gradient(90deg, #2667ff 0%, #9d41ff 100%);
  background-clip: text;
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  // font-weight: 700;
}
:deep(.el-tabs__active-bar) {
  background: linear-gradient(to right, #2667ff, #9d41ff);
}
:deep(.el-tabs__item) {
  padding: 0 12px;
  color: #ababab;
}
:deep(.code-nav .el-tabs__nav) {
  width: 100%;
}
:deep(.code-nav .el-tabs__item) {
  min-width: calc(100% / 5);
  max-width: calc(100% / 3);
  width: calc(100% / 3);
}
</style>
