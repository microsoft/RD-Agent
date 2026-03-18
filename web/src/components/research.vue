<template>
  <div class="research-component">
    <div class="content-box">
      <h2>
        Hypothesis
        <img
          v-if="isWaitingForHypothesis"
          src="@/assets/playground-images/loading-tab.gif"
          alt="loading"
        />
      </h2>
      <div class="deduction">
        <div
          class="deduction-content"
          :class="{ 'deduction-content--pdf-only': isPdfOnlyHypothesis }"
          :style="{
            height: developer ? 'calc(100vh - 20.3em)' : 'calc(100vh - 17.9em)',
          }"
        >
          <div
            class="pdf-content"
            :class="{ 'pdf-content--full': isPdfOnlyHypothesis }"
            v-if="researchPdfImage"
          >
            <img
              :src="researchPdfImage"
              alt="pdf image"
              :class="{ 'pdf-image--full': isPdfOnlyHypothesis }"
            />
            <div class="pdf-full" @click="zoom">
              <span class="fullscreen"></span> Full Screen
            </div>
          </div>
          <div v-if="researchHypothesis">
            <h3>Hypothesis</h3>
            <div>
              <p v-if="researchHypothesis.hypothesis">
                {{ researchHypothesis.hypothesis }}
              </p>
              <p v-else>
                {{ researchHypothesis.name_map["no_hypothesis"] }}
              </p>
            </div>
            <h3>Component</h3>
            <p>{{ researchHypothesis.component }}</p>
            <h3>Reason</h3>
            <div>
              <p v-if="researchHypothesis.reason">
                {{ researchHypothesis.reason }}
              </p>
            </div>
          </div>
          <div v-if="!isWaitingForHypothesis && !researchHypothesis && !researchPdfImage">
            <p style="padding-left: 1em">
              No hypothesis generated due to some errors happened in previous
              steps.
            </p>
          </div>
        </div>
      </div>
    </div>
    <div class="content-box">
      <h2>
        Tasks<img
          v-if="!researcTasks && !updateEnd"
          src="@/assets/playground-images/loading-tab.gif"
          alt="loading"
        />
      </h2>
      <div v-if="researcTasks">
        <selectComponent
          :scenarioList="researcTasks"
          :scenarioIndex="scenarioCheckedIndex"
          :showStatus="false"
          @scenarioCheckedItem="scenarioCheckedItem"
        ></selectComponent>
        <div class="deduction" style="margin-top: 1em">
          <div
            class="deduction-content modelTask"
            v-if="scenarioChecked"
            :style="{
              height: developer ? 'calc(100vh - 24em)' : 'calc(100vh - 21.5em)',
            }"
          >
            <div
              v-for="field in taskFields"
              :key="field.key"
              class="task-field"
            >
              <h3>{{ field.label }}</h3>
              <div v-if="field.key === 'description' || field.key === 'formulation'">
                <markdown :content="toTaskFieldMarkdownContent(field)"></markdown>
              </div>
              <div v-else-if="field.key === 'variables'" class="task-table-wrap">
                <table class="task-table">
                  <thead>
                    <tr>
                      <th>Variable</th>
                      <th>Value</th>
                    </tr>
                  </thead>
                  <tbody>
                    <tr
                      v-for="row in toVariablesRows(field.value)"
                      :key="`${field.key}-${row.name}`"
                    >
                      <td><markdown :content="toVariableNameCellContent(row.name)"></markdown></td>
                      <td><markdown :content="toVariableValueCellContent(row.value)"></markdown></td>
                    </tr>
                  </tbody>
                </table>
              </div>
              <p v-else class="task-field-text">
                {{ toDisplayText(field.value) }}
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
    <div class="dialog-box" v-if="showDialog">
      <div class="dialog-content gradient-border">
        <div class="close" @click="close"></div>
        <div class="dialog-pdf-box">
          <img :src="researchPdfImage" alt="pdf image" />
        </div>
      </div>
    </div>
  </div>
</template>
<script setup>
import { ref, watch, onMounted, computed, defineProps } from "vue";
import selectComponent from "../components/sm-select-component.vue";
import markdown from "../components/markdown.vue";
const props = defineProps({
  currentData: Object,
  updateEnd: Boolean,
  developer: Boolean,
});
const currentData = ref(props.currentData);
const updateEnd = ref(props.updateEnd);
const developer = ref(props.developer);
const researchHypothesis = ref(null);
const researcTasks = ref(null);
const researchPdfImage = ref("");
const scenarioChecked = ref(null);
const scenarioCheckedIndex = ref(0);
const showDialog = ref(false);

const isPdfOnlyHypothesis = computed(() => {
  return !developer.value && Boolean(researchPdfImage.value) && !researchHypothesis.value;
});

const isWaitingForHypothesis = computed(() => {
  return !updateEnd.value && !researchHypothesis.value && !researchPdfImage.value;
});

const isEmptyTaskField = (value) => {
  if (value == null) {
    return true;
  }

  if (typeof value === "string") {
    return value.trim() === "";
  }

  if (Array.isArray(value)) {
    return value.length === 0;
  }

  if (typeof value === "object") {
    return Object.keys(value).length === 0;
  }

  return false;
};

const toFieldLabel = (key) => {
  return key
    .split("_")
    .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
    .join(" ");
};

const toMarkdownContent = (value) => {
  if (Array.isArray(value)) {
    return value.join("\n\n");
  }

  if (typeof value === "string") {
    return value;
  }

  return JSON.stringify(value, null, 2);
};

const hasLatexDelimiters = (content) => {
  if (!content || typeof content !== "string") {
    return false;
  }

  const trimmedContent = content.trim();

  return (
    trimmedContent.includes("$$") ||
    trimmedContent.includes("\\(") ||
    trimmedContent.includes("\\[") ||
    trimmedContent.includes("\\begin{")
  );
};

const looksLikeLatexFormula = (content, { includeOperators = true } = {}) => {
  if (!content || typeof content !== "string") {
    return false;
  }

  const trimmedContent = content.trim();

  if (trimmedContent === "") {
    return false;
  }

  const hasLatexCommand = /\\[a-zA-Z]+/.test(trimmedContent);
  const hasSubOrSuperscript = /[A-Za-z][A-Za-z0-9]*(?:[_^](?:\{[^{}]+\}|[A-Za-z0-9]+))/.test(
    trimmedContent
  );
  const hasLatexGrouping = /[_^{}]/.test(trimmedContent);
  const hasMathOperators = /[=<>+\-*/]/.test(trimmedContent);

  return (
    hasLatexCommand ||
    hasSubOrSuperscript ||
    hasLatexGrouping ||
    (includeOperators && hasMathOperators)
  );
};

const wrapStandaloneLatexContent = (
  value,
  { displayMode = false, allowSentencePunctuation = true, includeOperators = true } = {}
) => {
  const content = toMarkdownContent(value);

  if (typeof content !== "string") {
    return content;
  }

  const trimmedContent = content.trim();

  if (
    !trimmedContent ||
    hasLatexDelimiters(trimmedContent) ||
    !looksLikeLatexFormula(trimmedContent, { includeOperators })
  ) {
    return content;
  }

  const hasPlainSentencePunctuation = /[.!?]|:\s+[A-Za-z]/.test(trimmedContent);

  if (!allowSentencePunctuation && hasPlainSentencePunctuation) {
    return content;
  }

  if (displayMode) {
    return `$$\n${trimmedContent}\n$$`;
  }

  return `$${trimmedContent}$`;
};

const formatFormulationContent = (value) => {
  return wrapStandaloneLatexContent(value, {
    displayMode: true,
    allowSentencePunctuation: true,
    includeOperators: true,
  });
};

const toTaskFieldMarkdownContent = (field) => {
  if (field.key === "formulation") {
    return formatFormulationContent(field.value);
  }

  return toMarkdownContent(field.value);
};

const toDisplayText = (value) => {
  if (Array.isArray(value)) {
    return value.join(", ");
  }

  if (typeof value === "object" && value !== null) {
    return JSON.stringify(value, null, 2);
  }

  return String(value);
};

const wrapBareInlineLatex = (content) => {
  if (!content || typeof content !== "string") {
    return content;
  }

  const protectedSegments = [];
  const protectedContent = content.replace(
    /(\$\$[\s\S]+?\$\$|\$[^$\n]+\$|\\\([\s\S]+?\\\)|\\\[[\s\S]+?\\\])/g,
    (match) => {
      const marker = `@@LATEX_${protectedSegments.length}@@`;
      protectedSegments.push(match);
      return marker;
    }
  );

  const withCommandLatex = protectedContent.replace(
    /\\[a-zA-Z]+(?:\s*\[[^\]]*\]|\s*\{[^{}]*\}|[_^](?:\{[^{}]*\}|[A-Za-z0-9]))*/g,
    (match) => `$${match}$`
  );

  const withInlineLatex = withCommandLatex.replace(
    /\b[A-Za-z][A-Za-z0-9]*(?:[_^](?:\{[^{}]+\}|[A-Za-z0-9]+))+/g,
    (match) => `$${match}$`
  );

  return withInlineLatex.replace(/@@LATEX_(\d+)@@/g, (_, index) => {
    return protectedSegments[Number(index)];
  });
};

const toVariableCellContent = (value) => {
  return wrapStandaloneLatexContent(value, {
    displayMode: false,
    allowSentencePunctuation: false,
    includeOperators: false,
  });
};

const toVariableNameCellContent = (value) => {
  return toVariableCellContent(value);
};

const toVariableValueCellContent = (value) => {
  return wrapBareInlineLatex(toVariableCellContent(value));
};

const toVariablesRows = (variables) => {
  if (Array.isArray(variables)) {
    return variables.map((item, index) => {
      if (item && typeof item === "object" && !Array.isArray(item)) {
        return {
          name: item.name || item.key || `item_${index + 1}`,
          value: toMarkdownContent(item.value ?? item.expression ?? item),
        };
      }

      return {
        name: `item_${index + 1}`,
        value: toMarkdownContent(item),
      };
    });
  }

  if (variables && typeof variables === "object") {
    return Object.entries(variables).map(([name, value]) => ({
      name,
      value: toMarkdownContent(value),
    }));
  }

  return [];
};

const taskFields = computed(() => {
  if (!scenarioChecked.value) {
    return [];
  }

  return Object.entries(scenarioChecked.value)
    .filter(([key, value]) => key !== "name" && !isEmptyTaskField(value))
    .map(([key, value]) => ({
      key,
      label: toFieldLabel(key),
      value,
    }))
    .sort((left, right) => {
      if (left.key === "model_type") {
        return -1;
      }
      if (right.key === "model_type") {
        return 1;
      }

      return 0;
    });
});

const setScenarioChecked = (task) => {
  scenarioChecked.value = task;
};

const updateData = () => {
  if (currentData.value) {
    researchHypothesis.value = currentData.value.researchHypothesis;
    researcTasks.value = currentData.value.researcTasks;
    researchPdfImage.value = currentData.value.researchPdfImage;
    scenarioCheckedIndex.value = 0;
    if (researcTasks.value) {
      setScenarioChecked(researcTasks.value[scenarioCheckedIndex.value]);
    }
  }
};

const zoom = (color, data, name) => {
  showDialog.value = true;
};
const close = () => {
  showDialog.value = false;
};

watch(
  () => [props.currentData, props.updateEnd, props.developer],
  (newValue, oldValue) => {
    currentData.value = newValue[0];
    updateEnd.value = newValue[1];
    developer.value = newValue[2];
    updateData();
  },
  {
    deep: true,
    immediate: true,
  }
);

const scenarioCheckedItem = (data) => {
  scenarioCheckedIndex.value = data.scenarioCheckedIndex;
  setScenarioChecked(data.scenarioChecked);
};
onMounted(() => {
  if (currentData.value) {
    updateData();
  }
});
</script>

<style scoped lang="scss">
.research-component {
  height: 100%;
  display: flex;
  gap: 1.89em;
  .content-box {
    width: 50%;
    height: 100%;
    color: var(--text-color);
    h2 {
      font-size: 1.26em;
      font-weight: 700;
      line-height: 200%;
      margin-bottom: 0.45em;
      position: relative;

      img {
        width: 2.25em;
        height: 2.25em;
        margin-left: 0.405em;
        position: absolute;
        top: -0.18em;
      }
    }
    .deduction {
      border-radius: 11px;
      background: var(--bg-white);
      padding: 0.9em 0;
      box-sizing: border-box;
      overflow-y: hidden;
      .deduction-content {
        height: calc(100vh - 19.8em);
        padding: 0 1.6875em;
        overflow: auto;
        &::-webkit-scrollbar-thumb {
          background-color: #fff;
        }
        &:hover {
          &::-webkit-scrollbar-thumb {
            background-color: #e4e7ff;
          }
        }

        .pdf-content {
          text-align: center;
          &.pdf-content--full {
            height: 100%;
            display: flex;
            flex-direction: column;
            justify-content: space-between;
          }
          img {
            height: 18em;
            &.pdf-image--full {
              width: 100%;
              height: calc(100% - 2.4em);
              object-fit: contain;
              object-position: center top;
            }
          }
          .pdf-full {
            font-weight: 700;
            font-size: 0.9em;
            line-height: 1.8em;
            color: var(--text-color);
            display: flex;
            justify-content: center;
            align-items: center;
            cursor: pointer;
            max-width: 7.5em;
            margin: 0 auto;

            .fullscreen {
              display: inline-block;
              width: 1.125em;
              height: 1.125em;
              background: url(@/assets/playground-images/fullscreen.svg)
                no-repeat;
              background-size: contain;
              margin-right: 0.45em;
            }
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
        &.deduction-content--pdf-only {
          display: flex;
          flex-direction: column;
        }
      }
      .modelTask {
        height: calc(100vh - 23.4em);
        .task-field {
          margin-top: 1em;
          &:first-child {
            margin-top: 0;
          }
        }
        .task-field-text {
          white-space: pre-wrap;
          word-break: break-word;
        }
        .task-table-wrap {
          overflow-x: auto;
        }
        .task-table {
          width: 100%;
          border-collapse: collapse;
          font-size: 0.9em;
          line-height: 180%;
          th,
          td {
            padding: 0.65em 0.8em;
            border: 1px solid #d9e2f2;
            text-align: left;
            vertical-align: top;
          }
          th {
            background: #f5f8ff;
            font-weight: 700;
          }
        }
      }
    }
  }

  .dialog-box {
    width: 100vw;
    height: 100vh;
    position: fixed;
    left: 0;
    top: 0;
    background: rgba(255, 255, 255, 0.29);
    backdrop-filter: blur(4.599999904632568px);
    z-index: 999999;
    display: flex;
    align-items: center;
    justify-content: center;
    .dialog-content {
      width: 800px;
      height: 80vh;
      background-color: #fff;
      border-radius: 18px;
      --border-radius: 20px;
      --border-width: 2px;
      padding: 2em 0;
      margin-top: -4em;
      position: relative;
      box-sizing: border-box;

      .dialog-pdf-box {
        width: 100%;
        height: 100%;
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
      img {
        display: block;
        max-width: 100%;
        margin: 0 auto;
      }
      .close {
        position: absolute;
        right: 1.5em;
        top: 1em;
        width: 1.125em;
        height: 1.125em;
        background: url(@/assets/playground-images/close.svg) no-repeat;
        background-size: contain;
        cursor: pointer;
        z-index: 1;
        &:hover {
          opacity: 0.5;
        }
      }
    }
  }
}
:deep(.el-table) {
  font-size: 0.9em;
}
:deep(.el-table thead) {
  color: var(--text-color);
}
</style>
