<template>
  <div class="playground-page-root">
    <div class="playground-page">
      <loopComponent
        v-if="developer"
        :loadingIndex="loadingIndex"
        :loopNumber="loopNumber"
        :currentData="allData"
        :editLoop="editLoop"
        :updateEnd="updateEnd"
        :traceName="traceName"
        @addLoop="addLoop"
        @clickIndex="clickIndex"
        @clickStop="clickStop"
        @toggleAutoSkip="handleAutoSkipToggle"
      ></loopComponent>
      <div class="main-content">
        <div class="tab-title" v-if="developer">
          <div class="tab-box">
            <div
              class="tab-item-btn"
              @click="tabIndex = 0"
              :class="{ active: tabIndex == 0 }"
            >
              <span>
                <SvgIcon
                  class="pg-tab-icon"
                  name="pg-process"
                  :color="tabIndex == 0 ? '#fff' : '#2B2B2B'"
                ></SvgIcon>
                PROCESS
              </span>
              <SvgIcon
                class="arrow-right-icon"
                name="right-arrow"
                :color="tabIndex == 0 ? '#fff' : '#2B2B2B'"
              ></SvgIcon>
            </div>
            <div
              class="tab-item-btn"
              v-if="allData.length != 0"
              @click="tabIndex = 1"
              :class="{ active: tabIndex == 1 }"
            >
              <span>
                <SvgIcon
                  class="pg-tab-icon"
                  name="pg-result"
                  :color="tabIndex == 1 ? '#fff' : '#2B2B2B'"
                ></SvgIcon>
                RESULT
              </span>
              <SvgIcon
                class="arrow-right-icon"
                name="right-arrow"
                :color="tabIndex == 1 ? '#fff' : '#2B2B2B'"
              ></SvgIcon>
            </div>
            <div class="tab-item-btn" v-if="allData.length == 0 && !stopFlag">
              <span>
                <SvgIcon
                  class="pg-tab-icon"
                  name="pg-result"
                  :color="tabIndex == 1 ? '#fff' : '#2B2B2B'"
                ></SvgIcon>
                RESULT
              </span>
              <img
                src="@/assets/playground-images/loading-tab.gif"
                alt="loading"
              />
            </div>
          </div>
        </div>
        <div style="width: 100%" v-show="tabIndex == 0">
          <div class="nav-content">
            <nav v-if="developer">
              <ul ref="tabs">
                <li
                  :class="{
                    'borderRadius-right': tabProcessIndex == 0,
                    'borderRadius-none': tabProcessIndex !== 0,
                  }"
                  style="width: 2em"
                ></li>
                <li
                  :class="{
                    active: tabProcessIndex == 0,
                    'borderRadius-right': tabProcessIndex == 1,
                    'borderRadius-none': tabProcessIndex == 2,
                  }"
                >
                  <div class="tab-bg">
                    <span @click="tabChange(0)">Research</span>
                  </div>
                </li>
                <li
                  :class="{
                    active: tabProcessIndex == 1,
                    'borderRadius-right': tabProcessIndex == 2,
                    'borderRadius-left': tabProcessIndex == 0,
                  }"
                >
                  <div class="tab-bg">
                    <span
                      :class="{
                        'tab-label--clickable':
                          updateEnd ||
                          (currentData &&
                            currentData.evolvingFeedbacks.length !== 0),
                      }"
                      @click="
                        (updateEnd ||
                          (currentData &&
                            currentData.evolvingFeedbacks.length !== 0)) &&
                          tabChange(1)
                      "
                      >Development</span
                    >
                    <img
                      v-if="
                        !updateEnd &&
                        (!currentData || currentData.evolvingFeedbacks.length === 0)
                      "
                      src="@/assets/playground-images/loading-tab.gif"
                      alt="loading"
                    />
                  </div>
                </li>
                <li
                  :class="{
                    active: tabProcessIndex == 2,
                    'borderRadius-left': tabProcessIndex == 1,
                    'borderRadius-none': tabProcessIndex == 0,
                  }"
                >
                  <div class="tab-bg">
                    <span
                      :class="{
                        'tab-label--clickable':
                          updateEnd ||
                          (currentData && currentData.feedbackHypothesis),
                      }"
                      @click="
                        (updateEnd ||
                          (currentData && currentData.feedbackHypothesis)) &&
                          tabChange(2)
                      "
                      >Feedback</span
                    >
                    <img
                      v-if="
                        !updateEnd &&
                        (!currentData || !currentData.feedbackHypothesis)
                      "
                      src="@/assets/playground-images/loading-tab.gif"
                      alt="loading"
                    />
                  </div>
                </li>
                <li
                  :class="{
                    'borderRadius-left': tabProcessIndex == 2,
                    'borderRadius-none': tabProcessIndex !== 2,
                  }"
                  style="width: 2em"
                ></li>
              </ul>
            </nav>
            <nav v-if="!developer" style="justify-content: center">
              <ul ref="tabs">
                <li
                  :class="{
                    'borderRadius-right': tabProcessIndex == 0,
                    'borderRadius-none': tabProcessIndex !== 0,
                  }"
                  style="width: 2em"
                ></li>
                <li
                  :class="{
                    active: tabProcessIndex == 0,
                    'borderRadius-right': tabProcessIndex == 1,
                  }"
                >
                  <div class="tab-bg">
                    <span @click="tabChange(0)">Research</span>
                  </div>
                </li>
                <li
                  :class="{
                    active: tabProcessIndex == 1,
                    'borderRadius-left': tabProcessIndex == 0,
                  }"
                >
                  <div class="tab-bg">
                    <span
                      :class="{
                        'tab-label--clickable':
                          updateEnd ||
                          (currentData &&
                            currentData.evolvingFeedbacks.length !== 0),
                      }"
                      @click="
                        (updateEnd ||
                          (currentData &&
                            currentData.evolvingFeedbacks.length !== 0)) &&
                          tabChange(1)
                      "
                      >Development</span
                    >
                    <img
                      v-if="
                        !updateEnd &&
                        (!currentData || currentData.evolvingFeedbacks.length === 0)
                      "
                      src="@/assets/playground-images/loading-tab.gif"
                      alt="loading"
                    />
                  </div>
                </li>
                <li
                  :class="{
                    'borderRadius-left': tabProcessIndex == 1,
                    'borderRadius-none': tabProcessIndex !== 1,
                  }"
                  style="width: 2em"
                ></li>
              </ul>
            </nav>
          </div>
          <div
            class="bg-content"
            :style="{
              height: developer ? 'calc(100vh - 14.5em)' : 'calc(100vh - 12em)',
            }"
          >
            <research
              v-show="tabProcessIndex == 0"
              :currentData="currentData"
              :developer="developer"
              :updateEnd="updateEnd"
            ></research>
            <development
              v-if="tabProcessIndex == 1"
              :currentData="currentData"
              :updateEnd="updateEnd"
              :developer="developer"
            ></development>
            <feedback
              v-if="tabProcessIndex == 2"
              :currentData="currentData"
              :updateEnd="updateEnd"
            ></feedback>
          </div>
        </div>
        <div v-show="tabIndex == 1">
          <resultComponent
            :currentData="allData"
            :scenarioName="scenarioName"
            :baseFactors="initialBaseFactors"
            :traceName="traceName"
          ></resultComponent>
        </div>
      </div>
    </div>
    <dialogComponent :showDialog="showDialogForLoop"></dialogComponent>
    <div class="dialog-box" v-if="userInteractionVisible && !userInteractionMinimized">
      <div
        class="dialog-content gradient-border user-interaction-dialog"
        :class="{ 'user-interaction-dialog--wide': isFeatureInteraction }"
      >
        <div class="dialog-header">
          <h1>User Interaction Required</h1>
          <button
            class="dialog-minimize"
            type="button"
            @click="minimizeUserInteraction"
          >
            Minimize
          </button>
        </div>
        <template v-if="userInteractionWaitingHypothesis && !updateEnd">
          <div class="interaction-waiting">
            <span class="interaction-waiting-spinner" aria-hidden="true"></span>
            <span>R&amp;D-Agent is generating hypothesis</span>
          </div>
          <div class="interaction-form read-only">
            <div
              class="interaction-row"
              v-for="(entry, index) in userInteractionLastFeedbackEntries"
              :key="entry.key + '-readonly-' + index"
            >
              <label class="interaction-key">{{ entry.key }}</label>
              <select
                v-if="entry.key === 'decision'"
                class="interaction-select"
                :value="entry.value"
                disabled
              >
                <option :value="true">true</option>
                <option :value="false">false</option>
              </select>
              <textarea
                v-else
                class="interaction-textarea"
                :value="entry.value"
                rows="8"
                readonly
              ></textarea>
            </div>
          </div>
        </template>
        <template v-else>
          <p v-if="isFeatureInteraction">
            Update base features, then submit to continue.
          </p>
          <p v-else-if="isUserInstructionInteraction">
            Please update the overall instruction, then submit to continue.
          </p>
          <p v-else-if="isFeedbackInteraction">
            You can edit the system-generated decision and reason, then submit to continue.
          </p>
          <p v-else>
            You can edit the system-generated hypothesis and reason, then submit to continue.
          </p>
          <div
            class="feature-validation-msg"
            v-if="isFeatureInteraction && (localFeatureError || featureValidationMsg)"
          >
            {{ localFeatureError || featureValidationMsg }}
          </div>
          <div class="interaction-form">
            <div v-if="isFeatureInteraction" class="feature-table">
              <div class="feature-layout">
                <div class="feature-pool-block" v-if="availableFeatureTags.length">
                  <div class="feature-pool-title">Base features (Alpha158)</div>
                  <div class="feature-pool">
                    <div class="feature-pool-tags">
                    <button
                      class="feature-tag"
                      type="button"
                      v-for="tag in availableFeatureTags"
                      :key="tag.name"
                      @mouseenter="showFeatureTooltip($event, tag.expression)"
                      @mousemove="moveFeatureTooltip($event)"
                      @mouseleave="hideFeatureTooltip"
                      @click="addFeatureFromPool(tag)"
                    >
                      {{ tag.name }}
                    </button>
                  </div>
                  </div>
                </div>
                <div class="feature-editor">
                  <div class="feature-sticky-head">
                    <div class="feature-editor-meta">
                      Configured features: {{ configuredFeatureCount }}
                    </div>
                    <div class="feature-header">
                      <span>Feature name</span>
                      <span>Feature expression</span>
                    </div>
                  </div>
                  <div
                    class="feature-row"
                    v-for="(row, index) in featureRows"
                    :key="`feature-${index}`"
                  >
                    <input
                      class="feature-input"
                      type="text"
                      v-model="row.name"
                      placeholder="name"
                    />
                    <input
                      class="feature-input feature-input--math"
                      type="text"
                      v-model="row.expression"
                      placeholder="expression"
                    />
                    <button
                      class="feature-remove"
                      type="button"
                      @click="removeFeatureRow(index)"
                      :disabled="featureRows.length === 1"
                      aria-label="Remove feature"
                    >
                      ×
                    </button>
                  </div>
                  <button
                    class="feature-add"
                    type="button"
                    @click="addFeatureRow"
                  >
                    + Add feature
                  </button>
                </div>
              </div>
            </div>
            <div
              class="interaction-row"
              v-for="(entry, index) in userInteractionEntries"
              :key="entry.key + '-' + index"
              :class="{ 'interaction-row--stack': entry.key === 'user_instruction' }"
            >
              <label
                class="interaction-key"
                v-if="entry.key !== 'user_instruction'"
              >
                {{ entry.key }}
              </label>
              <div
                v-else
                class="interaction-key interaction-key--highlight"
              >
                Your overall instruction
              </div>
              <select
                v-if="entry.key === 'decision'"
                class="interaction-select"
                v-model="entry.value"
              >
                <option :value="true">true</option>
                <option :value="false">false</option>
              </select>
              <textarea
                v-else
                class="interaction-textarea"
                v-model="entry.value"
                rows="8"
                :placeholder="
                  entry.key === 'user_instruction' ? 'Example: 请使用中文表示hypothesis' : ''
                "
              ></textarea>
            </div>
          </div>
          <div class="btn-box">
            <button
              class="gradient-border back"
              @click="submitOriginalUserInteraction"
              :disabled="userInteractionSubmitting"
            >
              SKIP
            </button>
            <button
              class="add-loops active"
              @click="submitUserInteractionForm"
              :disabled="userInteractionSubmitting"
            >
              SUBMIT
            </button>
          </div>
        </template>
      </div>
    </div>
    <div
      class="dialog-minimized"
      v-if="userInteractionVisible && userInteractionMinimized"
      @click="restoreUserInteraction"
    >
      <div class="dialog-minimized-content">
        <span class="dialog-waiting-spinner" aria-hidden="true"></span>
        <span>User interaction pending</span>
      </div>
    </div>
    <teleport to="body">
      <div
        v-if="featureTooltip.visible"
        class="feature-tag-floating-tooltip"
        :style="{
          left: `${featureTooltip.left}px`,
          top: `${featureTooltip.top}px`,
        }"
      >
        {{ featureTooltip.text }}
      </div>
    </teleport>
  </div>
</template>
<script setup>
import {
  computed,
  defineProps,
  onMounted,
  onUnmounted,
  nextTick,
  reactive,
  ref,
  watch,
} from "vue";
import $ from "jquery";
import { ElNotification } from "element-plus";
import { trace, control, url, submitUserInteraction } from "../utils/api";
import ALPHA158 from "../constants/qlib";
import loopComponent from "../components/loop-component.vue";
import dialogComponent from "../components/dialog.vue";
import research from "../components/research.vue";
import development from "../components/development.vue";
import feedback from "../components/feedback.vue";
import resultComponent from "./ResultPage.vue";

const props = defineProps({
  id: String,
  editLoop: Boolean,
  developer: Boolean,
  loopNumber: Number,
  scenarioName: String,
});

const completedTraceStorageKey = "completedTraceIdList";

const editLoop = ref(props.editLoop);
const developer = ref(props.developer);
const scenarioName = ref(props.scenarioName);
const stopFlag = ref(false);
let transitionTimer = undefined;
const tabIndex = ref(0);
const tabProcessIndex = ref(0);
const tabs = ref(null);
const showDialogForLoop = ref(0);
const allData = ref([]);
const initialBaseFactors = ref(null);
let onePollDataObj = {
  researchHypothesis: null,
  researcTasks: null,
  researchPdfImage: "",
  evolvingCodes: [],
  evolvingFeedbacks: [],
  userBaseFactors: null,
  feedbackCharts: null,
  feedbackMetric: null,
  feedbackConfig: null,
  feedbackHypothesis: null,
};
const currentData = ref(null);
const loadingIndex = ref(1);
const loopNumber = ref(props.loopNumber);
const updateEnd = ref(false);
const pauseEnd = ref(false);
const endTagHandled = ref(false);

const userInteractionVisible = ref(false);
const userInteractionSubmitting = ref(false);
const userInteractionQueue = ref([]);
const userInteractionEntries = ref([]);
const userInteractionOriginalPayload = ref({});
const autoSkipInteraction = ref(false);
const userInteractionMinimized = ref(false);
const userInteractionWaitingHypothesis = ref(false);
const userInteractionLastFeedbackEntries = ref([]);
let userInteractionTimeout = null;
const userInstructionPlaceholder = "Example: 使用中文来生成假设";
const featureRows = ref([]);
const featureValidationMsg = ref("");
const localFeatureError = ref("");
const featureTooltip = reactive({
  visible: false,
  text: "",
  left: 0,
  top: 0,
});
const featureTooltipOffset = { x: 12, y: 18 };

const availableFeatureTags = computed(() => {
  const used = new Set(
    featureRows.value
      .map((row) => (row.name == null ? "" : String(row.name).trim()))
      .filter(Boolean)
  );
  return Object.keys(ALPHA158)
    .filter((name) => !used.has(name))
    .map((name) => ({ name, expression: ALPHA158[name] }));
});

const configuredFeatureCount = computed(
  () =>
    featureRows.value.filter((row) => {
      const name = row.name == null ? "" : String(row.name).trim();
      const expression =
        row.expression == null ? "" : String(row.expression).trim();
      return Boolean(name) && Boolean(expression);
    }).length
);

const traceName = computed(() => {
  const traceId = String(props.id || "").trim();

  if (!traceId) {
    return "";
  }

  const separatorIndex = traceId.indexOf("/");
  return separatorIndex === -1 ? traceId : traceId.slice(separatorIndex + 1);
});

const isFeedbackInteraction = computed(() => {
  const payload = userInteractionOriginalPayload.value || {};
  if (userInteractionWaitingHypothesis.value) {
    return false;
  }
  return !Object.prototype.hasOwnProperty.call(payload, "hypothesis");
});

const isUserInstructionInteraction = computed(() => {
  const payload = userInteractionOriginalPayload.value || {};
  if (userInteractionWaitingHypothesis.value) {
    return false;
  }
  return Object.prototype.hasOwnProperty.call(payload, "user_instruction");
});

const isFeatureInteraction = computed(() => {
  const payload = userInteractionOriginalPayload.value || {};
  if (userInteractionWaitingHypothesis.value) {
    return false;
  }
  return Object.prototype.hasOwnProperty.call(payload, "features");
});

const normalizeDecision = (value) => {
  if (value === true || value === false) return value;
  if (value == null) return false;
  if (typeof value === "string") {
    return value.trim().toLowerCase() === "true";
  }
  return Boolean(value);
};

const isFeedbackPayload = (payload) => {
  if (!payload || typeof payload !== "object") return false;
  if (Object.prototype.hasOwnProperty.call(payload, "user_instruction")) {
    return false;
  }
  if (Object.prototype.hasOwnProperty.call(payload, "features")) {
    return false;
  }
  return !Object.prototype.hasOwnProperty.call(payload, "hypothesis");
};

const openUserInteraction = (payload) => {
  const hasUserInstruction =
    payload && Object.prototype.hasOwnProperty.call(payload, "user_instruction");
  const hasFeatures =
    payload && Object.prototype.hasOwnProperty.call(payload, "features");
  if (autoSkipInteraction.value && !hasUserInstruction && !hasFeatures) {
    if (userInteractionSubmitting.value) {
      userInteractionQueue.value.push(payload || {});
      return;
    }
    submitUserInteractionPayload(payload || {});
    return;
  }
  if (userInteractionVisible.value && !userInteractionWaitingHypothesis.value) {
    userInteractionQueue.value.push(payload);
    return;
  }
  if (userInteractionWaitingHypothesis.value) {
    userInteractionWaitingHypothesis.value = false;
  }
  userInteractionOriginalPayload.value = payload || {};
  const hasHypothesis =
    payload && Object.prototype.hasOwnProperty.call(payload, "hypothesis");
  const filteredKeys = hasFeatures
    ? []
    : hasUserInstruction
      ? ["user_instruction"]
      : hasHypothesis
        ? ["hypothesis", "reason"]
        : ["decision", "reason"];
  const entries = filteredKeys.map((key) => ({
    key,
    value:
      payload && Object.prototype.hasOwnProperty.call(payload, key)
        ? key === "decision"
          ? normalizeDecision(payload[key])
          : payload[key] == null
            ? ""
            : String(payload[key])
        : key === "decision"
          ? false
          : "",
  }));
  if (hasFeatures) {
    const featureDict = payload && payload.features ? payload.features : {};
    featureRows.value = Object.entries(featureDict).map(([name, expression]) => ({
      name: String(name),
      expression: expression == null ? "" : String(expression),
    }));
    if (featureRows.value.length === 0) {
      featureRows.value.push({ name: "", expression: "" });
    }
    localFeatureError.value = "";
    featureValidationMsg.value =
      payload && payload.feature_validation_msg
        ? String(payload.feature_validation_msg)
        : "";
  } else {
    featureRows.value = [];
    featureValidationMsg.value = "";
    localFeatureError.value = "";
  }
  userInteractionEntries.value = entries;
  userInteractionVisible.value = true;
  userInteractionMinimized.value = false;
  userInteractionWaitingHypothesis.value = false;
  if (userInteractionTimeout) {
    clearTimeout(userInteractionTimeout);
  }
  userInteractionTimeout = setTimeout(() => {
    submitOriginalUserInteraction();
  }, 10 * 60 * 1000);
};

const closeUserInteraction = () => {
  userInteractionVisible.value = false;
  userInteractionMinimized.value = false;
  userInteractionWaitingHypothesis.value = false;
  userInteractionLastFeedbackEntries.value = [];
  userInteractionEntries.value = [];
  userInteractionOriginalPayload.value = {};
  if (userInteractionTimeout) {
    clearTimeout(userInteractionTimeout);
    userInteractionTimeout = null;
  }
  if (userInteractionQueue.value.length > 0) {
    const nextPayload = userInteractionQueue.value.shift();
    openUserInteraction(nextPayload);
  }
};

const minimizeUserInteraction = () => {
  userInteractionMinimized.value = true;
};

const restoreUserInteraction = () => {
  userInteractionMinimized.value = false;
};

const submitUserInteractionPayload = (payload) => {
  if (userInteractionSubmitting.value) return;
  userInteractionSubmitting.value = true;
  const feedbackPayload = isFeedbackPayload(payload);
  const data = {
    id: props.id,
    payload,
  };
  return submitUserInteraction(data)
    .then(() => {
      if (feedbackPayload && userInteractionVisible.value) {
        userInteractionWaitingHypothesis.value = true;
        userInteractionLastFeedbackEntries.value =
          userInteractionEntries.value.map((entry) => ({
            key: entry.key,
            value: entry.value,
          }));
        userInteractionEntries.value = [];
        userInteractionOriginalPayload.value = {};
        if (userInteractionTimeout) {
          clearTimeout(userInteractionTimeout);
          userInteractionTimeout = null;
        }
        return;
      }
      closeUserInteraction();
    })
    .finally(() => {
      userInteractionSubmitting.value = false;
      if (autoSkipInteraction.value && userInteractionQueue.value.length > 0) {
        const nextPayload = userInteractionQueue.value.shift();
        submitUserInteractionPayload(nextPayload || {});
      }
    });
};

const submitUserInteractionForm = () => {
  const payload = { ...(userInteractionOriginalPayload.value || {}) };
  if (isFeatureInteraction.value) {
    const features = {};
    const seenNames = new Set();
    localFeatureError.value = "";
    for (const row of featureRows.value) {
      const name = row.name == null ? "" : String(row.name).trim();
      const expression =
        row.expression == null ? "" : String(row.expression).trim();
      if (!name || !expression) {
        localFeatureError.value =
          "Feature name and expression cannot be empty.";
        break;
      }
      if (seenNames.has(name)) {
        localFeatureError.value = "Feature names must be unique.";
        break;
      }
      seenNames.add(name);
      features[name] = expression;
    }
    if (localFeatureError.value) {
      return;
    }
    if (!initialBaseFactors.value) {
      initialBaseFactors.value = { ...features };
    }
    submitUserInteractionPayload(features);
    return;
  }
  userInteractionEntries.value.forEach((entry) => {
    if (entry.key === "decision") {
      payload[entry.key] = normalizeDecision(entry.value);
      return;
    }
    payload[entry.key] = entry.value == null ? "" : String(entry.value);
  });
  submitUserInteractionPayload(payload);
};

const addFeatureRow = () => {
  featureRows.value.push({ name: "", expression: "" });
};

const addFeatureFromPool = (tag) => {
  const emptyRow = featureRows.value.find(
    (row) => !row.name || !String(row.name).trim()
  );
  if (emptyRow) {
    emptyRow.name = tag.name;
    emptyRow.expression = tag.expression;
    return;
  }
  featureRows.value.push({ name: tag.name, expression: tag.expression });
};

const removeFeatureRow = (index) => {
  if (featureRows.value.length <= 1) {
    featureRows.value[0] = { name: "", expression: "" };
    return;
  }
  featureRows.value.splice(index, 1);
};

const showFeatureTooltip = (event, text) => {
  featureTooltip.visible = true;
  featureTooltip.text = text == null ? "" : String(text);
  moveFeatureTooltip(event);
};

const moveFeatureTooltip = (event) => {
  featureTooltip.left = event.clientX + featureTooltipOffset.x;
  featureTooltip.top = event.clientY + featureTooltipOffset.y;
};

const hideFeatureTooltip = () => {
  featureTooltip.visible = false;
};

const submitOriginalUserInteraction = () => {
  if (isFeatureInteraction.value) {
    const original = userInteractionOriginalPayload.value || {};
    const features = original.features || {};
    if (!initialBaseFactors.value) {
      initialBaseFactors.value = { ...features };
    }
    submitUserInteractionPayload(features);
    return;
  }
  submitUserInteractionPayload(userInteractionOriginalPayload.value || {});
};

const handleAutoSkipToggle = (enabled) => {
  autoSkipInteraction.value = enabled;
  if (!enabled) {
    return;
  }
  if (userInteractionVisible.value) {
    submitOriginalUserInteraction();
    return;
  }
  if (!userInteractionSubmitting.value && userInteractionQueue.value.length > 0) {
    const nextPayload = userInteractionQueue.value.shift();
    submitUserInteractionPayload(nextPayload || {});
  }
};

const clearAllDialogs = () => {
  showDialogForLoop.value = 0;
  userInteractionQueue.value = [];
  userInteractionVisible.value = false;
  userInteractionMinimized.value = false;
  userInteractionWaitingHypothesis.value = false;
  userInteractionLastFeedbackEntries.value = [];
  userInteractionEntries.value = [];
  userInteractionOriginalPayload.value = {};
  featureRows.value = [];
  featureValidationMsg.value = "";
  localFeatureError.value = "";
  featureTooltip.visible = false;
  if (userInteractionTimeout) {
    clearTimeout(userInteractionTimeout);
    userInteractionTimeout = null;
  }
};

const saveCompletedTraceId = (traceId) => {
  const normalizedTraceId = String(traceId || "").trim();
  if (!normalizedTraceId) {
    return;
  }

  const savedTraceIds = JSON.parse(
    localStorage.getItem(completedTraceStorageKey) || "[]"
  );

  if (savedTraceIds.includes(normalizedTraceId)) {
    return;
  }

  savedTraceIds.push(normalizedTraceId);
  localStorage.setItem(
    completedTraceStorageKey,
    JSON.stringify(savedTraceIds)
  );
};

const getEndTraceId = (content) => {
  const candidateTraceId =
    content?.trace_id ?? content?.traceId ?? content?.id ?? props.id;
  return String(candidateTraceId || "").trim();
};

const getEndMessage = (content) => {
  const errorMsg = content?.error_msg == null ? "" : String(content.error_msg).trim();
  const traceId = getEndTraceId(content);
  const baseMessage = errorMsg || "RD-Agent process has completed.";

  return traceId ? `${baseMessage} [${traceId}]` : baseMessage;
};

const getEndMessageType = (content) => {
  const endCode = Number(content?.end_code);
  return endCode === 0 || endCode === -1 ? "success" : "error";
};

const handleEndTag = (content) => {
  if (endTagHandled.value) {
    return;
  }

  const resolvedTraceId = getEndTraceId(content);
  const endMessage = getEndMessage(content);
  const endMessageType = getEndMessageType(content);
  endTagHandled.value = true;
  saveCompletedTraceId(resolvedTraceId);
  clearAllDialogs();
  updateEnd.value = true;
  loopNumber.value = allData.value.length;
  const endNotification = ElNotification({
    title: endMessageType === "success" ? "Completed" : "Run Ended",
    message: endMessage,
    type: endMessageType,
    position: "top-right",
    duration: 5000,
    showClose: true,
    offset: 24,
    onClick: () => {
      endNotification.close();
    },
  });
};

const loopClickFlag = ref(false);

let feedbackConfig = null;
let pdfImageTemp = "";
const firstPollFlag = ref(true);
function getData(data) {
  data.forEach((item) => {
    if (item.tag == "feedback.hypothesis_feedback") {
      onePollDataObj.feedbackHypothesis = item.content;
      if (!loopClickFlag.value) {
        tabChange(2);
      }
    } else if (item.tag == "feedback.config") {
      onePollDataObj.feedbackConfig = item.content;
      feedbackConfig = item.content;
    } else if (item.tag == "research.pdf_image") {
      pdfImageTemp = url + item.content.image;
    } else if (item.tag == "research.hypothesis") {
      // General Model Implementation 没有research.hypothesis
      if (!firstPollFlag.value) {
        loadingIndex.value += 1;
        allData.value.push(Object.assign({}, onePollDataObj));
        if (loopNumber.value <= allData.value.length) {
          loopNumber.value = allData.value.length + 1;
        }
        onePollDataObj = {
          researchHypothesis: null,
          researcTasks: null,
          researchPdfImage: pdfImageTemp,
          evolvingCodes: [],
          evolvingFeedbacks: [],
          userBaseFactors: null,
          feedbackCharts: null,
          feedbackMetric: null,
          feedbackConfig: feedbackConfig,
          feedbackHypothesis: null,
        };
      }
      if (!loopClickFlag.value) {
        tabChange(0);
        currentData.value = onePollDataObj;
      }
      onePollDataObj.researchHypothesis = item.content;
      firstPollFlag.value = false;
    } else if (item.tag == "research.tasks") {
      if (!developer.value) {
        if (!firstPollFlag.value) {
          loadingIndex.value += 1;
          allData.value.push(Object.assign({}, onePollDataObj));
          if (loopNumber.value <= allData.value.length) {
            loopNumber.value = allData.value.length + 1;
          }
          onePollDataObj = {
            researchHypothesis: null,
            researcTasks: null,
            researchPdfImage: pdfImageTemp,
            evolvingCodes: [],
            evolvingFeedbacks: [],
            userBaseFactors: null,
            feedbackCharts: null,
            feedbackMetric: null,
            feedbackConfig: feedbackConfig,
            feedbackHypothesis: null,
          };
        }
        if (!loopClickFlag.value) {
          tabChange(0);
          currentData.value = onePollDataObj;
        }
        firstPollFlag.value = false;
      }
      onePollDataObj.researcTasks = item.content;
      onePollDataObj.researchPdfImage = pdfImageTemp;
      pdfImageTemp = "";
    } else if (item.tag == "evolving.codes") {
      onePollDataObj.evolvingCodes.push(item);
    } else if (item.tag == "evolving.feedbacks") {
      onePollDataObj.evolvingFeedbacks.push(item);
      if (!loopClickFlag.value) {
        tabChange(1);
      }
    } else if (item.tag == "feedback.return_chart") {
      onePollDataObj.feedbackCharts = item.content;
    } else if (item.tag == "feedback.metric") {
      // 场景多只需要显示这四个
      //    "IC",
      // "1day.excess_return_without_cost.annualized_return",
      // "1day.excess_return_without_cost.information_ratio",
      // "1day.excess_return_without_cost.max_drawdown",
      const metricResult = JSON.parse(item.content.result);
      if (Object.keys(metricResult).length > 4) {
        onePollDataObj.feedbackMetric = {
          IC: metricResult["IC"],
          "1day.excess_return_without_cost.annualized_return":
            metricResult["1day.excess_return_without_cost.annualized_return"],
          "1day.excess_return_without_cost.information_ratio":
            metricResult["1day.excess_return_without_cost.information_ratio"],
          "1day.excess_return_without_cost.max_drawdown":
            metricResult["1day.excess_return_without_cost.max_drawdown"],
        };
      } else {
        onePollDataObj.feedbackMetric = metricResult;
      }
    } else if (item.tag == "END") {
      allData.value.push(Object.assign({}, onePollDataObj));
      userInteractionWaitingHypothesis.value = false;
      handleEndTag(item.content || {});
    } else if (item.tag == "user_interaction.request" && !endTagHandled.value) {
      openUserInteraction(item.content || {});
    }
  });
  if (!loopClickFlag.value) {
    currentData.value = Object.assign({}, onePollDataObj);
  }
}
const tabChange = (index) => {
  // moveSlider(index, tabProcessIndex.value);
  tabProcessIndex.value = index;
};
const addLoop = (flag) => {
  showDialogForLoop.value += 1;
};
const clickIndex = (obj) => {
  if (obj.loading) {
    loopClickFlag.value = false;
    currentData.value = Object.assign({}, onePollDataObj);
  } else {
    loopClickFlag.value = true;
    currentData.value = allData.value[obj.index - 1];
  }
};

const clickStop = (flag) => {
  stopFlag.value = flag;
  controlBtn("stop");
  updateEnd.value = true;
  loopNumber.value = allData.value.length;
};
// "stop"
const controlBtn = (action) => {
  const data = {
    id: props.id,
    action: action,
  };
  control(data).then((response) => {
    console.log(response);
  });
};
const firstTrace = () => {
  endTagHandled.value = false;
  firstPollFlag.value = true;
  allData.value = [];
  initialBaseFactors.value = null;
  onePollDataObj = {
    researchHypothesis: null,
    researcTasks: null,
    researchPdfImage: "",
    evolvingCodes: [],
    evolvingFeedbacks: [],
    userBaseFactors: null,
    feedbackCharts: null,
    feedbackMetric: null,
    feedbackConfig: null,
    feedbackHypothesis: null,
  };
  const data = {
    id: props.id,
    all: true,
    reset: true, // 从第一个log msg开始返回
  };
  trace(data).then((response) => {
    if (response && response.length > 0) {
      getData(response);
    }

    if (
      response &&
      response.length > 0 &&
      response[response.length - 1].tag == "END"
    ) {
      handleEndTag(response[response.length - 1].content || {});
      console.log("allData: ", allData.value);
    } else {
      tracePoll();
    }
  });
};
const tracePoll = () => {
  if (stopFlag.value) {
    return;
  }
  endTagHandled.value = false;
  updateEnd.value = false;
  const data = {
    id: props.id,
    all: true,
    reset: false, // 从第一个log msg开始返回
  };
  trace(data).then((response) => {
    if (response && response.length > 0) {
      getData(response);
    }
    if (
      response &&
      response.length > 0 &&
      response[response.length - 1].tag == "END"
    ) {
      handleEndTag(response[response.length - 1].content || {});
      console.log("allData: ", allData.value);
    } else {
      setTimeout(tracePoll, 3000);
    }
  });
};

function moveSlider(index, preIndex, init) {
  const tab = tabs.value;
  const currentTab = tab.children[index];
  const tabsWidth = tab.getBoundingClientRect().width;
  const currentTabWidth = currentTab.getBoundingClientRect().width;
  const rightPercentage =
    ((currentTab.offsetLeft - 15 + currentTabWidth) * 100) / tabsWidth + "%";
  const leftPercentage = ((currentTab.offsetLeft + 15) * 100) / tabsWidth + "%";

  const sideProperty = (withTimer) => {
    if (!withTimer) return index > preIndex ? "--right-side" : "--left-side";

    return index > preIndex ? "--left-side" : "--right-side";
  };

  const sidePercentage = (withTimer) => {
    if (!withTimer) return index > preIndex ? rightPercentage : leftPercentage;

    return index > preIndex ? leftPercentage : rightPercentage;
  };

  tab.style.setProperty(sideProperty(), sidePercentage());

  if (init) {
    tab.style.setProperty(sideProperty(true), sidePercentage(true));

    return;
  }

  if (transitionTimer) {
    clearTimeout(transitionTimer);
  }

  transitionTimer = setTimeout(() => {
    tab.style.setProperty(sideProperty(true), sidePercentage(true));
    transitionTimer = undefined;
  }, 350);
}

onMounted(() => {
  firstTrace();
});

// 在组件被卸载前移除全局点击事件监听
onUnmounted(() => {});
</script>

<style scoped lang="scss">
.playground-page-root {
  height: 100%;
  overflow: hidden;
}

.playground-page {
  display: flex;
  width: 100%;
  height: 100%;
  min-height: 0;
  overflow: hidden;
}
.main-content {
  flex: 1;
  min-width: 0;
  min-height: 0;
  overflow: hidden;
  .tab-title {
    display: flex;
    justify-content: center;
    .tab-box {
      display: flex;
      gap: 2.7em;
      .tab-item-btn {
        display: flex;
        width: 8.6em;
        height: 3em;
        padding: 0px 0.99em;
        justify-content: space-between;
        align-items: center;
        border-radius: 24px;
        background: var(--bg-white);
        color: var(--text-color);
        font-size: 0.81em;
        font-weight: 700;
        line-height: 150%;
        text-transform: uppercase;
        cursor: pointer;
        box-shadow: 1px 1px 2px 0px rgba(255, 255, 255, 0.3) inset,
          -1px -1px 2px 0px rgba(235, 235, 235, 0.5) inset,
          -3px 3px 6px 0px rgba(235, 235, 235, 0.2),
          3px -3px 6px 0px rgba(235, 235, 235, 0.2),
          -3px -3px 6px 0px rgba(255, 255, 255, 0.9),
          3px 3px 8px 0px rgba(235, 235, 235, 0.9);
        &.active {
          background: linear-gradient(90deg, #2667ff 0%, #9d41ff 100%);
          box-shadow: 1px 1px 2px 0px rgba(255, 255, 255, 0.3) inset,
            -1px -1px 2px 0px rgba(235, 235, 235, 0.5) inset,
            -3px 3px 6px 0px rgba(235, 235, 235, 0.2),
            3px -3px 6px 0px rgba(235, 235, 235, 0.2),
            -3px -3px 6px 0px rgba(255, 255, 255, 0.9),
            3px 3px 8px 0px rgba(235, 235, 235, 0.9);
          color: var(--text-white-color);
        }
        img {
          width: 3em;
          height: 3em;
          margin: 0 auto;
        }
        span {
          display: flex;
          align-items: center;
          gap: 0.9em;
        }
        .pg-tab-icon {
          width: 1.08em;
          height: 1.08em;
        }
        .arrow-right-icon {
          width: 0.504em;
          height: 1.08em;
        }
      }
    }
  }
  .nav-content {
    margin-top: 0.45em;
    nav {
      display: flex;
      padding: 0 2.7em;
      ul {
        display: flex;
        position: relative;
        --left-side: 0;
        --right-side: 0;
        background-color: var(--bg-white-blue-color);
        border-radius: 0 0 20px 20px;

        li {
          border-radius: 0 0 20px 20px;
          background-color: #fff;
          width: 9.9em;

          .tab-bg {
            width: 100%;
            padding: 0 1.413em;
            height: 2.25em;
            box-sizing: border-box;
            display: flex;
            align-items: center;
            justify-content: center;
            gap: 0.45em;
            cursor: pointer;
            -webkit-user-select: none;
            -moz-user-select: none;
            -ms-user-select: none;
            user-select: none;
            color: rgba(0, 0, 0, 0.3);
            text-shadow: 8px 11px 30px var(--wg-shadow-color);
            font-size: 1.06875em;
            font-weight: 700;
            line-height: 200%;
          }

          img {
            width: 2.7em;
            height: 2.7em;
            flex-shrink: 0;
          }

          span {
            cursor: default;
          }

          .tab-label--clickable {
            cursor: pointer;
          }
          &.borderRadius-left {
            border-radius: 0 0 0 20px;
          }
          &.borderRadius-right {
            border-radius: 0 0 20px 0;
          }
          &.borderRadius-none {
            border-radius: 0;
          }

          &.active {
            .tab-bg {
              background-color: var(--bg-white-blue-color);
              border-radius: 20px 20px 0 0;
            }
            span {
              background: linear-gradient(90deg, #2667ff 0%, #9d41ff 100%);
              background-clip: text;
              -webkit-background-clip: text;
              -webkit-text-fill-color: transparent;
            }
          }
        }
      }
    }
  }
  .bg-content {
    width: 100%;
    height: calc(100vh - 14.5em);
    overflow: hidden;
    box-sizing: border-box;
    padding: 0.9em 1.8em;
    justify-content: center;
    align-items: center;
    border-radius: 20px;
    background: var(--bg-white-blue-color);
    position: relative;
    z-index: 1;
  }
}

.dialog-box {
  width: 100vw;
  height: 100vh;
  position: fixed;
  left: 0;
  top: 0;
  background: rgba(255, 255, 255, 0.29);
  backdrop-filter: blur(4.6px);
  z-index: 999999;
  display: flex;
  align-items: center;
  justify-content: center;
}

.dialog-content.user-interaction-dialog {
  background-color: #fff;
  border-radius: 18px;
  --border-radius: 20px;
  --border-width: 2px;
  padding: 3.5em 4.5em;
  max-width: 72em;
  width: calc(100% - 4em);
  font-family: inherit;
  .dialog-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 1.5em;
  }
  .dialog-minimize {
    border: none;
    background: linear-gradient(90deg, #2667ff 0%, #9d41ff 100%);
    color: #fff;
    font-size: 1em;
    font-weight: 700;
    padding: 0.6em 1.2em;
    border-radius: 999px;
    box-shadow: 0 8px 18px rgba(38, 103, 255, 0.35);
    cursor: pointer;
  }
  h1 {
    color: var(--text-color);
    text-shadow: 8px 11px 30px #edf0ff;
    font-size: 1.7em;
    font-weight: 700;
    line-height: 200%;
  }
  p {
    color: var(--text-color);
    font-size: 1.1em;
    line-height: 150%;
    margin: 0.8em 0 1.5em;
  }
  .feature-validation-msg {
    padding: 0.75em 1em;
    margin: 0 0 1.2em;
    border-radius: 10px;
    background: rgba(255, 107, 0, 0.08);
    color: #b94b00;
    font-weight: 600;
    line-height: 1.5;
  }
  .interaction-form {
    display: flex;
    flex-direction: column;
    gap: 0.9em;
    max-height: none;
    overflow: visible;
  }
  .feature-table {
    display: flex;
    flex-direction: column;
    gap: 1em;
  }
  .feature-layout {
    display: grid;
    grid-template-columns: 1fr 3fr;
    gap: 1.5em;
    align-items: start;
  }
  .feature-pool-block {
    display: flex;
    flex-direction: column;
    gap: 0.6em;
  }
  .feature-pool {
    padding: 0.9em 1em;
    border-radius: 12px;
    background: rgba(38, 103, 255, 0.06);
    border: 1px solid rgba(38, 103, 255, 0.2);
    max-height: 56vh;
    overflow: visible;
  }
  .feature-pool-title {
    font-weight: 700;
    color: #1c2b57;
    margin-bottom: 0.6em;
  }
  .feature-pool-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 0.6em;
    max-height: 56vh;
    overflow: auto;
    overflow-x: hidden;
  }
  .feature-editor {
    display: flex;
    flex-direction: column;
    gap: 0.9em;
    max-height: 56vh;
    overflow: auto;
    padding-right: 0.4em;
  }
  .feature-sticky-head {
    position: sticky;
    top: 0;
    z-index: 3;
    background: #fff;
    padding-bottom: 0.2em;
  }
  .feature-editor-meta {
    font-weight: 700;
    color: #1c2b57;
    font-size: 0.95em;
    line-height: 1.4;
    padding: 0.15em 0.2em 0.45em;
  }
  .feature-tag {
    border: 1px solid rgba(38, 103, 255, 0.35);
    background: #fff;
    color: #1c2b57;
    font-weight: 600;
    font-size: 0.9em;
    padding: 0.35em 0.7em;
    border-radius: 999px;
    cursor: pointer;
    position: relative;
  }
  .feature-tag:hover {
    background: rgba(38, 103, 255, 0.12);
  }
  .feature-header,
  .feature-row {
    display: grid;
    grid-template-columns: 1fr 4fr auto;
    gap: 1.4em;
    align-items: center;
  }
  .feature-header {
    font-weight: 700;
    color: #1c2b57;
    font-size: 0.95em;
    text-transform: uppercase;
    letter-spacing: 0.04em;
    background: #fff;
    padding: 0.45em 0.2em;
    border-bottom: 1px solid #e0e6f5;
  }
  .feature-input {
    width: 100%;
    padding: 0.45em 0.7em;
    border-radius: 10px;
    border: 1px solid #c5d2e6;
    font-size: 0.88em;
    color: var(--text-color);
    min-width: 0;
  }
  .feature-input--math {
    font-family: "STIX Two Math", "Cambria Math", "Times New Roman", serif;
  }
  .feature-add {
    align-self: flex-start;
    border: none;
    background: rgba(38, 103, 255, 0.12);
    color: #1c2b57;
    font-weight: 700;
    padding: 0.5em 1em;
    border-radius: 999px;
    cursor: pointer;
  }
  .feature-remove {
    border: 1px solid #ee6a58;
    background: #ee6a58;
    color: #fff;
    font-weight: 700;
    width: 1.7em;
    height: 1.7em;
    padding: 0;
    border-radius: 8px;
    cursor: pointer;
    white-space: nowrap;
    justify-self: end;
    font-size: 1em;
    display: inline-flex;
    align-items: center;
    justify-content: center;
    box-shadow: 0 6px 14px rgba(238, 106, 88, 0.3);
    transition: transform 0.15s ease, box-shadow 0.15s ease,
      background-color 0.15s ease;
  }
  .feature-remove:hover {
    background: #e15f4e;
    transform: translateY(-1px);
    box-shadow: 0 8px 18px rgba(238, 106, 88, 0.45);
  }
  .feature-remove:disabled {
    cursor: not-allowed;
    opacity: 0.5;
  }
  .interaction-form.read-only {
    opacity: 0.7;
    pointer-events: none;
  }
  .interaction-waiting {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.8em;
    min-height: 12em;
    font-size: 1.15em;
    font-weight: 600;
    color: var(--text-color);
  }
  .interaction-waiting-spinner {
    width: 1.3em;
    height: 1.3em;
    border-radius: 999px;
    border: 2px solid rgba(38, 103, 255, 0.2);
    border-top-color: #2667ff;
    animation: dialog-spin 0.9s linear infinite;
  }
  .dialog-content.user-interaction-dialog.user-interaction-dialog--wide {
    max-width: 88em;
    padding: 3.75em 5.25em;
  }
  .interaction-row {
    display: flex;
    align-items: flex-start;
    gap: 1em;
  }
  .interaction-row--stack {
    flex-direction: column;
    align-items: stretch;
  }
  .interaction-key--highlight {
    width: 100%;
    font-size: 1.1em;
    font-weight: 700;
    color: #1c2b57;
    margin-bottom: 0.4em;
    text-shadow: 0 8px 20px rgba(38, 103, 255, 0.18);
    white-space: nowrap;
  }
  .interaction-key {
    width: 12%;
    font-weight: 600;
    color: var(--text-color);
    word-break: break-all;
    font-size: 1em;
    line-height: 1.2;
    padding-top: 0.2em;
  }
  .interaction-textarea {
    flex: 1;
    min-height: 14em;
    padding: 1em 1.1em;
    border-radius: 10px;
    border: 1px solid #c5d2e6;
    color: var(--text-color);
    font-size: 1.1em;
    font-family: inherit;
    outline: none;
    resize: vertical;
    line-height: 1.5;
    &::placeholder {
      font-style: italic;
    }
  }
  .interaction-select {
    flex: 1;
    min-height: 3.2em;
    padding: 0.6em 1.1em;
    border-radius: 10px;
    border: 1px solid #c5d2e6;
    color: var(--text-color);
    font-size: 1.05em;
    font-family: inherit;
    outline: none;
    background: #fff;
  }
  .btn-box {
    display: flex;
    justify-content: space-between;
    padding: 0 0.25em;
    position: relative;
    z-index: 1;
    margin-top: 2.5em;
    button {
      width: 10em;
      height: 3em;
      color: var(--text-color);
      font-size: 1.05em;
      font-weight: 700;
      line-height: 150%;
      text-transform: uppercase;
      border: none;
      cursor: pointer;
      --border-radius: 999px;
      --border-width: 2px;
      &.active {
        border-radius: 37.5px;
        background: linear-gradient(90deg, #2667ff 0%, #9d41ff 100%), #979797;
        box-shadow: 8px 11px 30px 0px var(--wg-shadow-color);
        color: #fff;
      }
      &.back:hover {
        background-color: var(--card-bg-hover-color);
      }
      &:disabled {
        cursor: not-allowed;
        opacity: 0.6;
      }
    }
  }
}

.feature-tag-floating-tooltip {
  position: fixed;
  z-index: 2000001;
  max-width: min(92vw, 96em);
  padding: 0.5em 0.7em;
  border-radius: 8px;
  background: #1c2b57;
  color: #fff;
  font-size: 0.95em;
  line-height: 1.4;
  white-space: normal;
  word-break: break-word;
  box-shadow: 0 10px 24px rgba(28, 43, 87, 0.25);
  pointer-events: none;
}

.dialog-content.user-interaction-dialog.user-interaction-dialog--wide {
  width: 86vw;
  max-width: 86vw;
  padding: 3.75em 4.5em;
}

.dialog-minimized {
  position: fixed;
  right: 1.8em;
  bottom: 1.8em;
  z-index: 1000000;
  display: flex;
  align-items: center;
  justify-content: center;
  cursor: pointer;
}

.dialog-minimized-content {
  display: flex;
  align-items: center;
  gap: 0.9em;
  padding: 1em 1.5em;
  border-radius: 999px;
  background: linear-gradient(90deg, #2667ff 0%, #9d41ff 100%);
  box-shadow: 0 16px 40px rgba(38, 103, 255, 0.35);
  color: #fff;
  font-weight: 700;
  border: 2px solid rgba(255, 255, 255, 0.65);
  animation: dialog-pulse 1.6s ease-in-out infinite;
}

.dialog-waiting-spinner {
  width: 1.2em;
  height: 1.2em;
  border-radius: 999px;
  border: 2px solid rgba(255, 255, 255, 0.45);
  border-top-color: #fff;
  animation: dialog-spin 0.9s linear infinite;
}

@keyframes dialog-spin {
  to {
    transform: rotate(360deg);
  }
}

@keyframes dialog-pulse {
  0%,
  100% {
    transform: translateY(0);
    box-shadow: 0 16px 40px rgba(38, 103, 255, 0.35);
  }
  50% {
    transform: translateY(-3px);
    box-shadow: 0 22px 48px rgba(38, 103, 255, 0.5);
  }
}

</style>
