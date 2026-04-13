<template>
  <div class="page-content">
    <div class="nav-bar">
      <span :class="{ active: !showPlayground && showPanel == 1 }" @click="Back"
        >Start</span
      >
      <span class="nav-separator">></span>
      <span
        v-if="showPlayground && currentScenarioLabel"
        class="nav-highlight-name"
      >
        {{ currentScenarioLabel }}
      </span>
      <span
        v-else
        @click="changePanel"
        :class="{
          active: !showPlayground && (showPanel == 2 || showPanel == 3),
        }"
        >Pick a scenario</span
      >
      <span class="nav-separator">></span>
      <span class="nav-highlight-name" v-if="showPlayground && currentTraceName">
        {{ currentTraceName }}
      </span>
      <span
        class="nav-summary"
        :class="{ active: showPlayground, 'with-trace': showPlayground && currentTraceName }"
        >Summary</span
      >
    </div>
    <div class="setup-content" v-show="!showPlayground">
      <div class="main-content" v-show="showPanel == 1">
        <h1>Let’s Get Started: Select Your Action</h1>
        <div class="card-box">
          <div class="card-item gradient-big-border" @click="changePanel">
            <h2>First time?</h2>
            <p>Select a scenario for your analysis</p>
            <img
              class="img1"
              src="@/assets/images/first-time-img.png"
              alt="R&D-Agent"
            />
          </div>
          <div class="card-item gradient-big-border" @click="openHistoryPanel">
            <h2>View previous traces?</h2>
            <p>Open a trace ID from an earlier run to review its history.</p>
            <img
              class="img2"
              src="@/assets/images/continue-img.png"
              alt="R&D-Agent"
            />
          </div>
        </div>
      </div>
      <div
        class="main-content"
        :class="{
          'split-two': scenarioCheckedIndex != -1,
          'no-upload': scenarioChecked && !scenarioChecked.upload,
        }"
        v-show="showPanel == 2"
      >
        <div class="select-upload">
          <h1
            class="h1"
            :class="{ margintop: scenarioChecked && !scenarioChecked.upload }"
            v-show="
              scenarioCheckedIndex == -1 ||
              (scenarioChecked && !scenarioChecked.upload)
            "
          >
            <p>Select a scenario for your analysis</p>
          </h1>
          <h1 class="h1" v-show="scenarioChecked && scenarioChecked.upload">
            Upload materials you want to analyze
          </h1>
          <div class="nav-content">
            <nav>
              <ul>
                <li :class="{ active: tabIndex == 0 }">
                  <el-tooltip
                    effect="dark"
                    :offset="8"
                    raw-content
                    content="<div style='width: 500px;font-size: 14px;padding: 0.5em 0.5em 0.7em;line-height:160%; '>R&D-Agent autonomously generates, implements, and tests ideas in iterative loops for continuous improvement and optimal performance.</div>"
                    placement="bottom"
                  >
                    <span @click="tabChange(0)">Continuous Exploration</span>
                  </el-tooltip>
                </li>
                <li :class="{ active: tabIndex == 1 }">
                  <el-tooltip
                    effect="dark"
                    :offset="8"
                    raw-content
                    content="<div style='width: 500px;font-size: 14px;padding: 0.5em 0.5em 0.7em;line-height:160%; '>R&D-Agent executes and tests user-provided ideas in limited loops, with the number of loops depending on the provided input for targeted outcomes.</div>"
                    placement="bottom"
                  >
                    <span @click="tabChange(1)">Guided Implementation</span>
                  </el-tooltip>
                </li>
                <div class="nav-line" ref="line"></div>
              </ul>
            </nav>
          </div>
          <div class="main-panel">
            <div class="title small-config-title">
              Scenario
            </div>
            <selectComponent
              :scenarioList="scenarioList"
              :scenarioIndex="scenarioCheckedIndex"
              @scenarioCheckedItem="scenarioCheckedItem"
            ></selectComponent>
            <div v-if="scenarioChecked && scenarioChecked.upload">
              <div class="title with-tip">
                Material
                <el-tooltip
                  effect="dark"
                  :offset="8"
                  content="Research reports, academic or conference papers, etc."
                  placement="top"
                >
                  <span class="tip-icon">?</span>
                </el-tooltip>
              </div>
              <el-upload
                drag
                multiple
                accept=".pdf"
                :auto-upload="false"
                :on-change="changeFile"
                :file-list="selectedFiles"
                :show-file-list="false"
                action="#"
              >
                <div class="upload-box">
                  <div class="upload-box-bg">
                    <span class="upload-small"></span>
                    <h3>research reports, papers, etc.</h3>
                    <p>(Supported format: .pdf)</p>
                  </div>
                </div>
              </el-upload>
              <div class="file-tag-list" v-if="selectedFiles.length">
                <span class="file-tag" v-for="file in selectedFiles" :key="file.uid">
                  <span class="tag-name">{{ file.name }}</span>
                  <span class="tag-close" @click="removeSelectedFile(file.uid)">
                    ×
                  </span>
                </span>
              </div>
            </div>
            <div
              class="loop-content"
              v-if="scenarioChecked && !scenarioChecked.upload"
            >
              <div class="title with-tip small-config-title">
                Material (Optional)
                <el-tooltip
                  effect="dark"
                  :offset="8"
                  content="Upload references or related files for this run."
                  placement="top"
                >
                  <span class="tip-icon">?</span>
                </el-tooltip>
              </div>
              <el-upload
                class="loop-upload"
                drag
                multiple
                accept=".json,.py"
                :auto-upload="false"
                :on-change="changeFile"
                :file-list="selectedFiles"
                :show-file-list="false"
                action="#"
              >
                <div class="upload-box">
                  <div class="upload-box-bg">
                    <span class="upload-small"></span>
                    <h3>Upload base factors</h3>
                    <p>base_factors.json and &lt;factor_name&gt;.py</p>
                  </div>
                </div>
              </el-upload>
              <div class="file-tag-list" v-if="selectedFiles.length">
                <span class="file-tag" v-for="file in selectedFiles" :key="file.uid">
                  <span class="tag-name">{{ file.name }}</span>
                  <span class="tag-close" @click="removeSelectedFile(file.uid)">
                    ×
                  </span>
                </span>
              </div>
              <div class="compact-setting-row">
                <div class="title with-tip compact-setting-title">
                  Loop count
                  <el-tooltip
                    effect="dark"
                    :offset="8"
                    content="Choose the number of R&D loops: 5, 10, 20, or customize."
                    placement="top"
                  >
                    <span class="tip-icon">?</span>
                  </el-tooltip>
                </div>
                <div class="radio-box compact-config-box compact-setting-box">
                  <el-radio-group
                    class="compact-radio-group"
                    v-model="loopRadio"
                    @change="radioChange"
                  >
                    <el-radio value="3">3 Loops</el-radio>
                    <el-radio value="5">5 Loops</el-radio>
                    <el-radio value="10">10 Loops</el-radio>
                    <el-radio value="-1"
                      ><el-input-number
                        class="number-input"
                        v-model="num"
                        :controls="false"
                        :min="1"
                        :max="100"
                        @change="handleChange"
                      />
                      Loops</el-radio
                    >
                  </el-radio-group>
                </div>
              </div>
              <div class="compact-setting-row is-second">
                <div class="title with-tip compact-setting-title">
                  Loop duration
                  <el-tooltip
                    effect="dark"
                    :offset="8"
                    content="Choose how many hours you want to run R&D-Agent: 6, 12, 24, or customize."
                    placement="top"
                  >
                    <span class="tip-icon">?</span>
                  </el-tooltip>
                </div>
                <div class="radio-box compact-config-box compact-setting-box">
                  <el-radio-group
                    class="compact-radio-group"
                    v-model="hourRadio"
                    @change="hourRadioChange"
                  >
                    <el-radio value="6">6 hours</el-radio>
                    <el-radio value="12">12 hours</el-radio>
                    <el-radio value="24">24 hours</el-radio>
                    <el-radio value="-1"
                      ><el-input-number
                        class="number-input"
                        v-model="num1"
                        :controls="false"
                        :min="1"
                        :max="48"
                        @change="handleChange1"
                      />
                      hours</el-radio
                    >
                  </el-radio-group>
                </div>
              </div>
            </div>
            <div
              class="btn-main"
              :style="{
                'margin-top':
                  scenarioChecked && scenarioChecked.upload ? '3.5em' : '2em',
              }"
            >
              <button class="gradient-border back" @click="Back">BACK</button>
              <button
                class="disable"
                v-if="!loading"
                @click="generate"
                :class="{
                  active:
                    (scenarioChecked && !scenarioChecked.upload) ||
                    selectedFiles.length > 0,
                  disable:
                    !scenarioChecked ||
                    (scenarioChecked &&
                      scenarioChecked.upload &&
                      selectedFiles.length === 0),
                }"
              >
                generate
              </button>
              <button class="active" v-if="loading">
                <loadingSvg></loadingSvg>
              </button>
            </div>
          </div>
        </div>
        <div class="intro-txt" v-if="scenarioCheckedIndex != -1">
          <div v-for="item in introName" :key="item">
            <h3>{{ item }}</h3>
            <markdown
              class="intro-markdown"
              :content="scenarioChecked.introduce[item]"
            ></markdown>
          </div>
        </div>
      </div>
      <div class="main-content" v-show="showPanel == 3">
        <h1 class="h1">
          View traces from previous runs <br />
          and inspect their execution history.
        </h1>
        <div class="main-panel history-panel">
          <div class="title">Trace ID List</div>
          <div class="desc">
            <p>Pick a scenario first, then choose one of its trace names</p>
          </div>
          <div class="history-select-row">
            <div class="history-select-item">
              <div class="title small-config-title">Scenario</div>
              <smSelectComponent
                :scenarioList="historyScenarioList"
                :scenarioIndex="historyScenarioCheckedIndex"
                placeholder="Select a scenario"
                @scenarioCheckedItem="historyScenarioCheckedItem"
              ></smSelectComponent>
            </div>
            <div class="history-select-item">
              <div class="title small-config-title">Trace name</div>
              <smSelectComponent
                :scenarioList="historyTraceList"
                :scenarioIndex="historyTraceCheckedIndex"
                placeholder="Select a trace name"
                @scenarioCheckedItem="historyTraceCheckedItem"
              ></smSelectComponent>
            </div>
          </div>
          <div
            class="btn-main"
            :style="{
              'margin-top':
                scenarioChecked && scenarioChecked.upload ? '3.5em' : '7.5em',
            }"
          >
            <button class="gradient-border back" @click="Back">BACK</button>
            <button
              class="disable"
              :class="{
                active: historyTraceChecked,
                disable: !historyTraceChecked,
              }"
              @click="viewTracePage"
            >
              view trace
            </button>
          </div>
        </div>
      </div>
    </div>
    <div class="playground-shell" v-if="showPlayground">
      <playgroundPage
        :id="id"
        :editLoop="editLoop"
        :scenarioName="scenarioName"
        :developer="developer"
        :loopNumber="loopNumber"
      ></playgroundPage>
    </div>
  </div>
</template>
<script setup>
import { computed, ref, watch, reactive, onMounted, onUnmounted, nextTick } from "vue";
import { ElMessage } from "element-plus";
import { getHistoryTraceIds, uploadFile } from "../utils/api";
import selectComponent from "../components/select-component.vue";
import smSelectComponent from "../components/sm-select-component.vue";
import loadingSvg from "../components/loading-dot.vue";
import markdown from "../components/markdown.vue";
import playgroundPage from "./PlaygroundPage.vue";
import { useRouter } from "vue-router";
import { kaggleCompetitions } from "../constants/mle-competitions";
const router = useRouter();
const completedTraceStorageKey = "completedTraceIdList";
const showPanel = ref(1);
const showPlayground = ref(false);
const uploaDone = ref(false);
const loading = ref(false);
const uploadMatchedLoopScenarios = new Set(["Finance Data Building (Reports)"]);

const loopRadio = ref("3");
const loopNumber = ref(3);
const hourRadio = ref("6");
const hourNumber = ref(6);
const scenarioName = ref("");
const num = ref();
const num1 = ref();
const continuousUploadExtensions = [".json", ".py"];
const guidedUploadExtensions = [".pdf"];

const getTraceNameFromId = (traceId) => {
  const normalizedTraceId = String(traceId || "").trim();

  if (!normalizedTraceId) {
    return "";
  }

  const separatorIndex = normalizedTraceId.indexOf("/");
  return separatorIndex === -1
    ? normalizedTraceId
    : normalizedTraceId.slice(separatorIndex + 1);
};

const currentTraceName = computed(() => getTraceNameFromId(id.value));
const currentScenarioLabel = computed(() => {
  const name = String(scenarioName.value || scenarioChecked.value?.name || "").trim();

  return name ? name : "";
});

const shouldMatchLoopCountToUploads = () => {
  return (
    scenarioChecked.value &&
    uploadMatchedLoopScenarios.has(scenarioChecked.value.name)
  );
};

const syncLoopCountWithSelectedFiles = () => {
  if (!shouldMatchLoopCountToUploads()) {
    return;
  }

  loopRadio.value = "-1";
  loopNumber.value = selectedFiles.value.length;
  num.value = selectedFiles.value.length;
};

const getAllowedUploadExtensions = () => {
  if (scenarioChecked.value && scenarioChecked.value.upload) {
    return guidedUploadExtensions;
  }

  return continuousUploadExtensions;
};

const isAllowedUploadFile = (file) => {
  const fileName = String(file?.name || "").trim().toLowerCase();
  const allowedExtensions = getAllowedUploadExtensions();

  return allowedExtensions.some((extension) => fileName.endsWith(extension));
};

const handleChange = (value) => {
  if (loopRadio.value == -1) {
    loopNumber.value = Number(value);
  }
};
const handleChange1 = (value) => {
  if (hourRadio.value == -1) {
    hourNumber.value = Number(value);
  }
};
const radioChange = (value) => {
  if (value == -1) {
    loopNumber.value = Number(num.value);
  } else {
    loopNumber.value = Number(value);
  }
};
const hourRadioChange = (value) => {
  if (value == -1) {
    hourNumber.value = Number(num1.value);
  } else {
    hourNumber.value = Number(value);
  }
};

const continuousScenarioList = [
  {
    name: "Finance Data Building",
    icon: "Piggy-Bank",
    color: "#2e65ff",
    upload: false,
    developer: true,
    editLoop: true,
    hourRadio: "6",
    hourNumber: 6,
    loopRadio: "3",
    loopNumber: 3,
    introduce: {
      Introduction: `Applying R&D-Agent on finance Data Agent to automate the iterative process of evolving and trading financial factors by proposing, developing, evaluating, and refining them. The scenario is built on Qlib. `,
      "Data Description": `The dataset is includes daily stock data from the CSI300 index, with training data from 2008-2014, validation data from 2015-2016, and test data from 2017-2020. `,
      "Evaluation Method": `The performance of new financial factors is assessed through quantitative backtesting using Qlib. This process evaluates both the prediction accuracy and the final profit. `,
      "Scenario Breakdown": `... Round♾️ N:
  	→ [🔍Research to generate hypothesis] → (hypothesis)
  	→ [🔍Design Experiment] → (Experiment Tasks)
  	→ [🛠️Experiment Implementation] → (Iterative Implementation in workspace)
  	→ [📝Evaluation and Analysis] → (Feedbacks)
  → ...Next Round♾️... `,
    },
  },
  {
    name: "Finance Model Implementation",
    icon: "Piggy-Bank",
    color: "#595cff",
    upload: false,
    developer: true,
    editLoop: true,
    hourRadio: "6",
    hourNumber: 6,
    loopRadio: "3",
    loopNumber: 3,
    introduce: {
      Introduction: `Applying R&D-Agent on finance data to automate iterative model evolution and quantitative trading by generating, implementing, and refining financial models for optimal performance. The scenario is built on Qlib. `,
      "Data Description": `The dataset includes daily stock data from the CSI300 index, with training data from 2008-2014, validation data from 2015-2016, and test data from 2017-2020. `,
      "Evaluation Method": `The performance of new developed models is assessed through quantitative backtesting using Qlib. This process evaluates both the prediction accuracy and the final profit. `,
      "Scenario Breakdown": `... Round♾️ N:
  	→ [🔍Research to generate hypothesis] → (hypothesis)
  	→ [🔍Design Experiment] → (Experiment Tasks)
  	→ [🛠️Experiment Implementation] → (Iterative Implementation in workspace)
  	→ [📝Evaluation and Analysis] → (Feedbacks)
  → ...Next Round♾️... `,
    },
  },
  {
    name: "Finance Whole Pipeline",
    icon: "Tablet-Capsule",
    color: "#6d52ff",
    upload: false,
    developer: true,
    editLoop: true,
    hourRadio: "6",
    hourNumber: 6,
    loopRadio: "3",
    loopNumber: 3,
    introduce: {
      Introduction: `R&D-Agent runs a full finance pipeline on Qlib, combining Finance Data Building and Finance Model Implementation. In each loop, the LLM decides whether to focus on factor engineering or model implementation based on current feedback.`,
      "Data Description": `Daily CSI300 stock data is used (train: 2008-2014, valid: 2015-2016, test: 2017-2020). Each round may work on factors or models, depending on what the LLM judges as most beneficial.`,
      "Evaluation Method": `Each loop is validated by quantitative backtesting in Qlib. Backtesting results are fed back to the LLM, which then chooses the next focus (factor or model) to improve prediction and trading performance.`,
      "Scenario Breakdown": `... Round♾️ N:
  	→ [🔍Research + Planning] → (LLM chooses factor or model focus)
  	→ [🔍Design Experiment] → (Tasks for the selected focus)
  	→ [🛠️Experiment Implementation] → (Iterative implementation in workspace)
  	→ [📝Evaluation and Analysis] → (Backtesting feedback)
  	→ [🔁Next Round] → (LLM re-decides factor or model)
  → ...Next Round♾️... `,
    },
  },
  {
    name: "Data Science",
    icon: "Graph-Dot",
    color: "#a858ff",
    upload: false,
    developer: true,
    editLoop: true,
    loopRadio: "20",
    loopNumber: 20,
    hourRadio: "24",
    hourNumber: 24,
    introduce: {
      Introduction: `R&D-Agent automates Kaggle feature engineering, model tuning, and iterative development to help participants improve their performance in data science competitions.`,
      "Data Description": `R&D-Agent works with various datasets from Kaggle competitions, focusing on tasks such as regression, classification, and others using structured and unstructured data.
  In this scenario, it involves predicting forest cover type using cartographic variables determined from USFS and US Geological Survey data.`,
      "Evaluation Method": `The models and features are evaluated based on their performance on a test set or Kaggle Leaderboard, with the aim of achieving the highest possible leaderboard score.
  In this scenario, the solution should enhance the accuracy of forest cover type identification.`,
      "Scenario Breakdown": `... Round♾️ N:
  	→ [🔍Research to generate hypothesis] → (hypothesis)
  	→ [🔍Design Experiment, e.g. feature engineering, model tuning] → (Experiment Tasks)
  	→ [🛠️Experiment Implementation] → (Iterative implementation in workspace)
  	→ [📝Evaluation and Analysis] → (Feedback)
  → ...Next Round♾️...`,
    },
    child: kaggleCompetitions,
  },
];

const visibleContinuousScenarioList = continuousScenarioList.filter(
  (scenario) => scenario.name !== "Data Science"
);

const guidedScenarioList = [
  {
    name: "Finance Data Building (Reports)",
    id: "",
    icon: "Piggy-Bank",
    color: "#475dff",
    upload: true,
    developer: true,
    editLoop: false,
    loopRadio: "10",
    loopNumber: 10,
    hourRadio: "24",
    hourNumber: 24,
    introduce: {
      Introduction: `Applying R&D-Agent on finance data like a copilot to automatically extract knowledge from research reports on well-known financial factors, then implements and evaluates them to improve quantitative trading strategies. The scenario is built on Qlib.`,
      "Data Description": `The dataset includes daily stock data from the CSI300 index, with training data from 2008-2014, validation data from 2015-2016, and test data from 2017-2020. `,
      "Evaluation Method": `The performance of new financial factors is assessed through quantitative backtesting using Qlib. This process evaluates both the prediction accuracy and the final profit. `,
      "Scenario Breakdown": `... Round♾️ N:
	→ [🔍Research to extract well-known financial factors] → (Experiment Tasks) 
	→ [🛠️Experiment Implementation] → (Iterative Implementation in workspace) 
	→ [📝Evaluation and Analysis] → (Feedback) 
→ ...Next Round♾️... `,
    },
  },
  {
    name: "General Model Implementation",
    id: "",
    icon: "Web-Streamline",
    color: "#844bff",
    upload: true,
    developer: false,
    editLoop: true,
    loopRadio: "-1",
    loopNumber: 1,
    hourRadio: "24",
    hourNumber: 24,
    introduce: {
      Introduction: `Apply R&D-Agent as a copilot to automate the extraction, implementation, and iterative refinement of models from academic papers, enabling the efficient reproduction of state-of-the-art AI techniques.`,
      "Example PDF reports": `- [2210.09789](https://arxiv.org/pdf/2210.09789)
- [2305.10498](https://arxiv.org/pdf/2305.10498)
- [2110.14446](https://arxiv.org/pdf/2110.14446)
- [2205.12454](https://arxiv.org/pdf/2205.12454)
- [2210.16518](https://arxiv.org/pdf/2210.16518)`,
      "Data Description": `The system supports various data types including tabular, time-series, and graph data, facilitating diverse applications across AI research. `,
      "Evaluation Method": `The extracted models are validated through back-testing and iterative refinement to ensure functionality, correctness, and alignment with source material specifications. `,
      "Scenario Breakdown": `[🔍Paper Reader] → (Experiment Tasks containing model structure)  
→ [🛠️Experiment Implementation] → (Iterative implementation in PyTorch code) `,
    },
  },
];

const scenarioList = ref(visibleContinuousScenarioList);
const scenarioCheckedIndex = ref(0);
const scenarioChecked = ref(visibleContinuousScenarioList[0]);
const introName = ref(Object.keys(visibleContinuousScenarioList[0].introduce));
const editLoop = ref(visibleContinuousScenarioList[0].editLoop);
const developer = ref(visibleContinuousScenarioList[0].developer);
const id = ref("");
const line = ref(null);
const tabIndex = ref(0);

const historyScenarioList = ref([]);
const historyScenarioCheckedIndex = ref(-1);
const historyScenarioChecked = ref(null);
const historyTraceList = ref([]);
const historyTraceCheckedIndex = ref(-1);
const historyTraceChecked = ref(null);
const selectedFiles = ref([]);

const selectLastHistoryTrace = (scenario, scenarioIndex = -1) => {
  const traceList = Array.isArray(scenario?.children) ? scenario.children : [];
  const lastTraceIndex = traceList.length - 1;

  historyScenarioChecked.value = scenario || null;
  historyScenarioCheckedIndex.value = scenarioIndex;
  historyTraceList.value = traceList;
  historyTraceCheckedIndex.value = lastTraceIndex;
  historyTraceChecked.value = lastTraceIndex >= 0 ? traceList[lastTraceIndex] : null;
};

const getScenarioConfigByName = (name) => {
  const normalizedName = String(name || "").trim();

  if (!normalizedName) {
    return null;
  }

  return (
    visibleContinuousScenarioList.find(
      (scenario) => scenario.name === normalizedName
    ) ||
    guidedScenarioList.find((scenario) => scenario.name === normalizedName) ||
    null
  );
};

const applyScenarioConfig = (scenario) => {
  if (!scenario) {
    return;
  }

  const scenarioNameToApply = String(scenario.name || "").trim();
  const matchedContinuousScenario = visibleContinuousScenarioList.find(
    (item) => item.name === scenarioNameToApply
  );
  const matchedGuidedScenario = guidedScenarioList.find(
    (item) => item.name === scenarioNameToApply
  );
  const resolvedScenario = matchedContinuousScenario || matchedGuidedScenario || scenario;
  const isContinuousScenario = Boolean(matchedContinuousScenario);

  tabIndex.value = isContinuousScenario ? 0 : 1;
  scenarioList.value = isContinuousScenario
    ? visibleContinuousScenarioList
    : guidedScenarioList;
  scenarioCheckedIndex.value = scenarioList.value.findIndex(
    (item) => item.name === resolvedScenario.name
  );
  scenarioChecked.value = resolvedScenario;
  introName.value = Object.keys(resolvedScenario.introduce);
  editLoop.value = resolvedScenario.editLoop;
  scenarioName.value = resolvedScenario.name;
  developer.value = resolvedScenario.developer;
  loopRadio.value = resolvedScenario.loopRadio;
  hourRadio.value = resolvedScenario.hourRadio;
  num.value = loopRadio.value == "-1" ? resolvedScenario.loopNumber : 1;
  num1.value = hourRadio.value == "-1" ? resolvedScenario.hourNumber : 1;
  loopNumber.value = resolvedScenario.loopNumber;
  hourNumber.value = resolvedScenario.hourNumber;
};

const historyScenarioCheckedItem = (data) => {
  selectLastHistoryTrace(data.scenarioChecked, data.scenarioCheckedIndex);
};

const historyTraceCheckedItem = (data) => {
  historyTraceCheckedIndex.value = data.scenarioCheckedIndex;
  historyTraceChecked.value = data.scenarioChecked;
};

const scenarioCheckedItem = (data) => {
  scenarioCheckedIndex.value = data.scenarioCheckedIndex;
  applyScenarioConfig(data.scenarioChecked);
  // id.value = scenarioChecked.value.id; // 新场景id id由后端传入不需要
  num.value = 1;
  num1.value = 1;
  if (loopRadio.value == "-1") {
    num.value = scenarioChecked.value.loopNumber;
  }
  if (hourRadio.value == "-1") {
    num1.value = scenarioChecked.value.hourNumber;
  }
  loopNumber.value = scenarioChecked.value.loopNumber;
  hourNumber.value = scenarioChecked.value.hourNumber;
  uploaDone.value = false;
  selectedFiles.value = [];
  id.value = "";
  syncLoopCountWithSelectedFiles();
};
const changeFile = (file, fileList) => {
  const nameSet = new Set();
  const uniqueFiles = [];
  const duplicateNames = [];
  const invalidFiles = [];
  const allowedExtensionsText = getAllowedUploadExtensions().join(", ");

  fileList.forEach((item) => {
    const normalizedName = (item.name || "").trim().toLowerCase();
    if (!normalizedName) {
      uniqueFiles.push(item);
      return;
    }
    if (!isAllowedUploadFile(item)) {
      invalidFiles.push(item.name);
      return;
    }
    if (nameSet.has(normalizedName)) {
      duplicateNames.push(item.name);
      return;
    }
    nameSet.add(normalizedName);
    uniqueFiles.push(item);
  });

  if (duplicateNames.length) {
    const duplicateText = [...new Set(duplicateNames)].join(", ");
    ElMessage.warning(`Duplicate file name is not allowed: ${duplicateText}`);
  }

  if (invalidFiles.length) {
    const invalidText = [...new Set(invalidFiles)].join(", ");
    ElMessage.warning(
      `Unsupported file type: ${invalidText}. Allowed formats: ${allowedExtensionsText}`
    );
  }

  selectedFiles.value = uniqueFiles;
  id.value = "";
  syncLoopCountWithSelectedFiles();
};
const removeSelectedFile = (uid) => {
  selectedFiles.value = selectedFiles.value.filter((item) => item.uid !== uid);
  id.value = "";
  syncLoopCountWithSelectedFiles();
};

const createScenarioFormData = () => {
  const formData = new FormData();
  const resolvedLoopNumber = shouldMatchLoopCountToUploads()
    ? selectedFiles.value.length
    : loopNumber.value;

  if (scenarioChecked.value) {
    formData.append("scenario", scenarioChecked.value.name);
  }
  selectedFiles.value.forEach((file) => {
    formData.append("files", file.raw || file);
  });
  formData.append("competition", "");
  formData.append("competition", scenarioChecked.value.checkedName || "");
  formData.append("loops", resolvedLoopNumber);
  formData.append("all_duration", hourNumber.value);

  return formData;
};

const submitScenarioUpload = (formData) => {
  loading.value = true;
  uploadFile(formData)
    .then((response) => {
      loading.value = false;
      id.value = response.id;
      uploaDone.value = true;
      showPlayground.value = true;
    })
    .catch(() => {
      loading.value = false;
    });
};

const Back = () => {
  showPanel.value = 1;
  showPlayground.value = false;
  scenarioCheckedIndex.value = -1;
  scenarioChecked.value = null;
  uploaDone.value = false;
  selectedFiles.value = [];
  id.value = "";
};

function getCompletedIdList() {
  const data = localStorage.getItem(completedTraceStorageKey);
  if (!data) {
    return [];
  }

  try {
    return JSON.parse(data);
  } catch {
    return [];
  }
}

function appendTraceId(groupedTraceMap, traceId) {
  const normalizedTraceId = String(traceId || "").trim();

  if (!normalizedTraceId) {
    return;
  }

  const separatorIndex = normalizedTraceId.indexOf("/");
  const scenario =
    separatorIndex === -1
      ? normalizedTraceId
      : normalizedTraceId.slice(0, separatorIndex);
  const traceName =
    separatorIndex === -1
      ? normalizedTraceId
      : normalizedTraceId.slice(separatorIndex + 1);

  if (!groupedTraceMap.has(scenario)) {
    groupedTraceMap.set(scenario, new Map());
  }

  groupedTraceMap.get(scenario).set(traceName, {
    name: traceName,
    id: normalizedTraceId,
  });
}

async function buildHistoryTraceList() {
  const groupedTraceMap = new Map();
  const completedIdList = getCompletedIdList();
  const backendTraceIds = await getHistoryTraceIds().then((response) =>
    Array.isArray(response) ? response : []
  ).catch(() => []);

  const traceIdList = [...new Set([...completedIdList, ...backendTraceIds])];
  traceIdList.forEach((traceId) => appendTraceId(groupedTraceMap, traceId));

  historyScenarioList.value = Array.from(groupedTraceMap.entries()).map(
    ([scenario, traceMap]) => ({
      name: scenario,
      children: Array.from(traceMap.values()),
    })
  );

  const lastCompletedTraceId = String(
    completedIdList[completedIdList.length - 1] || ""
  ).trim();
  const separatorIndex = lastCompletedTraceId.indexOf("/");
  const lastScenarioName =
    separatorIndex === -1 ? "" : lastCompletedTraceId.slice(0, separatorIndex);
  const defaultScenarioIndex = historyScenarioList.value.findIndex(
    (scenario) => scenario.name === lastScenarioName
  );
  const defaultScenario =
    defaultScenarioIndex >= 0
      ? historyScenarioList.value[defaultScenarioIndex]
      : historyScenarioList.value[historyScenarioList.value.length - 1] || null;

  selectLastHistoryTrace(defaultScenario, defaultScenarioIndex);
}

const generate = () => {
  uploaDone.value = false;
  if (id.value) {
    showPlayground.value = true;
    return;
  }
  if (scenarioChecked.value && scenarioChecked.value.upload) {
    if (!selectedFiles.value.length) {
      return;
    }
    submitScenarioUpload(createScenarioFormData());
    return;
  }
  if (scenarioChecked.value && !scenarioChecked.value.upload) {
    submitScenarioUpload(createScenarioFormData());
  }
};

const viewTracePage = () => {
  if (!historyTraceChecked.value) {
    return;
  }

  const traceId = String(historyTraceChecked.value.id || "").trim();
  const separatorIndex = traceId.indexOf("/");
  const scenarioNameFromTrace =
    historyScenarioChecked.value?.name ||
    (separatorIndex === -1 ? "" : traceId.slice(0, separatorIndex));
  const matchedScenario = getScenarioConfigByName(scenarioNameFromTrace);

  applyScenarioConfig(matchedScenario);

  id.value = historyTraceChecked.value.id;
  showPlayground.value = true;
};

const openHistoryPanel = async () => {
  await buildHistoryTraceList();
  showPanel.value = 3;
  showPlayground.value = false;
};

const tabChange = (index, flag) => {
  moveSlider(index);
  tabIndex.value = index;
  if (index == 0) {
    scenarioList.value = visibleContinuousScenarioList;
  } else {
    scenarioList.value = guidedScenarioList;
  }
  if (flag) {
    scenarioCheckedIndex.value = -1;
    scenarioChecked.value = null;
  } else {
    applyScenarioConfig(scenarioList.value[0]);
    // id.value = scenarioChecked.value.id; // 新场景id
    num.value = 1;
    num1.value = 1;
    if (loopRadio.value == "-1") {
      num.value = scenarioChecked.value.loopNumber;
    }
    if (hourRadio.value == "-1") {
      num1.value = scenarioChecked.value.hourNumber;
    }
    loopNumber.value = scenarioChecked.value.loopNumber;
    hourNumber.value = scenarioChecked.value.hourNumber;
    uploaDone.value = false;
    selectedFiles.value = [];
    id.value = "";
    syncLoopCountWithSelectedFiles();
  }
};

const changePanel = () => {
  showPanel.value = 2;
  showPlayground.value = false;
  selectedFiles.value = [];
  id.value = "";

  nextTick(() => {
    moveSlider(0);
    tabChange(0);
  });
};

function moveSlider(index) {
  const lines = line.value;
  lines.style.left = `${12 * index + 0.75 * (2 * index + 1)}em`; // 更新下划线位置
}

onMounted(() => {
  void buildHistoryTraceList();
});
</script>

<style scoped lang="scss">
.page-content {
  position: relative;
  width: 100%;
  height: 100%;
  display: flex;
  flex-direction: column;
  overflow: hidden;
  .nav-bar {
    padding: 1.05em 1.8em;
    box-sizing: border-box;
    position: fixed;
    z-index: 100;
    top: 1.2em;
    right: 2.4em;
    display: flex;
    gap: 0.67em;
    flex-wrap: nowrap;
    justify-content: flex-end;
    align-items: center;
    flex-direction: row;
    color: #868ca5;
    background: #fff;
    border-radius: 999px;
    box-shadow: 0 12px 32px rgba(17, 24, 39, 0.08);
    span {
      font-size: 1.125em;
      line-height: 200%;
      color: #868ca5;
      cursor: pointer;
      &.active {
        font-weight: 600;
        color: var(--text-color);
        &:hover {
          color: var(--text-color);
        }
      }
      &:hover {
        color: #c5d2e6;
      }
    }

    .nav-highlight-name {
      max-width: 18em;
      cursor: default;
      font-size: 1.125em;
      font-weight: 700;
      line-height: 200%;
      text-shadow: 8px 11px 30px var(--wg-shadow-color);
      background: linear-gradient(90deg, #2667ff 0%, #9d41ff 100%);
      background-clip: text;
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;

      &:hover {
        color: inherit;
      }
    }

    .nav-separator {
      cursor: default;

      &:hover {
        color: #868ca5;
      }
    }
  }
}

.setup-content {
  flex: 1;
  min-height: 0;
  overflow: auto;
}

.playground-shell {
  flex: 1;
  min-height: 0;
  overflow: hidden;
}
.main-content {
  width: 100%;
  max-width: 1560px;
  margin: 0 auto;
  padding-left: 1rem;
  padding-right: 1rem;
  box-sizing: border-box;
  padding: 4em 0 5em;
  padding: 3.6em 0 4.5em;
  padding: 3.6em 0 0;
  &.split-two {
    display: flex;
    justify-content: space-around;
    align-items: stretch;
    padding: 2em 0 2em;
    padding: 1.8em 0 2em;
    box-sizing: border-box;
    .h1 {
      font-size: 2em;
      font-size: 1.8em;
    }
    .select-upload {
      width: 50%;
      box-sizing: border-box;
      border-right: 2px solid;
      border-image-source: linear-gradient(
        to bottom,
        rgba(38, 103, 255, 0.2),
        rgba(157, 65, 255, 0.2)
      );
      border-image-slice: 30;
      overflow: visible;
    }

    .main-panel {
      width: 39em;
      margin: 2.5em auto 0;
      margin: 2.25em auto 0;
      .title {
        font-size: 1.5em;
        font-size: 1.35em;
        padding-left: 10px;
      }
      .desc {
        font-size: 1.2em;
        font-size: 1.08em;
        padding-left: 10px;
      }
      p {
        font-size: 1.2em;
        font-size: 1.08em;
      }
      .select-box {
        margin-top: 1.5em;
        margin-bottom: 3em;
        margin-top: 1.35em;
        margin-bottom: 2.7em;
      }
      .loop-content {
        margin-top: 1.5em;
        margin-top: 1.35em;
        .radio-box {
          margin-top: 1.5em;
          margin-top: 1.35em;
          padding-left: 10px;
        }
      }
    }

    .intro-txt {
      overflow: visible;
      padding-right: 1.2em;
    }
  }
  &.no-upload {
    .main-panel {
      margin-top: 4em;
      margin-top: 3.6em;
    }
  }
  .select-upload {
    padding: 0 6em;
    padding: 0 5.4em;
  }
  .intro-txt {
    padding: 0 7em 0;
    padding: 0 6.3em 0;
    width: 50%;
    box-sizing: border-box;

    h3 {
      color: var(--text-color);
      font-size: 1.25em;
      font-size: 1.125em;
      font-weight: 700;
      line-height: 200%;
      margin-bottom: 0.5em;
      margin-bottom: 0.45em;
    }
    p {
      color: var(--text-color);
      font-size: 1.125em;
      font-size: 1.0125em;
      line-height: 200%;
      margin-bottom: 1.5em;
      margin-bottom: 1.35em;
      white-space: break-spaces;
    }

    .intro-markdown {
      margin-bottom: 1.35em;

      :deep(.markdown-body) {
        color: var(--text-color);
        font-size: 1.0125em;
        line-height: 200%;
        white-space: break-spaces;
      }

      :deep(p),
      :deep(ul) {
        margin: 0;
      }

      :deep(ul) {
        padding-left: 1.4em;
      }

      :deep(a) {
        color: #2667ff;
        text-decoration: underline;
      }
    }
  }
  h1 {
    color: var(--text-color);
    text-align: center;
    font-size: 2.5em;
    font-size: 2.25em;
    font-weight: 700;
    line-height: 200%;
  }
  .h1 {
    line-height: 120%;
    font-size: 2.1875em;
    font-size: 1.96875em;
  }
  .card-box {
    display: flex;
    justify-content: center;
    margin-top: 4em;
    margin-top: 3.6em;
    padding-bottom: 7em;
    padding-bottom: 6.3em;
    gap: 6em;
    gap: 5.4em;

    .card-item {
      display: flex;
      padding: 2.5em 4.875em 0px 4.875em;
      padding: 2.25em 4.3875em 0 4.3875em;
      flex-direction: column;
      justify-content: flex-end;
      align-items: center;
      gap: 0.875em;
      gap: 0.7875em;
      background: var(--bg-white);
      cursor: pointer;

      --border-width: 2px;
      --border-radius: 2.5em;

      h2 {
        text-align: center;
        text-shadow: 8px 11px 30px var(--wg-shadow-color);
        font-family: "Microsoft YaHei";
        font-size: 2em;
        font-size: 1.8em;
        font-weight: 700;
        background: linear-gradient(90deg, #4c5cff 0%, #794dff 100%);
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
      }
      p {
        width: 18em;
        height: 2.42em;
        height: 2.178em;
        color: var(--text-color);
        text-align: center;
        text-shadow: 8px 11px 30px var(--wg-shadow-color);
        font-size: 1.5em;
        font-size: 1.35em;
        font-style: normal;
        font-weight: 700;
        line-height: 120%;
        margin: 0.89em 0 0.5em;
        margin: 0.8em 0 0.45em;
      }
      img {
        height: 20em;
        height: 18em;
        transition: transform 0.5s ease; /* 平滑的过渡效果 */
        transform-origin: center 0%;
      }
      .img2 {
        transform-origin: center center;
      }
      &:hover {
        background: var(--card-bg-hover-color);
        .img1 {
          transform: scale(1.3);
          transform-origin: center 0%;
        }
        .img2 {
          transform: scale(1.3) rotate(-10deg);
          transform-origin: center center;
        }
      }
    }
  }
  .main-panel {
    width: 40em;
    margin: 4em auto 0;
    margin: 3.6em auto 0;

    &.history-panel {
      width: 52em;
      max-width: min(52em, calc(100vw - 4rem));
    }

    .history-select-row {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 1.5em;
      margin-top: 1.62em;

      .history-select-item {
        min-width: 0;

        .title {
          padding-left: 10px;
        }

        :deep(.select-box) {
          margin-top: 1.2em;
          margin-bottom: 0;
        }
      }
    }

    .title {
      color: var(--text-color);
      text-shadow: 8px 11px 30px var(--wg-shadow-color);
      font-size: 1.68em;
      font-size: 1.512em;
      font-weight: 700;
      padding-left: 20px;

      &.with-tip {
        display: flex;
        align-items: center;
        gap: 0.45em;
      }

      .tip-icon {
        width: 1.25em;
        height: 1.25em;
        border-radius: 50%;
        border: 1px solid var(--card-border-color);
        color: var(--text-color);
        display: inline-flex;
        align-items: center;
        justify-content: center;
        font-size: 0.62em;
        line-height: 1;
        cursor: pointer;
      }
    }

    .small-config-title {
      font-size: 1.44em;
      font-size: 1.296em;
    }

    .desc {
      color: var(--text-color);
      font-size: 1.25em;
      font-size: 1.125em;
      line-height: 120%;
      padding-left: 20px;
      margin-top: 0.5em;
      margin-top: 0.45em;
    }
    p {
      color: var(--text-color);
      font-size: 1.3em;
      font-size: 1.17em;
      margin-top: 0.4em;
      margin-top: 0.36em;
    }
    .select-box {
      margin-top: 1.8em;
      margin-top: 1.62em;
      margin-bottom: 2.8em;
      margin-bottom: 2.52em;
      position: relative;
      .select-div {
        display: flex;
        height: 3.75em;
        height: 3.375em;
        justify-content: space-between;
        align-items: center;
        border-radius: 999px;
        --border-radius: 999px;
        --border-width: 2px;
        cursor: pointer;
        .down-arrow {
          width: 1.5em;
          height: 1.5em;
          width: 1.35em;
          height: 1.35em;
          background: url(/src/assets/images/down-arrow.svg) no-repeat;
          background-size: contain;
          position: absolute;
          right: 1.5em;
          right: 1.35em;
        }
        .checked-item {
          padding: 0.625em 2.2em 0.625em;
          padding: 0.5625em 1.98em 0.5625em;
          display: flex;
          align-items: center;
          .select-item-icon {
            margin-right: 1em;
            margin-right: 0.9em;
          }
          span {
            color: var(--text-color);
            font-size: 1.5625em;
            font-size: 1.40625em;
            font-size: 1.3em;
            font-size: 1.17em;
            line-height: 200%;
            margin-top: -2px;
          }
        }
      }
      .select-drop-panel {
        width: 100%;
        height: 18.25em;
        height: 16.425em;
        position: absolute;
        left: 0;
        top: 3.75em;
        top: 3.375em;
        cursor: pointer;
        background-color: var(--bg-white);
        border-radius: 40px;
        z-index: 99;
        overflow: hidden;
        box-shadow: 8px 11px 30px 0px var(--wg-shadow-color);
        .select-drop-list {
          width: calc(100% - 4px);
          height: calc(16.425em - 4px);
          position: absolute;
          left: 2px;
          top: 2px;
          z-index: 1;
          background-color: var(--bg-white);
          border-radius: 40px;
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
          padding: 0.625em 2.2em 0.625em;
          padding: 0.5625em 1.98em 0.5625em;
          border-bottom: 2px solid #2e65ff;
          display: flex;
          align-items: center;

          &:last-child {
            border-bottom: none;
          }
          .select-item-icon {
            margin-right: 1em;
            margin-right: 0.9em;
          }
          span {
            color: var(--text-color);
            // font-size: 1.5625em;
            font-size: 1.3em;
            font-size: 1.17em;
            line-height: 200%;
            margin-top: -2px;
          }
          &:hover,
          &.active {
            background-color: var(--card-bg-hover-color);
          }
        }
      }
    }
    .upload-box {
      width: 100%;
      position: relative;
      z-index: 1;
      text-align: center;
      background: url(@/assets/images/small-bg.png) no-repeat;
      background-size: 100% 100%;
      border-radius: 40px;
      cursor: pointer;
      .upload-box-bg {
        width: 100%;
        padding: 1.08em 0.5em 2.07em;
        box-sizing: border-box;
        border-radius: 40px;
        .upload-small {
          display: inline-block;
          width: 3em;
          height: 3.75em;
          width: 2.7em;
          height: 3.375em;
          background: url(@/assets/images/upload.svg) no-repeat;
          background-size: contain;
        }
      }
      .upload-progress-bg {
        width: 100%;
        height: 13.8em;
        height: 12.42em;
        box-sizing: border-box;
        border-radius: 40px;
        text-align: left;
        position: relative;
        overflow: hidden;
        padding: 2px;
      }
      &.upload-file {
        // padding: 4em 0 6em;
        background: url(@/assets/images/big-bg.png) no-repeat;
        background-size: 100% 100%;
        text-align: center;

        padding: 3.6em 0 5.4em;
        box-sizing: border-box;
        .upload-big {
          display: inline-block;
          width: 3.75em;
          height: 4.5em;
          width: 3.375em;
          height: 4.05em;
          background: url(@/assets/images/file-upload.svg) no-repeat;
          background-size: contain;
        }
        &:hover {
          background: url(@/assets/images/big-bg-active.png) no-repeat;
          background-size: 100% 100%;

          .upload-big {
            background: url(@/assets/images/file-upload-active.svg) no-repeat;
            background-size: contain;
          }
        }
        .upload-box-bg {
          width: 100%;
          // padding: 3.6em 0 5.4em;
          border-radius: 40px;
        }
        .upload-progress-bg {
          height: 20.25em;
          height: 18.225em;
        }
      }
      &:hover {
        background: url(@/assets/images/small-bg-active.png) no-repeat;
        background-size: 100% 100%;
        .upload-small {
          background: url(@/assets/images/upload-active.svg) no-repeat;
          background-size: contain;
        }
      }
      h3 {
        color: var(--text-color);
        font-size: 1.3em;
        font-size: 1.17em;
        font-weight: 700;
        line-height: 200%;
        margin-top: 1em;
        margin-top: 0.9em;
      }
      p {
        color: var(--text-color);
        text-align: center;
        font-size: 1em;
        font-size: 0.9em;
      }
    }
    .file-tag-list {
      margin-top: 1.2em;
      margin-bottom: 0.4em;
      display: flex;
      flex-wrap: wrap;
      gap: 0.6em;
      padding: 0 0.25em;
      position: relative;
      z-index: 2;

      .file-tag {
        display: inline-flex;
        align-items: center;
        max-width: 100%;
        border-radius: 999px;
        padding: 0.36em 0.78em;
        background: var(--bg-white);
        color: var(--text-color);
        border: 1px solid var(--card-border-color);
        box-shadow: 8px 11px 30px 0px var(--wg-shadow-color);

        .tag-name {
          max-width: 24em;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
          font-size: 0.88em;
          line-height: 1.5;
        }

        .tag-close {
          margin-left: 0.5em;
          width: 1.05em;
          height: 1.05em;
          border-radius: 50%;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          cursor: pointer;
          font-size: 0.88em;
          line-height: 1;
          color: var(--text-color);
          background-color: var(--card-bg-hover-color);
        }
      }
    }
    .loop-content {
      margin-top: 1.8em;
      margin-top: 1.62em;

      .loop-upload {
        display: block;
        margin-top: 1.35em;
        margin-bottom: 0.45em;

        :deep(.el-upload) {
          display: block;
          width: 100%;
        }

        :deep(.el-upload-dragger) {
          margin-top: 0;
        }
      }

      .file-tag-list {
        margin-top: 1.2em;
        margin-bottom: 0.15em;
      }

      .radio-box {
        margin-top: 1.8em;
        margin-top: 1.62em;
        padding-left: 20px;
      }

      .compact-config-box {
        :deep(.compact-radio-group) {
          width: 100%;
          display: flex;
          flex-wrap: wrap;
          gap: 0.5em 1.1em;
          align-items: center;
        }

        :deep(.compact-radio-group .el-radio) {
          margin-right: 0;
          min-width: 0;
          display: inline-flex;
          align-items: center;
          padding: 0.2em 0.7em 0.2em 0;
          border-radius: 999px;
        }

        :deep(.compact-radio-group .el-radio:last-child) {
          display: inline-flex;
          align-items: center;
          gap: 0.45em;
        }

        :deep(.compact-radio-group .el-input-number) {
          width: 88px;
        }

        :deep(.compact-radio-group .el-radio__label) {
          padding-left: 0.35em;
        }
      }

      .compact-setting-title {
        font-size: 1.25em;
        font-size: 1.125em;
        flex-shrink: 0;
        min-width: 9.5em;
        padding-left: 10px;
        line-height: 1.6;
        margin: 0;
      }

      .compact-setting-box {
        margin-top: 0;
        padding-left: 0;

        :deep(.compact-radio-group) {
          justify-content: flex-end;
          gap: 0.35em 0.8em;
        }

        :deep(.compact-radio-group .el-radio) {
          padding: 0.1em 0.45em 0.1em 0;
          min-height: 1.6em;
        }

        :deep(.compact-radio-group .el-radio__label) {
          font-size: 0.92em;
          line-height: 1.6;
          padding-left: 0.25em;
        }

        :deep(.compact-radio-group .el-input-number) {
          width: 74px;
        }
      }

      .compact-setting-row {
        display: flex;
        align-items: baseline;
        justify-content: space-between;
        gap: 0.9em;
        margin-top: 0.08em;
        padding: 0.12em 0;
      }

      .compact-setting-row.is-second {
        margin-top: 0.28em;
        padding-top: 0;
      }
    }
    .btn-main {
      margin-top: 7.5em;
      margin-top: 6.75em;
      display: flex;
      justify-content: space-between;
      padding: 0 0.25em;
      padding: 0 0.225em;
      button {
        width: 12em;
        width: 10.8em;
        height: 3.78em;
        height: 3.4em;
        color: var(--text-color);
        font-size: 1.125em;
        font-size: 1.0125em;
        font-weight: 700;
        line-height: 150%;
        text-transform: uppercase;
        border: none;
        cursor: pointer;
        --border-radius: 999px;
        --border-width: 2px;
        &.disable {
          border-radius: 37.5px;
          background: #c4c4c4;
          box-shadow: 8px 11px 30px 0px var(--wg-shadow-color);
          color: var(--bg-white);
        }
        &.active {
          border-radius: 37.5px;
          background: linear-gradient(90deg, #2667ff 0%, #9d41ff 100%), #979797;
          box-shadow: 8px 11px 30px 0px var(--wg-shadow-color);
        }
        &.back:hover {
          background-color: var(--card-bg-hover-color);
        }
      }
    }
  }
  .nav-content {
    display: flex;
    justify-content: center;
    margin-top: 2em;
    margin-top: 1.8em;
    nav {
      display: flex;
      padding: 0 4em;
      padding: 0 3.6em;
      ul {
        display: flex;
        position: relative;

        li {
          margin: 0 0.75em;
          width: 12em;
          height: 3em;
          height: 2.7em;
          box-sizing: border-box;
          display: flex;
          align-items: center;
          justify-content: center;
          // text-transform: uppercase;
          cursor: pointer;
          -webkit-user-select: none;
          -moz-user-select: none;
          -ms-user-select: none;
          user-select: none;
          transition: 0.35s ease;
          color: var(--text-color);
          text-shadow: 8px 11px 30px var(--wg-shadow-color);
          font-size: 1.3em;
          font-size: 1.17em;
          line-height: 200%;

          &.active {
            font-weight: 700;
            transition: 0.35s ease;
            background: linear-gradient(90deg, #2667ff 0%, #9d41ff 100%);
            background-clip: text;
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
          }
        }
        .nav-line {
          width: 12em;
          height: 3px;
          position: absolute;
          left: 0;
          bottom: 0;
          background: linear-gradient(to right, #2667ff, #9d41ff);
          font-size: 1.3em;
          font-size: 1.17em;
          border-radius: 4px;
          transition: 0.35s ease;
        }
      }
    }
  }

  .nav-content + .main-panel {
    margin-top: 2.1em;
    margin-top: 1.9em;
  }
}
:deep(.el-upload-dragger) {
  padding: 0;
  margin-top: 1.5em;
  margin-top: 1.35em;
  background-color: transparent;
  border: none;
  border-radius: 40px;
  box-shadow: 8px 11px 30px 0px var(--wg-shadow-color);
}
:deep(.el-radio) {
  --el-radio-text-color: var(--text-color);
  --el-color-primary: var(--card-border-color);
}
:deep(.el-radio__label) {
  color: var(--text-color);
  font-family: "Segoe UI";
  font-size: 1.2em;
  font-size: 1.08em;
  font-weight: 700;
  line-height: 200%;
}
:deep(.el-radio__inner) {
  border-color: var(--text-color);
}
:deep(.el-input) {
  width: 80px;
}
:deep(.el-input-number) {
  width: 80px;
}

@media (max-width: 900px) {
  .main-content {
    .main-panel {
      &.history-panel {
        width: 100%;
        max-width: 100%;
      }

      .history-select-row {
        grid-template-columns: 1fr;
        gap: 1em;
      }
    }
  }
}
</style>
