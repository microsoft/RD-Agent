<template>
  <div class="result-component">
    <div class="download-btn">
      <div class="download-btn-item">
        <el-switch
          v-model="switchValue"
          @change="switchChange"
          style="--el-switch-on-color: #8749ff; --el-switch-off-color: #c9d0fc"
        />
        <span>Successful Hypotheses</span>
      </div>
      <div class="download-btn-item" @click="downloadLogs">
        <span class="download-icon"></span>
        <span>Log</span>
      </div>
      <div class="download-btn-item" @click="downloadAllLoops">
        <span class="download-icon"></span>
        <span>All loop files</span>
      </div>
    </div>
    <div class="bg-content">
      <div class="result-content">
        <h2>Metrics</h2>
        <div>
          <chartBox :metricData="metricData"></chartBox>
        </div>
        <div class="section-title-row">
          <h2>Summary</h2>
          <div class="trace-name-chip" v-if="traceName">{{ traceName }}</div>
        </div>
        <div class="table-box">
          <el-table
            :data="tableData"
            :border="parentBorder"
            style="width: 100%"
            cell-class-name="table-cell"
          >
            <el-table-column label="#" width="80">
              <template #header="scope">
                <span style="color: #000">#</span>
              </template>
              <template #default="scope">
                <span>{{ indexMethod(scope.row.num) }}</span>
              </template>
            </el-table-column>

            <el-table-column
              label="Component"
              width="200"
              prop="component"
              v-if="scenarioName == 'Data Science'"
            >
              <template #header="scope">
                <span class="text-color-blue">Component</span>
              </template>
            </el-table-column>
            <el-table-column label="Status" width="140">
              <template #header="scope">
                <span class="text-color-blue">Status</span>
              </template>
              <template #default="scope">
                <span v-if="scope.row.decision" class="success">Success</span>
                <span v-if="!scope.row.decision" class="fail">Failed</span>
              </template>
            </el-table-column>
            <el-table-column label="Hypothesis" prop="hypothesis">
              <template #header="scope">
                <span class="text-color-blue">Hypothesis</span>
              </template>
              <template #default="scope">
                {{ scope.row.hypothesis || "Component initializing" }}
              </template>
            </el-table-column>
            <el-table-column label="Feedback" prop="concise_knowledge">
              <template #header="scope">
                <span class="text-color-purple">Feedback</span>
              </template>
              <template #default="scope">
                {{
                  scope.row.reason ||
                  "No reason generated due to some errors happened in previous steps"
                }}
              </template>
            </el-table-column>
            <el-table-column label="Files" width="200">
              <template #header="scope">
                <span class="text-color-blue">Files</span>
              </template>
              <template #default="scope">
                <div class="download-file-list" v-if="scope.row.downloadFiles.length">
                  <button
                    class="download-file-btn download-all-btn"
                    type="button"
                    @click="downloadRowAllFiles(scope.row)"
                  >
                    download_all
                  </button>
                  <button
                    :class="[
                      'download-file-btn',
                      getDownloadFileClass(scope.row, file),
                    ]"
                    type="button"
                    v-for="(file, idx) in getDisplayFiles(scope.row)"
                    :key="scope.row.num + '-' + idx + '-' + file.name"
                    :title="file.name"
                    @click="downloadCodeFile(file)"
                  >
                    {{ file.name }}
                  </button>
                </div>
                <span v-else>-</span>
              </template>
            </el-table-column>
            <el-table-column type="expand" width="120">
              <template #default="props">
                <ul class="table-expand">
                  <li>
                    <div class="title">
                      <span class="Hypothesis-icon icon"></span>
                      <span class="name">Hypothesis</span>
                    </div>
                    <div class="text">
                      {{ props.row.hypothesis || "Component initializing" }}
                    </div>
                  </li>
                  <li>
                    <div class="title">
                      <span class="Reason-icon icon"></span>
                      <span class="name">Reason</span>
                    </div>
                    <div class="text">
                      {{ props.row.reason || "" }}
                    </div>
                  </li>
                  <li>
                    <div class="title">
                      <span class="Observation-icon icon"></span>
                      <span class="name">Observation</span>
                    </div>
                    <div class="text">
                      {{ props.row.observations || "" }}
                    </div>
                  </li>
                  <li>
                    <div class="title">
                      <span class="Conclusion-icon icon"></span>
                      <span class="name">Status</span>
                    </div>
                    <div class="text">
                      <span v-if="props.row.decision" class="success"
                        >Success</span
                      >
                      <span v-if="!props.row.decision" class="fail"
                        >Failed</span
                      >
                    </div>
                  </li>
                </ul>
              </template>
            </el-table-column>
          </el-table>
        </div>
      </div>
    </div>
  </div>
</template>
<script setup>
import { ref, watch, computed, defineProps, onMounted, nextTick } from "vue";
import { ElMessage } from "element-plus";
import JSZip from "jszip";
import chartBox from "../components/chartBox.vue";
import { getStdoutDownloadUrl } from "../utils/api";
const props = defineProps({
  currentData: Array,
  scenarioName: String,
  baseFactors: [Object, String],
  traceName: String,
});
const currentData = ref(props.currentData);
const scenarioName = ref(props.scenarioName);
const baseFactors = ref(props.baseFactors);
const traceName = ref(props.traceName);
const tableData = ref([]);
const switchValue = ref(false);
const metricData = ref(null);

const getTraceId = () => {
  const scenario = scenarioName.value == null ? "" : String(scenarioName.value).trim();
  const trace = traceName.value == null ? "" : String(traceName.value).trim();

  if (!scenario || !trace) {
    return "";
  }

  return `${scenario}/${trace}`;
};

const downloadLogs = async () => {
  const traceId = getTraceId();

  if (!traceId) {
    ElMessage.warning("Trace logs are not available yet.");
    return;
  }

  try {
    const response = await fetch(getStdoutDownloadUrl(traceId));
    if (!response.ok) {
      let errorMessage = "Failed to download logs.";
      try {
        const errorData = await response.json();
        if (errorData?.error) {
          errorMessage = errorData.error;
        }
      } catch {
        // Ignore JSON parsing failures and keep the fallback message.
      }
      throw new Error(errorMessage);
    }

    const blob = await response.blob();
    const contentDisposition = response.headers.get("content-disposition") || "";
    const fileNameMatch = contentDisposition.match(/filename\*?=(?:UTF-8''|\")?([^";]+)/i);
    const fileName = fileNameMatch?.[1] || `${traceName.value || "rdagent"}.log`;

    const link = document.createElement("a");
    const objectUrl = URL.createObjectURL(blob);
    link.href = objectUrl;
    link.download = decodeURIComponent(fileName);
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(objectUrl);
  } catch (error) {
    ElMessage.error(error?.message || "Failed to download logs.");
  }
};

const normalizeTaskFileName = (taskName) => {
  const text = taskName == null ? "task" : String(taskName).trim();
  const safeName = (text || "task").replace(/[\\/:*?"<>|]/g, "_");
  return `${safeName}.py`;
};

const toTextContent = (value) => {
  if (value == null) {
    return "";
  }
  if (typeof value === "string") {
    return value;
  }
  try {
    return JSON.stringify(value, null, 2);
  } catch {
    return String(value);
  }
};

const pickWorkspaceContent = (workspace, preferredName) => {
  if (!workspace || typeof workspace !== "object") {
    return "";
  }
  if (Object.prototype.hasOwnProperty.call(workspace, preferredName)) {
    return toTextContent(workspace[preferredName]);
  }
  const keys = Object.keys(workspace);
  if (!keys.length) {
    return "";
  }
  const pyKey = keys.find((key) => key.endsWith(".py"));
  const selectedKey = pyKey || keys[0];
  return toTextContent(workspace[selectedKey]);
};

const resolveBaseFactorsContent = () => {
  if (baseFactors.value == null) {
    return "";
  }
  if (typeof baseFactors.value === "string") {
    return baseFactors.value;
  }
  if (typeof baseFactors.value === "object") {
    if (!Object.keys(baseFactors.value).length) {
      return "";
    }
    return JSON.stringify(baseFactors.value, null, 2);
  }
  return toTextContent(baseFactors.value);
};

const buildTaskDescriptionsMarkdown = (loopItem) => {
  const tasks = Array.isArray(loopItem?.researcTasks) ? loopItem.researcTasks : [];
  const lines = ["# Task Descriptions", ""];

  if (!tasks.length) {
    lines.push("No task descriptions available.");
    return lines.join("\n");
  }

  tasks.forEach((task, index) => {
    const taskName = task?.name == null ? `Task ${index + 1}` : String(task.name).trim();
    const descriptionText =
      task?.description == null || String(task.description).trim() === ""
        ? "No description provided."
        : String(task.description);
    lines.push(`## ${index + 1}. ${taskName || `Task ${index + 1}`}`);
    lines.push("");
    lines.push(descriptionText);
    lines.push("");
  });

  return lines.join("\n").trim();
};

const normalizeDecision = (value) => {
  if (value === true || value === false) {
    return value;
  }
  if (value == null) {
    return null;
  }
  if (typeof value === "string") {
    const normalized = value.trim().toLowerCase();
    if (normalized === "true") {
      return true;
    }
    if (normalized === "false") {
      return false;
    }
  }
  return Boolean(value);
};

const getEntryDecision = (feedbackEntry) => {
  if (!feedbackEntry || typeof feedbackEntry !== "object") {
    return null;
  }
  if (Object.prototype.hasOwnProperty.call(feedbackEntry, "final_decision")) {
    return normalizeDecision(feedbackEntry.final_decision);
  }
  if (Object.prototype.hasOwnProperty.call(feedbackEntry, "decision")) {
    return normalizeDecision(feedbackEntry.decision);
  }
  return null;
};

const getLastEvoEntries = (loopItem) => {
  const evolvingCodes = Array.isArray(loopItem?.evolvingCodes)
    ? loopItem.evolvingCodes
    : [];
  const evolvingFeedbacks = Array.isArray(loopItem?.evolvingFeedbacks)
    ? loopItem.evolvingFeedbacks
    : [];
  const mergedEntries = [];

  evolvingCodes.forEach((codeItem, codeIndex) => {
    const codeEntries = Array.isArray(codeItem?.content) ? codeItem.content : [];
    const feedbackEntries = Array.isArray(evolvingFeedbacks[codeIndex]?.content)
      ? evolvingFeedbacks[codeIndex].content
      : [];

    codeEntries.forEach((entry, entryIndex) => {
      if (!entry || entry.evo_id == null) {
        return;
      }
      mergedEntries.push({
        ...entry,
        taskDecision: getEntryDecision(feedbackEntries[entryIndex]),
      });
    });
  });

  if (!mergedEntries.length) {
    return [];
  }

  const targetEvoId = mergedEntries[mergedEntries.length - 1].evo_id;
  return mergedEntries.filter((entry) => entry.evo_id === targetEvoId);
};

const downloadCodeFile = (file) => {
  if (!file || !file.name) {
    return;
  }
  const content = toTextContent(file.content);
  const mimeType = file.name.endsWith(".json")
    ? "application/json;charset=utf-8"
    : "text/x-python;charset=utf-8";
  const blob = new Blob([content], { type: mimeType });
  const link = document.createElement("a");
  const objectUrl = URL.createObjectURL(blob);
  link.href = objectUrl;
  link.download = file.name;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(objectUrl);
};

const downloadRowAllFiles = async (row) => {
  const files = row?.downloadFiles || [];
  if (!files.length) {
    return;
  }

  const zip = new JSZip();
  files.forEach((file) => {
    if (!file?.name) {
      return;
    }
    zip.file(file.name, toTextContent(file.content));
  });

  const zipBlob = await zip.generateAsync({ type: "blob" });
  const link = document.createElement("a");
  const objectUrl = URL.createObjectURL(zipBlob);
  link.href = objectUrl;
  link.download = `loop_${indexMethod(row.num)}_files.zip`;
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(objectUrl);
};

const getLoopLastEvoFiles = (loopItem) => {
  const lastEvoEntries = getLastEvoEntries(loopItem);
  const descriptionsMdContent = buildTaskDescriptionsMarkdown(loopItem);

  const fileMap = new Map();

  if (lastEvoEntries.length) {
    lastEvoEntries.forEach((entry) => {
      const fileName = normalizeTaskFileName(entry.target_task_name);
      const content = pickWorkspaceContent(entry.workspace, fileName);
      const status =
        entry.taskDecision === null
          ? "unknown"
          : entry.taskDecision
            ? "success"
            : "fail";
      fileMap.set(fileName, {
        name: fileName,
        content,
        status,
      });
    });

    const baseFactorsContent = resolveBaseFactorsContent();
    if (baseFactorsContent) {
      fileMap.set("base_factors.json", {
        name: "base_factors.json",
        content: baseFactorsContent,
        status: "unknown",
      });
    }

    fileMap.set("descriptions.md", {
      name: "descriptions.md",
      content: descriptionsMdContent,
      status: "unknown",
    });

    return Array.from(fileMap.values());
  }

  const baseFactorsContent = resolveBaseFactorsContent();
  if (baseFactorsContent) {
    fileMap.set("base_factors.json", {
      name: "base_factors.json",
      content: baseFactorsContent,
      status: "unknown",
    });
  }

  fileMap.set("descriptions.md", {
    name: "descriptions.md",
    content: descriptionsMdContent,
    status: "unknown",
  });

  return Array.from(fileMap.values());
};

const getFileStatus = (row, file) => {
  if (file?.status === "success" || file?.status === "fail") {
    return file.status;
  }
  const rowDecision = normalizeDecision(row?.decision);
  if (rowDecision === true) {
    return "success";
  }
  if (rowDecision === false) {
    return "fail";
  }
  return "unknown";
};

const getDownloadFileClass = (row, file) => {
  const isPinnedBlueFile =
    file?.name === "base_factors.json" || file?.name === "descriptions.md";
  if (isPinnedBlueFile) {
    return {
      "base-factor-file-btn": true,
    };
  }
  const status = getFileStatus(row, file);
  return {
    "base-factor-file-btn": false,
    "download-file-success": status === "success",
    "download-file-fail": status === "fail",
  };
};

const getDisplayFiles = (row) => {
  const files = row?.downloadFiles || [];
  const baseFactorFile = files.find((file) => file?.name === "base_factors.json");
  const descriptionsFile = files.find((file) => file?.name === "descriptions.md");
  const otherFiles = files.filter(
    (file) => file?.name !== "base_factors.json" && file?.name !== "descriptions.md"
  );

  if (baseFactorFile && descriptionsFile) {
    return [baseFactorFile, descriptionsFile, ...otherFiles];
  }
  if (baseFactorFile) {
    return [baseFactorFile, ...otherFiles];
  }
  if (descriptionsFile) {
    return [descriptionsFile, ...otherFiles];
  }
  return otherFiles;
};

const downloadAllLoops = async () => {
  if (!currentData.value || !currentData.value.length) {
    return;
  }

  const zip = new JSZip();
  let hasFile = false;

  currentData.value.forEach((loopItem, index) => {
    const files = getLoopLastEvoFiles(loopItem);
    if (!files.length) {
      return;
    }
    const folderName = `loop_${indexMethod(index)}`;
    const loopFolder = zip.folder(folderName);
    if (!loopFolder) {
      return;
    }
    files.forEach((file) => {
      loopFolder.file(file.name, toTextContent(file.content));
      hasFile = true;
    });
  });

  if (!hasFile) {
    return;
  }

  const zipBlob = await zip.generateAsync({ type: "blob" });
  const link = document.createElement("a");
  const objectUrl = URL.createObjectURL(zipBlob);
  link.href = objectUrl;
  link.download = "all_loops_files.zip";
  document.body.appendChild(link);
  link.click();
  document.body.removeChild(link);
  URL.revokeObjectURL(objectUrl);
};

const updateData = () => {
  const table = [];
  const metric = {};

  if (switchValue.value) {
    currentData.value.forEach((item, index) => {
      const tableItem = {};
      if (item.researchHypothesis) {
        tableItem.num = index;
        tableItem.hypothesis = item.researchHypothesis.hypothesis || "";
        tableItem.component = item.researchHypothesis.component || "";
        tableItem.downloadFiles = getLoopLastEvoFiles(item);
        if (item.feedbackHypothesis) {
          tableItem.reason = item.feedbackHypothesis.reason || "";
          tableItem.observations = item.feedbackHypothesis.observations || "";
          tableItem.decision = item.feedbackHypothesis.decision || false;
        }
        if (tableItem.decision) {
          table.push(tableItem);
          if (item.feedbackMetric) {
            Object.keys(item.feedbackMetric).forEach((metr) => {
              if (!metric[metr]) {
                metric[metr] = [
                  {
                    name: "Round" + (index + 1),
                    value: item.feedbackMetric[metr],
                    desc: item.researchHypothesis.hypothesis,
                  },
                ];
              } else {
                metric[metr].push({
                  name: "Round" + (index + 1),
                  value: item.feedbackMetric[metr],
                  desc: item.researchHypothesis.hypothesis,
                });
              }
            });
          }
        }
      }
    });
  } else {
    currentData.value.forEach((item, index) => {
      const tableItem = {};
      if (item.researchHypothesis) {
        tableItem.num = index;
        tableItem.hypothesis = item.researchHypothesis.hypothesis || "";
        tableItem.component = item.researchHypothesis.component || "";
        tableItem.downloadFiles = getLoopLastEvoFiles(item);
        if (item.feedbackHypothesis) {
          tableItem.reason = item.feedbackHypothesis.reason || "";
          tableItem.observations = item.feedbackHypothesis.observations || "";
          tableItem.decision = item.feedbackHypothesis.decision || false;
        }
        table.push(tableItem);
      }
      if (item.feedbackMetric) {
        Object.keys(item.feedbackMetric).forEach((metr) => {
          if (!metric[metr]) {
            metric[metr] = [
              {
                name: "Round" + (index + 1),
                value: item.feedbackMetric[metr],
                desc: item.researchHypothesis.hypothesis,
              },
            ];
          } else {
            metric[metr].push({
              name: "Round" + (index + 1),
              value: item.feedbackMetric[metr],
              desc: item.researchHypothesis.hypothesis,
            });
          }
        });
      }
    });
  }
  tableData.value = table;
  metricData.value = metric;
};

watch(
  () => [props.currentData, props.scenarioName, props.baseFactors],
  (newValue, oldValue) => {
    currentData.value = newValue[0];
    scenarioName.value = newValue[1];
    baseFactors.value = newValue[2];
    updateData();
  },
  {
    deep: true,
    immediate: true,
  }
);

// table
const parentBorder = ref(false);
const childBorder = ref(false);
const indexMethod = (index) => {
  return String(index + 1).padStart(2, "0");
};

const handleRowClick = (row, column, event) => {
  // 切换当前行的展开状态
  row.isExpanded = !row.isExpanded;
  if (this.expands.includes(row.id)) {
    this.expands = this.expands.filter((item) => item !== row.id);
  } else {
    this.expands = [row.id];
  }
};

const switchChange = () => {
  updateData();
};

onMounted(() => {
  if (currentData.value) {
    updateData();
  }
});

watch(
  () => props.traceName,
  (newValue) => {
    traceName.value = newValue;
  }
);
</script>

<style scoped lang="scss">
.result-component {
  height: 100%;
  .download-btn {
    display: flex;
    justify-content: flex-end;
    align-items: center;
    gap: 1.8em;
    padding: 0.45em 1.8em;
    .download-btn-item {
      display: flex;
      align-items: center;
      gap: 0.45em;
      .download-icon {
        display: inline-block;
        width: 1.35em;
        height: 1.35em;
        background: url(@/assets/playground-images/download.svg) no-repeat;
        background-size: contain;
      }
      span {
        color: #3f3f3f;
        font-size: 0.9em;
        font-weight: 700;
        line-height: 200%;
      }
    }
  }
  .bg-content {
    width: 100%;
    height: calc(100vh - 13.95em);
    box-sizing: border-box;
    padding: 1.35em 1.8em;
    justify-content: center;
    align-items: center;
    border-radius: 20px;
    background: var(--bg-white-blue-color);
    overflow: auto;
    &::-webkit-scrollbar-thumb {
      background-color: #fff;
    }
    &:hover {
      &::-webkit-scrollbar-thumb {
        background-color: #e4e7ff;
      }
    }

    .result-content {
      .section-title-row {
        display: flex;
        align-items: center;
        gap: 0.75em;
        margin-bottom: 0.45em;

        .trace-name-chip {
          display: inline-flex;
          align-items: center;
          max-width: min(32em, calc(100% - 8em));
          padding: 0.2em 0.8em;
          border-radius: 999px;
          background: linear-gradient(90deg, #edf4ff 0%, #f5efff 100%);
          border: 1px solid #d7e1ff;
          color: #4a5576;
          font-size: 0.9em;
          font-weight: 700;
          line-height: 1.6;
          white-space: nowrap;
          overflow: hidden;
          text-overflow: ellipsis;
        }
      }

      h2 {
        font-size: 1.26em;
        font-weight: 700;
        line-height: 200%;
        margin-bottom: 0.45em;
      }

      .section-title-row h2 {
        margin-bottom: 0;
        flex-shrink: 0;
      }

      .table-box {
        --el-border-color-lighter: #c5d2e6;
        --el-fill-color-light: #f6f6f6;
        border: 1px solid #c5d2e6;
        border-radius: 20px;
        overflow: hidden;

        .text-color-blue {
          background: linear-gradient(271deg, #3062ff 2.3%, #589aff 96.87%);
          background-clip: text;
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }
        .text-color-purple {
          background: linear-gradient(271deg, #7426ff 2.3%, #423cff 96.87%);
          background-clip: text;
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
        }
        .success {
          display: inline-block;
          padding: 0.27em 1.35em;
          border-radius: 999px;
          border: 1px solid #16a427;
          background: #dbf4de;
          color: #16a427;
          font-size: 12px;
        }
        .fail {
          display: inline-block;
          padding: 0.27em 1.35em;
          border-radius: 999px;
          border: 1px solid #e4452c;
          background: #ffe6e3;
          color: #e4452c;
          font-size: 12px;
        }
        .add-icon {
          display: inline-block;
          width: 1.35em;
          height: 1.35em;
          background: url(@/assets/playground-images/add.svg) no-repeat;
          background-size: contain;
          cursor: pointer;
        }

        .download-file-list {
          display: flex;
          flex-wrap: wrap;
          gap: 0.45em;
          justify-content: flex-start;
          width: 100%;
        }

        .download-file-btn {
          display: inline-flex;
          align-items: center;
          justify-content: flex-start;
          max-width: 18em;
          padding: 0.2em 0.6em;
          border: 1px solid #c5d2e6;
          border-radius: 999px;
          font-size: 0.9em;
          line-height: 1.6;
          background: #f6f8ff;
          color: #3f3f3f;
          overflow: hidden;
          text-overflow: ellipsis;
          white-space: nowrap;
          cursor: pointer;
        }

        .download-file-btn:hover {
          background: #edf2ff;
        }

        .download-all-btn {
          border-color: #8749ff;
          background: #f2ecff;
          color: #5f2bd9;
          font-weight: 700;
        }

        .download-all-btn:hover {
          background: #e8dcff;
        }

        .base-factor-file-btn {
          border-color: #3062ff;
          background: #edf2ff;
          color: #3062ff;
        }

        .base-factor-file-btn:hover {
          background: #e4e7ff;
        }

        .download-file-success {
          border-color: #16a427;
          background: #dbf4de;
          color: #16a427;
        }

        .download-file-success:hover {
          background: #cdeed2;
        }

        .download-file-fail {
          border-color: #e4452c;
          background: #ffe6e3;
          color: #e4452c;
        }

        .download-file-fail:hover {
          background: #ffd9d4;
        }

        .table-expand {
          background: #f6f6f6;
          li {
            display: flex;
            padding: 1.35em 0;
            justify-content: flex-start;
            align-items: center;
            border-bottom: 1px solid #c5d2e6;
            &:last-child {
              border: none;
            }
            .title {
              display: flex;
              width: 16.2em;
              flex-direction: column;
              align-items: center;
              justify-content: center;
              .icon {
                display: inline-block;
                width: 1.6875em;
                height: 1.6875em;
                margin-bottom: 0.72em;
              }
              .Hypothesis-icon {
                background: url(@/assets/playground-images/Hypothesis-expand.svg)
                  no-repeat;
                background-size: contain;
              }
              .Reason-icon {
                background: url(@/assets/playground-images/Reason-expand.svg)
                  no-repeat;
                background-size: contain;
              }
              .Observation-icon {
                background: url(@/assets/playground-images/Observation-expand.svg)
                  no-repeat;
                background-size: contain;
              }
              .Conclusion-icon {
                background: url(@/assets/playground-images/Conclusion-expand.svg)
                  no-repeat;
                background-size: contain;
              }
              .name {
                color: #000;
                font-size: 1.0125em;
                font-weight: 700;
                line-height: 150%;
                text-transform: uppercase;
              }
            }
            .text {
              color: #000;
              font-family: "Microsoft YaHei";
              font-size: 1.0125em;
              line-height: 180%; /* 32.4px */
              padding: 0 4.5em 0 2.7em;
              flex: 1;
            }
          }
        }
      }
    }
  }
}
:deep(.el-table thead th.el-table__cell) {
  background-color: var(--bg-white-blue-color);
}
:deep(.el-table .el-table__cell) {
  padding: 0;
}
:deep(.el-table .cell) {
  display: flex;
  align-items: center;
  justify-content: center;
  padding: 1.35em 1.8em;
  color: #000;
  font-family: "Microsoft YaHei";
  font-size: 1.0125em;
  line-height: 180%; /* 32.4px */
}
:deep(.el-table tbody .indexClass) {
  color: #000;
  font-size: 1.35em;
  line-height: 200%;
}
:deep(.el-table__expand-icon > .el-icon) {
  display: none; /* 隐藏原生图标 */
}
:deep(.el-table__expand-icon) {
  height: auto;
}
:deep(.el-table__row .el-table__expand-icon:before) {
  content: "\002B";
  color: blue;
  font-size: 49px;
  font-family: "Segoe UI";
}

:deep(.el-table__row .el-table__expand-icon--expanded:before) {
  content: "\002D";
  color: blue;
  font-size: 57px;
  font-family: "Segoe UI";
}
:deep(.el-table__expand-icon--expanded) {
  transform: none;
}
</style>
