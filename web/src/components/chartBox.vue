<template>
  <div class="chart-box">
    <div
      class="chart-item"
      v-for="(item, index) in keyList"
      :key="item"
      :style="{ width: 100 / keyList.length + '%' }"
    >
      <div
        class="zoom"
        @click="zoom(colors[index], metricData[item], item)"
      ></div>
      <lineChart
        :color="colors[index]"
        :data="metricData[item]"
        :chartName="item"
        :smallSize="true"
      ></lineChart>
    </div>
    <div class="dialog-box" v-if="showDialog">
      <div class="dialog-content gradient-border">
        <div class="close" @click="close"></div>
        <lineChart
          :color="dialogColor"
          :data="dialogData"
          :chartName="dialogName"
          :smallSize="false"
        ></lineChart>
      </div>
    </div>
  </div>
</template>

<script setup>
import { onMounted, defineProps, watch, ref } from "vue";
import lineChart from "../components/lineChartOne.vue";

const props = defineProps({
  metricData: Object,
});
const metricData = ref(props.metricData);
const colors = ["red", "blue", "orange", "green"];
const keyList = ref([]);
const showDialog = ref(false);
const updateData = () => {
  keyList.value = Object.keys(metricData.value);
};
const dialogColor = ref("");
const dialogData = ref(null);
const dialogName = ref("");
const zoom = (color, data, name) => {
  dialogColor.value = color;
  dialogData.value = data;
  showDialog.value = true;
  dialogName.value = name;
};
const close = () => {
  showDialog.value = false;
  dialogColor.value = "";
  dialogData.value = null;
  dialogName.value = "";
};

watch(
  () => props.metricData,
  (newValue, oldValue) => {
    metricData.value = newValue;
    updateData();
  },
  {
    deep: true,
    immediate: true,
  }
);

onMounted(() => {
  updateData();
});
</script>

<style scoped lang="scss">
.chart-box {
  display: flex;
  gap: 1.8em;
  margin-bottom: 1.8em;
  .chart-item {
    background-color: var(--bg-white);
    max-width: 500px;
    min-width: 0;
    border-radius: 35.5px;
    position: relative;
    box-shadow: 1px 1px 2px 0px rgba(255, 255, 255, 0.3) inset,
      -1px -1px 2px 0px rgba(221, 221, 221, 0.5) inset,
      -10px 10px 20px 0px rgba(221, 221, 221, 0.2),
      10px -10px 20px 0px rgba(221, 221, 221, 0.2),
      -10px -10px 20px 0px rgba(255, 255, 255, 0.9),
      10px 10px 25px 0px rgba(221, 221, 221, 0.9);
    .zoom {
      position: absolute;
      right: 1.125em;
      top: 0.8em;
      width: 1.125em;
      height: 1.125em;
      background: url(@/assets/playground-images/zoom.svg) no-repeat;
      background-size: contain;
      cursor: pointer;
      z-index: 1;
      &:hover {
        opacity: 0.5;
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
      width: 60%;
      height: 498px;
      background-color: #fff;
      border-radius: 18px;
      --border-radius: 20px;
      --border-width: 2px;
      // padding: 3em 4em;
      padding-bottom: 2em;
      margin-top: -4em;
      position: relative;
      .close {
        position: absolute;
        right: 1.35em;
        top: 0.9em;
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
</style>
