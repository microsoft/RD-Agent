<template>
  <div ref="chart" style="width: 100%; height: 600px"></div>
</template>

<script setup>
import { onMounted, ref } from "vue";
import * as echarts from "echarts";

const chart = ref(null);

onMounted(() => {
  let base = +new Date(1968, 9, 3);
  let oneDay = 24 * 3600 * 1000;
  let dateList = [];
  for (let i = 1; i < 20000; i++) {
    var now = new Date((base += oneDay));
    dateList.push(
      [now.getFullYear(), now.getMonth() + 1, now.getDate()].join("/")
    );
  }
  function getData() {
    let data = [Math.random() * 300];
    for (let i = 1; i < 20000; i++) {
      data.push(Math.round((Math.random() - 0.5) * 20 + data[i - 1]));
    }
    return data;
  }

  const option = {
    tooltip: {
      trigger: "axis",
    },
    toolbox: {
      feature: {
        dataZoom: {
          yAxisIndex: "none",
        },
        restore: {},
        saveAsImage: {},
      },
    },
    xAxis: [
      {
        data: dateList,
        show: false,
      },
      {
        data: dateList,
        gridIndex: 1,
        show: false,
      },
      {
        data: dateList,
        gridIndex: 2,
        show: false,
      },
      {
        data: dateList,
        gridIndex: 3,
      },
    ],
    yAxis: [
      {},
      {
        gridIndex: 1,
      },
      {
        gridIndex: 2,
      },
      {
        gridIndex: 3,
      },
    ],
    grid: [
      {
        bottom: "75%",
        top: "5%",
      },
      {
        top: "25%",
        bottom: "50%",
      },
      {
        bottom: "25%",
        top: "50%",
      },
      {
        top: "75%",
      },
    ],
    series: [
      {
        type: "line",
        showSymbol: false,
        data: getData(),
      },
      {
        type: "line",
        showSymbol: false,
        data: getData(),
        xAxisIndex: 1,
        yAxisIndex: 1,
      },

      {
        type: "line",
        showSymbol: false,
        data: getData(),
        xAxisIndex: 2,
        yAxisIndex: 2,
      },

      {
        type: "line",
        showSymbol: false,
        data: getData(),
        xAxisIndex: 3,
        yAxisIndex: 3,
      },
    ],
  };

  const chartInstance = echarts.init(chart.value);
  chartInstance.setOption(option);
});
</script>

<style>
/* 样式内容 */
</style>
