<template>
  <div ref="chart" style="width: 100%; height: 200px"></div>
</template>

<script setup>
import { onMounted, onBeforeUnmount, defineProps, watch, ref, nextTick } from "vue";
import * as echarts from "echarts";

const props = defineProps({
  color: String,
  data: Object,
  chartName: String,
  smallSize: Boolean,
});
const color = ref(props.color);
const data = ref(props.data);
const chartName = ref(props.chartName);
const smallSize = ref(props.smallSize);
const chart = ref(null);

let chartInstance = null;
let resizeObserver = null;
let textMeasureCanvas = null;

const getTextMeasureContext = () => {
  if (typeof document === "undefined") {
    return null;
  }
  if (!textMeasureCanvas) {
    textMeasureCanvas = document.createElement("canvas");
  }
  return textMeasureCanvas.getContext("2d");
};

const getAxisLabelLayout = (xLabels = []) => {
  const labels = Array.isArray(xLabels) ? xLabels : [];
  const width = chart.value?.clientWidth || 0;
  const count = labels.length;
  if (!width || count <= 1) {
    return {
      rotate: 0,
      bottom: smallSize.value ? "20%" : "18%",
    };
  }

  const fontSize = smallSize.value ? 11 : 12;
  const fontFamily =
    chart.value && typeof window !== "undefined"
      ? window.getComputedStyle(chart.value).fontFamily || "sans-serif"
      : "sans-serif";
  const measureContext = getTextMeasureContext();
  if (measureContext) {
    measureContext.font = `${fontSize}px ${fontFamily}`;
  }

  const maxLabelWidth = labels.reduce((max, label) => {
    const text = String(label == null ? "" : label);
    const measured = measureContext
      ? measureContext.measureText(text).width
      : text.length * fontSize * 0.62;
    return Math.max(max, measured);
  }, 0);

  // Keep this aligned with grid left/right (10% + 5%).
  const plotWidth = width * 0.85;
  const slotWidth = plotWidth / count;
  const minGap = 8;
  const useVertical = maxLabelWidth + minGap > slotWidth;

  return {
    rotate: useVertical ? -90 : 0,
    bottom: useVertical ? (smallSize.value ? "32%" : "30%") : smallSize.value ? "20%" : "18%",
  };
};

const updateContainerSize = () => {
  if (!chart.value) return;
  const width = chart.value.offsetWidth;
  // When mounted under v-show/display:none, width can be 0.
  if (!width) return;
  chart.value.style.height = width / 2 + "px";
};

const canInitChart = () => {
  if (!chart.value) return false;
  // Prefer client sizes (what ECharts checks internally)
  const w = chart.value.clientWidth;
  const h = chart.value.clientHeight;
  return w > 0 && h > 0;
};

const ensureChartInitialized = () => {
  if (!chart.value) return;
  if (chartInstance) return;

  updateContainerSize();
  if (!canInitChart()) {
    // Still hidden / size not ready; defer.
    return;
  }

  // Avoid double-init if something else already initialized it.
  chartInstance = echarts.getInstanceByDom(chart.value) || echarts.init(chart.value);
};

const updatData = () => {
  const tooltip = {};

  const ydatas = {};
  let minValue = Infinity;
  let maxValue = -Infinity;
  const series = [];
  const legend = [];
  let flag = false;
  (data.value || []).forEach((item) => {
    if (!item || item.name == null) return;
    tooltip[item.name] = item;

    const itemValue = item.value;
    // Single-series numeric chart: value can be number OR null/undefined.
    if (typeof itemValue === "number" || itemValue == null) {
      flag = true;
      const yKey = chartName.value;
      ydatas[yKey] = ydatas[yKey] || [];
      ydatas[yKey].push([item.name, itemValue ?? null]);
      if (Number.isFinite(itemValue)) {
        maxValue = Math.max(maxValue, itemValue);
        minValue = Math.min(minValue, itemValue);
      }
      return;
    }

    // Multi-series chart: value should be an object; each key can still be null.
    if (typeof itemValue === "object") {
      Object.keys(itemValue).forEach((yKey) => {
        const yVal = itemValue[yKey];
        if (!ydatas[yKey]) {
          ydatas[yKey] = [[item.name, yVal ?? null]];
        } else {
          ydatas[yKey].push([item.name, yVal ?? null]);
        }
        if (Number.isFinite(yVal)) {
          maxValue = Math.max(maxValue, yVal);
          minValue = Math.min(minValue, yVal);
        }
      });
    }
  });
  const keys = Object.keys(ydatas);
  let index = keys.indexOf("ensemble");
  if (index !== -1) {
    // 移除找到的元素
    keys.splice(index, 1);
    // 将元素添加到数组的第二个位置
    keys.unshift("ensemble");
  }
  keys.forEach((item) => {
    if (!flag) {
      legend.push(item);
    }
    series.push({
      name: item,
      data: ydatas[item],
      type: "line",
      symbol: "circle",
      symbolSize: smallSize.value ? 6 : 10,
    });
  });

  const xLabels = (data.value || [])
    .map((item) => item?.name)
    .filter((name) => name != null);
  const axisLayout = getAxisLabelLayout(xLabels);

  const option = {
    color: [
      "#5470c6",
      "#73c0de",
      "#ee6666",
      "#91cc75",
      "#fac858",
      "#3ba272",
      "#fc8452",
      "#9a60b4",
      "#ea7ccc",
    ],
    title: {
      text: chartName.value,
      textStyle: {
        fontSize: smallSize.value ? 12 : 18,
      },
      padding: [10, 20],
      top: smallSize.value ? "2%" : "5%",
      left: smallSize.value ? "1%" : "6%",
    },
    grid: {
      top: "20%",
      bottom: axisLayout.bottom,
      left: "10%",
      right: "5%",
    },
    legend: {
      show: !smallSize.value,
      data: legend,
      right: "8%",
      top: "5%",
    },
    tooltip: {
      trigger: "axis", // 可选项：'item'，'axis'，'none'
      appendToBody: true,
      position: function (point, params, dom, rect, size) {
        // point：鼠标位置，[x, y]
        // params：tooltip 的数据
        // dom：tooltip 的 DOM 元素
        // rect：坐标轴的位置等信息
        // size：图表的尺寸信息

        return [point[0], point[1]];
      },
      // confine: true,
      formatter: function (params) {
        if (!params || params.length === 0) return "";

        const axisValue = params[0]?.axisValue;
        let tooltipContent = `<div><div><strong>${axisValue} Value: <br>`;
        params.forEach((item) => {
          const v = Array.isArray(item.value) ? item.value[1] : item.value;
          tooltipContent += `<span style="color: blue">${
            item.seriesName + ": " + (v ?? "null")
          }</span><br>
       `;
        });

        const desc = tooltip?.[axisValue]?.desc ?? "";
        tooltipContent += ` </strong></div>`;
        if (desc) {
          tooltipContent += `<p style="margin-top: 10px;">${desc}</p>`;
        }
        tooltipContent += `</div>`;
        return tooltipContent;
      },
      extraCssText:
        "max-width: 400px; white-space: normal; word-wrap: break-word;",
    },
    xAxis: {
      type: "category",
      axisLabel: {
        show: true,
        rotate: axisLayout.rotate,
        interval: 0,
        margin: 10,
      },
    },
    yAxis: (() => {
      const yAxis = {
        type: "value",
        axisLabel: {
          formatter: function (value, index) {
            if (value % 1 === 0) {
              return value.toFixed(0); // 没有小数部分时，显示整数
            } else if ((value * 10) % 1 === 0) {
              return value.toFixed(1); // 如果小数部分只有一位，则保留一位小数
            } else {
              return value.toFixed(2); // 其他情况保留两位小数
            }
          },
        },
      };

      // Only set explicit bounds when there is at least one finite value.
      if (Number.isFinite(minValue) && Number.isFinite(maxValue)) {
        yAxis.min = Math.floor(minValue * 1000) / 1000;
        yAxis.max = Math.ceil(maxValue * 1000) / 1000;
      }
      return yAxis;
    })(),
    series: series,
  };

  if (chartInstance) {
    chartInstance.setOption(option);
  }
};

watch(
  () => props.data,
  (newValue, oldValue) => {
    data.value = newValue;
    ensureChartInitialized();
    if (chartInstance) {
      updatData();
      chartInstance.resize();
    }
  },
  {
    deep: true,
    immediate: true,
  }
);

onMounted(() => {
  nextTick(() => {
    ensureChartInitialized();
    if (chartInstance) {
      updatData();
      chartInstance.resize();
    }

    // Ensure charts render when the container becomes visible / resizes.
    if (typeof ResizeObserver !== "undefined") {
      resizeObserver = new ResizeObserver(() => {
        updateContainerSize();
        ensureChartInitialized();
        if (chartInstance) {
          updatData();
          chartInstance.resize();
        }
      });
      resizeObserver.observe(chart.value);
    }
  });
});

onBeforeUnmount(() => {
  if (resizeObserver && chart.value) {
    resizeObserver.unobserve(chart.value);
  }
  resizeObserver = null;
  if (chartInstance) {
    chartInstance.dispose();
    chartInstance = null;
  }
});
</script>

<style>
/* 样式内容 */
</style>
