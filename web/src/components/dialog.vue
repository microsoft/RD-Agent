<template>
  <div class="dialog-box" v-if="uniShowDialog">
    <div class="dialog-content gradient-border">
      <h1>Increase Loop Count</h1>
      <p>
        You can increase the number of loops. Please enter the desired number
        below.
      </p>
      <el-radio-group v-model="radio1">
        <el-radio value="5">5 Loops</el-radio>
        <el-radio value="10">10 Loops</el-radio>
        <el-radio value="20">20 Loops</el-radio>
        <el-radio value="num"
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
      <div class="btn-box">
        <button class="gradient-border back" @click="close">BACK</button>
        <button class="add-loops active">Add Loops</button>
      </div>
    </div>
  </div>
</template>
<script setup>
import { ref, watch, onMounted, defineProps, defineEmits, nextTick } from "vue";
const props = defineProps({
  showDialog: Number,
});
const uniShowDialog = ref(false);
const radio1 = ref("");
const num = ref();
const emit = defineEmits(["addLoop"]);

watch(
  () => props.showDialog,
  (newValue, oldValue) => {
    uniShowDialog.value = newValue > 0 ? true : false;
  }
);

const handleChange = (value) => {
  console.log(value);
};
const close = () => {
  uniShowDialog.value = false;
};
onMounted(() => {});
</script>

<style scoped lang="scss">
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
    background-color: #fff;
    border-radius: 18px;
    --border-radius: 20px;
    --border-width: 2px;
    padding: 3em 4em;
    margin-top: -4em;
    h1 {
      color: var(--text-color);
      text-shadow: 8px 11px 30px #edf0ff;
      font-size: 1.5em;
      font-weight: 700;
      line-height: 200%;
    }
    p {
      color: var(--text-color);
      font-size: 1.2em;
      line-height: 150%;
      margin: 1.25em 0;
    }
    .number-input {
      width: 80px;
      height: 40px;
      border-radius: 4px;
      border: 2px solid #c5d2e6;
      margin-right: 0.5em;
    }
    .btn-box {
      display: flex;
      justify-content: space-between;
      padding: 0 0.25em;
      position: relative;
      z-index: 1;
      margin-top: 4em;
      button {
        width: 12em;
        height: 3.78em;
        color: var(--text-color);
        font-size: 1em;
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
          color: #fff;
        }
        &.back:hover {
          background-color: var(--card-bg-hover-color);
        }
      }
    }
  }
}
:deep(.el-radio) {
  --el-radio-text-color: var(--text-color);
}
:deep(.el-radio__label) {
  font-size: 16px;
}
:deep(.el-radio__inner) {
  border-color: var(--text-color);
}
</style>
