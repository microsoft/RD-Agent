<template>
  <div>
    <div id="capture" ref="capture">
      <!-- 这里是你想要保存为图片的HTML内容 -->
      <h1>Hello World</h1>
    </div>
    <button @click="saveAsImage">保存为图片</button>
  </div>
</template>

<script>
import { ref } from "vue";
import html2canvas from "html2canvas";

export default {
  setup() {
    const capture = ref(null);

    const saveAsImage = async () => {
      try {
        const canvas = await html2canvas(capture.value);
        const img = canvas.toDataURL("image/png");

        const link = document.createElement("a");
        link.href = img;
        link.download = "capture.png";
        link.click();
      } catch (error) {
        console.error("Error capturing the image:", error);
      }
    };

    return {
      capture,
      saveAsImage,
    };
  },
};
</script>
