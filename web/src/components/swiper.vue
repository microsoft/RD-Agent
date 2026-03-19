<template>
  <swiper
    :modules="modules"
    :navigation="true"
    :pagination="{ clickable: true }"
  >
    <swiper-slide
      class="swiper-slide"
      v-for="(item, index) in caseData"
      :key="item.model + index"
    >
      <div class="swiper-main">
        <p class="title">
          <span>Care/Harm Score：</span>{{ item.score.toFixed(4) }}
        </p>
        <p class="title">{{ "Case" + (index + 1) + "-" + item.label }} Score</p>
        <div class="chart-content-desc">
          <img src="@/assets/images/Avatar-Q.png" alt="Q" />
          <p>
            {{ item.prompt }}
          </p>
        </div>
        <div class="chart-content-desc">
          <img src="@/assets/images/Avatar-A.png" alt="A" />
          <p class="highlight" v-html="item.highlight"></p>
        </div>
      </div>
    </swiper-slide>
  </swiper>
</template>
<script>
import { ref, watch } from "vue";
// import Swiper core and required modules
import { Pagination, Navigation, A11y, Autoplay } from "swiper/modules";

// Import Swiper Vue.js components
import { Swiper, SwiperSlide } from "swiper/vue";

// Import Swiper styles
import "swiper/css";
import "swiper/css/navigation";
import "swiper/css/pagination";

// Import Swiper styles
export default {
  components: {
    Swiper,
    SwiperSlide,
  },
  props: {
    data: {
      type: Array,
      required: true,
    },
  },
  mounted() {
    console.log("接收到的消息:", this.data); // 打印接收到的消息
  },
  setup(props) {
    const onSwiper = (swiper) => {
      console.log(swiper);
    };
    const onSlideChange = () => {
      console.log("slide change");
    };
    const caseData = ref(props.data);
    watch(
      () => props.data,
      (newMessage, oldMessage) => {
        caseData.value = newMessage;
        console.log(caseData.value);
      }
    );
    return {
      onSwiper,
      onSlideChange,
      modules: [Pagination, Navigation, A11y, Autoplay],
      caseData,
    };
  },
};
</script>
<style scoped lang="scss">
.swiper-slide {
  width: 100%;
  padding-bottom: 3.25em;
  box-sizing: border-box;
  padding: 2.5em 0 2.5em 0;
  .swiper-main {
    padding: 0 4em;
    height: 21em;
    overflow: auto;
  }
  .title {
    font-size: 1em;
    color: #fff;
    line-height: 2em;
    span {
      color: #ffd000;
    }
  }
  .chart-content-desc {
    display: flex;
    margin-top: 1.625em;
    img {
      display: block;
      width: 1.875em;
      height: 1.875em;
      margin-right: 1.625em;
    }
    p {
      width: 31.43em;
      font-size: 0.875em;
      color: #fff;
      line-height: 1.8em;
    }
  }
}

:deep(.swiper-pagination-bullet) {
  width: 1em;
  height: 3px;
  background-color: #fff;
  opacity: 0.3;
  border-radius: 1px;
}
:deep(.swiper-pagination-bullet-active) {
  width: 1.25em;
  opacity: 1;
}
:deep(.swiper-button-prev),
:deep(.swiper-button-next) {
  width: 22px;
  height: 22px;
  color: #fff;
}
:deep(.swiper-button-prev:after),
:deep(.swiper-button-next:after) {
  font-size: 1em;
}
:deep(.highlight i) {
  color: #90e0ef;
  font-weight: 500;
  line-height: 1.8em;
}
</style>
