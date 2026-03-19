<template>
  <div
    class="upload"
    :class="{ finished: displayProgress >= 100 }"
    :style="{ '--percent': displayProgress }"
  >
    <div class="text">
      <strong><span>{{ displayProgress >= 100 ? "Uploaded" : "Uploading" }}</span> files</strong>
      <div>
        <small>%</small>
        <div>
          <small>
            <span>{{ secondsLeftDisplay }}</span> seconds left
          </small>
        </div>
      </div>
    </div>
    <nav>
      <ul>
        <li>
          <a href="javascript:;" class="btn cancel" @click="cancelUpload"></a>
        </li>
      </ul>
    </nav>
    <div class="percent">
      <span></span>
      <div>
        <svg preserveAspectRatio="none" viewBox="0 0 600 12">
          <path
            d="M0,1 L200,1 C300,1 300,11 400,11 L600,11"
            stroke="currentColor"
            fill="none"
          ></path>
        </svg>
      </div>
    </div>
  </div>
</template>
<script setup>
import { computed, defineProps, defineEmits, ref, watch } from "vue";
const props = defineProps({
  progress: {
    type: Number,
    default: 0,
  },
});
const emit = defineEmits(["cancelUpload"]);
const displayProgress = ref(0);
const startedAt = ref(0);
const secondsLeft = ref(0);

const secondsLeftDisplay = computed(() => {
  if (displayProgress.value >= 100) {
    return 0;
  }
  return Math.max(0, secondsLeft.value);
});

watch(
  () => props.progress,
  (value) => {
    const safeValue = Number.isFinite(value) ? value : 0;
    const next = Math.max(0, Math.min(100, Math.floor(safeValue)));
    if (displayProgress.value === 0 && next > 0) {
      startedAt.value = Date.now();
    }
    displayProgress.value = next;

    if (next <= 0 || !startedAt.value) {
      secondsLeft.value = 0;
      return;
    }
    if (next >= 100) {
      secondsLeft.value = 0;
      return;
    }

    const elapsedMs = Date.now() - startedAt.value;
    const estimatedTotalMs = (elapsedMs * 100) / next;
    const remainMs = Math.max(0, estimatedTotalMs - elapsedMs);
    secondsLeft.value = Math.ceil(remainMs / 1000);
  },
  { immediate: true }
);

const cancelUpload = () => {
  emit("cancelUpload", true);
};
</script>

<style scoped lang="scss">
.upload {
  --percent: 0;
  counter-increment: percent var(--percent);
  background: #fff;
  border-radius: 40px;
  width: 100%;
  height: 100%;
  box-shadow: 0 4px 16px -1px rgba(18, 22, 33, 0.05);
  display: flex;
  align-items: center;
  position: relative;
  overflow: hidden;
  padding: 0 3em 2.3em;
  box-sizing: border-box;
  font-family: Roboto, Arial;
  //Safari fix
  -webkit-mask-image: -webkit-radial-gradient(white, black);
  .percent {
    background: #eeefff;
    position: absolute;
    left: 0;
    top: 0;
    bottom: 0;
    right: 0;
    transform-origin: 0 50%;
    overflow: hidden;
    transition: background 0.6s ease, transform 0.16s ease;
    transform: scaleX(calc(var(--percent) / 100));
    span {
      display: block;
      position: absolute;
      right: 3em;
      width: 100%;
      top: 54%;
      // top: 3em;
      // height: 3em;
      opacity: 0;
      transform: translateY(0.5px);
      transition: transform 0.8s ease;
      &:before,
      &:after {
        --r: 0;
        --s: 0.5;
        content: "";
        position: absolute;
        top: 0;
        height: 3px;
        border-radius: 1px;
        background: #5628ee;
        transition: background 0.8s ease, transform 0.8s ease, height 0.3s ease;
        transform: rotate(var(--r)) scaleY(var(--s));
      }
      &:before {
        right: 0;
        width: 64%;
        transform-origin: 0 50%;
      }
      &:after {
        left: 0;
        width: 38%;
        transform-origin: 100% 50%;
      }
    }
    div {
      --x: 0;
      transform: translateX(var(--x));
      transition: transform 1s ease;
      position: absolute;
      left: 0;
      bottom: 20%;
      width: 300%;
    }
    svg {
      display: block;
      height: 12px;
      width: 100%;
      stroke-width: 4px;
      color: #5628ee;
      color: linear-gradient(
          90deg,
          #3563ff -15.99%,
          #6b52ff 43.81%,
          #9146ff 101.25%
        ),
        #000;
      transition: color 0.5s ease;
    }
  }
  &.paused {
    &:not(.finished) {
      .percent {
        div {
          --x: -66.66%;
          svg {
            color: #cdd9ed;
            animation: down 0.8s linear forwards;
          }
        }
      }
      .text {
        & > div {
          div {
            small {
              &:first-child {
                opacity: 0;
              }
              &:last-child {
                opacity: 1;
                transition-delay: 0.4s;
              }
            }
          }
        }
      }
    }
  }
  &.finished {
    .percent {
      background: #fff;
      span {
        opacity: 1;
        transform: translate(-20px, -19px);
        &:before,
        &:after {
          --s: 1;
          background: #99a3ba;
          transition: background 0.6s ease, transform 0.6s ease 0.45s;
          animation: check 0.4s linear forwards 0.6s;
        }
        &:before {
          --r: -50deg;
        }
        &:after {
          --r: 38deg;
        }
      }
      svg {
        opacity: 0;
      }
    }
    .text {
      --y: 0;
      & > div {
        opacity: 0;
      }
    }
    nav {
      opacity: 0;
      pointer-events: none;
    }
  }
  .text {
    --y: -18px;
    position: relative;
    z-index: 1;
    transform: translateY(var(--y));
    transition: transform 0.6s ease;
    strong {
      display: block;
      color: #7209b7;
      font-size: 1.3em;
      font-weight: 700;
      line-height: 200%;
    }
    & > div {
      position: absolute;
      left: 0;
      top: 100%;
      transform: translateY(6px);
      line-height: 20px;
      display: flex;
      align-items: center;
      transition: opacity 0.4s ease;
      small {
        white-space: nowrap;
        vertical-align: top;
        display: block;
        color: #7209b7;
        font-size: 1em;
      }
      & > small {
        width: 30px;
        text-align: center;
        &:before {
          content: counter(percent);
        }
      }
      div {
        vertical-align: top;
        display: inline-block;
        position: relative;
        margin-left: 4px;
        &:before {
          content: "";
          width: 2px;
          height: 2px;
          display: block;
          border-radius: 50%;
          background: #99a3ba;
          display: inline-block;
          vertical-align: top;
          margin-top: 9px;
        }
        small {
          position: absolute;
          top: 0;
          left: 8px;
          transition: opacity 0.3s ease;
          &:first-child {
            transition-delay: 0.4s;
          }
          &:last-child {
            opacity: 0;
          }
        }
      }
    }
  }
  nav {
    z-index: 1;
    position: relative;
    display: flex;
    align-items: center;
    margin-left: auto;
    transition: opacity 0.4s ease;
    ul {
      margin: 0;
      padding: 0;
      list-style: none;
      display: flex;
      &:not(:last-child) {
        margin-right: 16px;
      }
      &:first-child {
        --y: 8px;
        opacity: 0;
        transform: translateY(var(--y));
        transition: opacity 0.3s ease, transform 0.4s ease;
      }
      li {
        &:not(:last-child) {
          margin-right: 12px;
        }
        a {
          --r: 0deg;
          --s: 1.01;
          display: block;
          transform: rotate(var(--r)) scale(var(--s)) translateZ(0);
          transition: transform 0.6s ease, background 0.4s ease;
          svg {
            display: block;
            width: 24px;
            height: 24px;
            color: #99a3ba;
            color: #919090;
          }
          &:active {
            --s: 0.84;
            transition: transform 0.3s ease, background 0.4s ease;
          }
          &.dots {
            --r: 90deg;
          }
          &.btn {
            width: 36px;
            height: 36px;
            border-radius: 50%;
            position: relative;
            background: rgba(170, 166, 166, 0.27);
            svg {
              position: absolute;
              left: 9px;
              top: 9px;
              width: 18px;
              height: 18px;
            }
            &:hover {
              background: #e4ecfa;
            }
            &.play {
              --r: 90deg;
              svg {
                &:last-child {
                  transform: scale(-1) translateZ(0);
                }
              }
              &.active {
                --r: 0;
              }
            }
            &.cancel {
              &:before,
              &:after {
                --r: -45deg;
                content: "";
                display: block;
                width: 3px;
                border-radius: 1px;
                height: 18px;
                background: #919090;
                position: absolute;
                left: 50%;
                top: 50%;
                margin: -9px 0 0 -2px;
                transform: rotate(var(--r)) scale(0.9) translateZ(0);
              }
              &:after {
                --r: 45deg;
              }
            }
          }
        }
      }
    }
  }
  &:hover {
    nav {
      ul {
        &:first-child {
          --y: 0;
          opacity: 1;
        }
      }
    }
  }
}

@keyframes down {
  40% {
    transform: translateY(2px);
  }
}

@keyframes check {
  100% {
    background: linear-gradient(
        90deg,
        #3563ff -15.99%,
        #6b52ff 43.81%,
        #9146ff 101.25%
      ),
      var(--Color, #000);
  }
}
</style>
