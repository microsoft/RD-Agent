<template>
  <div class="content">
    <div class="content-1">
      <h2>Get Valuable</h2>
      <h2 class="h2"><span>Insights</span> Instantly</h2>
      <div class="bg-1">
        <button ref="btn" @mousemove="moveMouse" @click="tryNow">
          <span>Try now</span>
        </button>
      </div>
    </div>
    <div class="modal">
      <div class="des">
        <p>
          R&D-Agent is your dedicated data scientist powered by an LLM. It
          automatically researches your unique data-mining tasks, learns domain
          knowledge to evolve from practice, develops the most tailored datasets
          and models for your data sources, and delivers solutions with
          significant industrial value.
        </p>
      </div>
      <div class="content-2">
        <h2>AI Drives Data-Driven AI</h2>
        <img src="@/assets/images/Data-Driven.png" alt="R&D-Agent" />
        <p>
          In modern industry, research and development (R&D) is crucial for the
          enhancement of industrial productivity, especially in the AI era,
          where the core aspects of R&D are mainly focused on data and models.
          We are committed to automate these high-value generic R&D processes
          through our open source R&D automation tool R&D-Agent, which let AI
          drive data-driven AI.   R&D-Agent harnesses the strengths of LLMs to
          create an integrated, automated system for data-driven R&D, ensuring
          that innovation and development proceed hand in hand, driven by the
          powerful capabilities of modern AI. 
        </p>
        <h2 class="h2">The Framework of R&D-Agent</h2>
        <img
          src="@/assets/images/Framework.png"
          alt="R&D-Agent"
          style="margin: 3em 0 2em"
        />
        <p>
          Methodologically, we propose an autonomous agent framework consisting
          of two key parts: (R)esearch and (D)evelopment. Research involves
          actively exploring by proposing new ideas, while Development focuses
          on realizing these ideas. The effectiveness of these two components is
          continually refined through practice, with both research and
          development capabilities learning and growing over time. 
        </p>
        <div class="content-2-bg"></div>
      </div>
      <div class="content-3">
        <h2>Ready to experiment?</h2>
        <p>Dive into our Playground and set your creativity free!</p>
        <div class="btn-box">
          <button class="playground" @click="tryNow">Playground</button>
            <button class="contact" @click="goToGitHub">GitHub</button>
        </div>
      </div>
    </div>
  </div>
</template>
<script setup>
import { ref, watch, reactive, nextTick } from "vue";
import { useRouter } from "vue-router";

const router = useRouter();
const btn = ref(null);
const moveMouse = (e) => {
  const x = e.offsetX;
  const y = e.offsetY;
  btn.value.style.setProperty("--x", x + "px");
  btn.value.style.setProperty("--y", y + "px");
};
const tryNow = () => {
  router.push("/Playground");
};
const goToGitHub = () => {
  window.open("https://github.com/microsoft/R&D-Agent");
};
</script>

<style scoped lang="scss">
.content {
  width: 100%;
  text-align: center;
  .content-1 {
    width: 100vw;
    height: calc(100vh - 6.125em);
    position: fixed;
    top: 6.125em;
    padding-top: 1em;
    h2 {
      color: var(--text-color);
      text-align: center;
      font-size: 4.5em;
      line-height: 1.3em;
      font-weight: 700;
    }
    .h2 {
      font-family: "Microsoft YaHei";
      span {
        background: linear-gradient(
          90deg,
          #2667ff 0%,
          #9d41ff 50%,
          #d453ff 99%
        );
        background-clip: text;
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
      }
    }
    .bg-1 {
      width: 70%;
      height: auto;
      padding-top: 37.4%;
      background: url(@/assets/images/bg-1.png) no-repeat center center;
      background-size: contain;
      margin: 0 auto;
      position: relative;
      // padding-top: 3em;
      display: flex;
      justify-content: center;
      button {
        // margin-top: 0.5em;
        position: absolute;
        top: 1.4em;
        width: 7.5em;
        background-image: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 999px;
        padding: 0.5em 1.2em 0.6em;
        // border-radius: 999px;
        background-image: linear-gradient(
          95deg,
          #2667ff -11.58%,
          #9d41ff 90.52%,
          #d453ff 190.58%
        );
        box-shadow: 1px 4px 5px 0px rgba(235, 224, 253, 0.32) inset,
          -1px -2px 3.4px 0px rgba(117, 89, 190, 0.65) inset,
          -1px 1px 2px 0px rgba(117, 89, 190, 0.2),
          1px 1px 3px 0px rgba(172, 158, 210, 0.72);
        backdrop-filter: blur(15px);
        // background-clip: text;
        // -webkit-background-clip: text;
        // -webkit-text-fill-color: transparent;
        font-size: 1.65em;
        font-weight: 600;
        text-transform: capitalize;

        // padding: 14px 50px;
        // border-radius: 4px;
        border: none;
        color: #fff;
        cursor: pointer;
        outline: none;
        overflow: hidden;
        // box-shadow: 0 14px 30px rgba(0, 0, 0, 0.15);
        // font-family: "Lato", sans-serif;
        // font-size: 16px;
        // text-transform: uppercase;
        // letter-spacing: 2px;

        span {
          position: relative;
        }

        &::before {
          --size: 0;
          // --x: 0;
          // --y: 0;
          content: "";
          position: absolute;
          left: var(--x);
          top: var(--y);
          width: var(--size);
          height: var(--size);
          background: radial-gradient(
            circle closest-side,
            #a3b1f1,
            transparent
          );
          transform: translate(-50%, -50%);
          transition: all 0.2s ease, left 0s, top 0s;
        }

        &:hover::before {
          --size: 200px;
        }
      }
    }
  }
  .modal {
    width: 100%;
    margin-top: 62vh;
    position: relative;
    z-index: 1;
    .des {
      width: 100%;
      opacity: 0.9;
      background: var(--bg-white);
      padding: 3.625em 0;
      p {
        width: 1130px;
        color: var(--text-color);
        font-size: 1.375em;
        line-height: 200%;
        margin: 0 auto;
      }
    }
    .content-2 {
      width: 100%;
      padding: 9em 0 16em;
      color: var(--text-color);
      background-color: var(--bg-white);
      position: relative;
      .content-2-bg {
        width: 100%;
        height: 340px;
        background: url(@/assets/images/footer-bg.png) no-repeat;
        background-position: top;
        background-size: cover;
        position: absolute;
        bottom: 0;
      }
      h2 {
        font-size: 3.125em;
        font-weight: 700;
        line-height: 200%;
      }
      img {
        margin-top: 1.625em;
      }
      p {
        width: 1220px;
        margin: 0 auto;
        font-size: 1.375em;
        font-weight: 400;
        line-height: 200%;
        margin-top: 1.64em;
        position: relative;
        z-index: 9;
      }
      .h2 {
        margin-top: 3em;
      }
    }
    .content-3 {
      background: var(--bg-grey);
      padding: 5em 0 7em;
      color: var(--text-color);
      h2 {
        font-size: 2.4em;
        font-weight: 700;
        line-height: 200%;
      }
      p {
        font-size: 1.8em;
        line-height: 200%;
      }
      .btn-box {
        margin-top: 3.125em;
        button {
          border-radius: 1.46em;
          font-size: 1.4em;
          font-weight: 700;
          line-height: 200%;
          width: 9em;
          height: 2.5em;
          cursor: pointer;
        }
        .playground {
          background: var(--bg-black);
          border-color: var(--bg-black);
          color: var(--text-white-color);
          margin-right: 2.2em;
          &:hover {
            background: rgba(43, 43, 43, 0.74);
            border-color: rgba(43, 43, 43, 0.74);
          }
        }
        .contact {
          border-color: var(--bg-black);
          border: 2px solid var(--bg-black);
          background: var(--bg-white);
          box-shadow: 8px 11px 30px 0px #edf0ff;
          color: var(--btn-color);
          &:hover {
            background: #f2f5fa;
          }
        }
      }
    }
  }
}
</style>
