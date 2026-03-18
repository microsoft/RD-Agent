<template>
  <div class="home">
    <div>
      <div class="token">
        <input
          type="password"
          @keydown.enter="login"
          placeholder="Please enter password"
          v-model.trim="token"
        />
      </div>
      <div>
        <button @click="login">Enter</button>
      </div>
    </div>
  </div>
</template>

<script>
import crypto from "@/utils/crypto.js";
export default {
  name: "Login",
  components: {},
  data() {
    return {
      token: "",
      defaultpwd: "",
    };
  },
  methods: {
    login() {
      if (this.token != this.defaultpwd) {
        alert("请输入正确的密码");
      } else {
        sessionStorage.setItem("token", this.token);
        let path = this.$route.query.redirect || "/";
        this.$router.push(path);
      }
    },
  },
  mounted() {
    this.defaultpwd = crypto.get("vCTcSPKS1eGmRXBh4c6RXA==");
  },
};
</script>
<style scoped lang="scss">
.home {
  width: 100%;
  height: 100%;
  background: #000;
  position: fixed;
  left: 0;
  top: 0;
  z-index: 999999999;
  // padding: 40px;
  display: flex;
  flex-direction: column;
  justify-content: center;
  align-items: center;
  & > div {
    width: 18.75rem;
  }
  .token {
    margin-bottom: 1.25rem;
  }
  input,
  button {
    padding: 0 0.625rem;
    height: 2.375rem;
    width: 15.625rem;
    outline: none;
    border: none;
    box-sizing: border-box;
    font-size: 1rem;
  }
  button {
    background: rgba(0, 101, 255, 0.5);
    color: #fff;
    cursor: pointer;
  }
}
</style>
