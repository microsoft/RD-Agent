<template>
  <div id="app">
    <Header />
    <router-view v-slot="{ Component }" class="component">
      <keep-alive>
        <component
          :is="Component"
          :key="$route.name"
          v-if="$route.meta.keepAlive"
        />
      </keep-alive>
      <component
        :is="Component"
        :key="$route.name"
        v-if="!$route.meta.keepAlive"
      />
    </router-view>
    <Footer :color="color" />
  </div>
</template>
<script setup>
import { provide, nextTick, ref, watch } from "vue";
import Header from "./components/navBar.vue";
import Footer from "./components/footer.vue";
import { useRoute } from "vue-router";
const isRouterActive = ref(true);
const route = useRoute();
const color = ref("#F6FAFF");
watch(
  () => route.path,
  (newValue, oldValue) => {
    color.value = route.meta.footerBg;
  }
);
provide("reload", () => {
  isRouterActive.value = false;
  nextTick(() => {
    isRouterActive.value = true;
  });
});
</script>

<style scoped>
#app {
  width: 100%;
  height: 100vh;
  box-sizing: border-box;
  background: #fff;
  position: relative;
  z-index: 1;
  display: flex;
  flex-direction: column;
  overflow: hidden;
}
</style>
