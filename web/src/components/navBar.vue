<template>
  <div class="header-component">
    <div class="nav">
      <router-link to="/">
        <div class="logo-container">
          <img src="@/assets/images/RDAgent-logo.png" alt="R&D-Agent logo" />
        </div>
      </router-link>
      <ul>
        <li>
          <router-link to="/">Homepage</router-link>
        </li>
        <li>
          <router-link to="/Playground">Playground</router-link>
        </li>
      </ul>
    </div>
  </div>
</template>
<script lang="ts">
export default {
  components: {},
  setup() {},
};
</script>
<style scoped lang="scss">
.header-component {
  padding: 1.75em 4.375em 0;
  // height: 6.125em;
  height: 5.15em;
  box-sizing: border-box;
  & > .nav {
    display: flex;
    flex-wrap: nowrap;
    justify-content: flex-start;
    align-items: center;
    flex-direction: row;
    .logo-container {
      margin-right: 4.375em;
      img {
        display: inline-block;
        height: 2.25em;
      }
    }
    ul {
      display: flex;
      flex-direction: row;
      li {
        margin-right: 2.25em;
        text-align: center;
        cursor: pointer;
        a {
          display: inline-block;
          color: var(--nav-default-color);
          font-size: 1.25em;
          font-weight: 700;
          line-height: 200%;
          &:hover {
            color: var(--nav-hover-color);
          }
        }
        .router-link-exact-active {
          color: var(--text-color);

          &:hover {
            color: var(--text-color);
          }
        }
      }
    }
  }
}
</style>
