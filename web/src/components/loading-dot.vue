<template>
  <span class="spinner"></span>
</template>

<style scoped lang="scss">
.spinner {
  display: inline-block;
  height: 35px;
  width: 35px;
  margin-top: 3px;
  //   background: rgba(0, 0, 0, 0.2);
  border-radius: 50%;
  border-top: 2px solid #fff;
  border-right: 2px solid transparent;
  animation: spinner6 700ms linear infinite;
}

@keyframes spinner6 {
  to {
    transform: rotate(360deg);
  }
}
</style>
