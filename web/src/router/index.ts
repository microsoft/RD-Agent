import { createRouter, RouteRecordRaw, createWebHashHistory } from 'vue-router'

const routes: Array<RouteRecordRaw> = [
  {
    path: '/',
    name: 'Home',
    component: () => import('../views/Home.vue'),
    meta: {
      keepAlive: true, //此页面需要缓存
      requiresFrontEndAuth: true,
      footerBg: "#F6FAFF"
    },
  },
  {
    path: '/Playground',
    name: 'Playground',
    component: () => import('../views/Playground.vue'),
    meta: {
      keepAlive: false, //此页面需要缓存
      requiresFrontEndAuth: true,
      footerBg: "#fff"
    },
  },
  {
    path: '/PlaygroundPage',
    name: 'PlaygroundPage',
    component: () => import('../views/PlaygroundPage.vue'),
    meta: {
      keepAlive: false, //此页面需要缓存
      requiresFrontEndAuth: true,
      footerBg: "#fff"
    },
  }
  // {
  //   path: '/Login',
  //   name: 'Login',
  //   component: () => import('../views/Login.vue'),
  //   meta: {
  //     keepAlive: false, //此页面需要缓存
  //     requiresFrontEndAuth: false
  //   },
  // }
]

const router = createRouter({
  history: createWebHashHistory(),
  routes
})

// 前端添加密码，防止release流程未走完，外部人员访问
// router.beforeEach((to, from, next) => {
//   console.log(from)
//     if (!!to.meta && to.meta.requiresFrontEndAuth === false) {
//         //这里判断用户是否登录，验证本地存储是否有token
//         next();
//         return;
//     }
//     if (!sessionStorage.getItem("token")) { // 判断当前的token是否存在
//         next({
//             name: 'Login',
//             query: { redirect: to.fullPath }
//         })
//     } else {
//         next();
//     }
// })

export default router