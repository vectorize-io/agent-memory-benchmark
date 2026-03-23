import { createApp } from 'vue'
import { createRouter, createWebHistory, RouterView } from 'vue-router'
import './style.css'
import DatasetList   from './pages/DatasetList.vue'
import DatasetDetail from './pages/DatasetDetail.vue'
import SplitDetail   from './pages/SplitDetail.vue'
import RunDetail     from './pages/RunDetail.vue'
import QueryView     from './pages/QueryView.vue'
import DocumentView  from './pages/DocumentView.vue'

const router = createRouter({
  history: createWebHistory(),
  routes: [
    { path: '/',                                    component: DatasetList },
    { path: '/dataset/:name',                       component: DatasetDetail },
    { path: '/dataset/:name/:split',                component: SplitDetail },
    { path: '/dataset/:name/:split/queries',        component: QueryView },
    { path: '/dataset/:name/:split/documents',      component: DocumentView },
    { path: '/run/:path(.*)',                        component: RunDetail },
  ],
})

const app = createApp(RouterView)
app.use(router)
app.mount('#app')
