import axios from 'axios'

axios.defaults.headers.post['Content-Type'] = 'application/json'

const service = axios.create({
    baseURL: ''
})
service.defaults.timeout = 5 * 60 * 1000;

// request拦截器
service.interceptors.request.use(
    config => {
        if (config.data) {}
        if (config.params) {
            // console.log('request: ', config.params)
        }
        return config
    },
    error => {
        console.log('error-request: ', error)
        return error
    }
)

// respone拦截器
service.interceptors.response.use(
    response => {
        return response.data
    },
    error => {
        console.log('error-response: ', error)
        console.log('error-response: ', error.response)
        return error.response
    }
)

export default service