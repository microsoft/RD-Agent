//crypto.js文件内容
import CryptoJS from 'crypto-js'
export default { // 加密
    /**
     * @description: 加密
     * @param {*} word
     * @param {*} keyStr
     */
    set(word, keyStr) {
        keyStr = keyStr || 'abcdef0123456789' // 16位的密钥，自己定义，和下面的密钥要相同
        var srcs = CryptoJS.enc.Utf8.parse(word) //  字符串到数组转换，解析明文
        var key = CryptoJS.enc.Utf8.parse(keyStr) //  字符串到数组转换，解析秘钥
            // mode:加密方式；padding:填充方式；iv便宜向量（可选）
        var encrypted = CryptoJS.AES.encrypt(srcs, key, { mode: CryptoJS.mode.ECB, padding: CryptoJS.pad.Pkcs7 })
        return encrypted.toString() // 加密后的结果是对象，要转换为文本
    },

    /**
     * @description: 解密
     * @param {*} word
     * @param {*} keyStr
     */
    get(word, keyStr) {
        keyStr = keyStr || 'abcdef0123456789'
        var key = CryptoJS.enc.Utf8.parse(keyStr) //  字符串到数组转换
        var decrypt = CryptoJS.AES.decrypt(word, key, { mode: CryptoJS.mode.ECB, padding: CryptoJS.pad.Pkcs7 })
        return CryptoJS.enc.Utf8.stringify(decrypt).toString() //  数组到字符串转换
    }
}