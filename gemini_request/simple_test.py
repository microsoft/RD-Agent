#!/usr/bin/env python3
"""
简单的API测试脚本
用于快速验证OpenAI兼容API是否正常工作
"""

import json
import requests

# API配置
BASE_URL = "https://amstrongzyf-gemini-57.deno.dev/chat/completions"
API_KEY = "AIzaSyDCyFB6802bm48E3tfgHqB1vDAYuhUq-pg"  # 可能不需要真实的API密钥

def simple_test():
    """简单的API测试"""
    
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "gemini-2.0-flash-exp",
        "messages": [
            {"role": "user", "content": "Hello! Can you say hi back?"}
        ],
        "temperature": 0.7,
        "max_tokens": 100
    }
    
    print("正在测试API连接...")
    print(f"URL: {BASE_URL}")
    print(f"请求数据: {json.dumps(payload, indent=2)}")
    print("-" * 50)
    
    try:
        response = requests.post(
            BASE_URL,
            headers=headers,
            json=payload,
            timeout=30
        )
        
        print(f"状态码: {response.status_code}")
        print(f"响应头: {dict(response.headers)}")
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API调用成功!")
            print(f"响应: {json.dumps(result, indent=2, ensure_ascii=False)}")
            
            # 提取回复内容
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                print(f"\n🤖 AI回复: {content}")
                
        else:
            print(f"❌ API调用失败! 状态码: {response.status_code}")
            print(f"错误响应: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ 连接错误: 无法连接到API服务器")
    except requests.exceptions.Timeout:
        print("❌ 超时错误: 请求超时")
    except requests.exceptions.RequestException as e:
        print(f"❌ 请求错误: {e}")
    except json.JSONDecodeError:
        print("❌ JSON解析错误: 响应不是有效的JSON格式")
    except Exception as e:
        print(f"❌ 未知错误: {e}")

if __name__ == "__main__":
    simple_test() 