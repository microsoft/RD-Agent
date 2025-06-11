#!/usr/bin/env python3
"""
OpenAI Compatible API Client
调用OpenAI兼容的API接口示例
"""

import os
import json
import requests
from typing import Dict, List, Optional, Any
from openai import OpenAI

# API配置
BASE_URL = "https://amstrongzyf-gemini-57.deno.dev/chat/completions"
API_KEY = "your-api-key-here"  # 根据需要设置API密钥

class OpenAICompatibleClient:
    """OpenAI兼容API客户端"""
    
    def __init__(self, base_url: str, api_key: str = "dummy"):
        self.base_url = base_url
        self.api_key = api_key
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def chat_completion_requests(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Dict[str, Any]:
        """使用requests库调用chat completion API"""
        
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "stream": stream,
            **kwargs
        }
        
        if max_tokens is not None:
            payload["max_tokens"] = max_tokens
        
        try:
            response = requests.post(
                self.base_url,
                headers=self.headers,
                json=payload,
                timeout=30
            )
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            print(f"请求错误: {e}")
            return {"error": str(e)}
        except json.JSONDecodeError as e:
            print(f"JSON解析错误: {e}")
            return {"error": "Invalid JSON response"}
    
    def chat_completion_openai_sdk(
        self,
        messages: List[Dict[str, str]],
        model: str = "gpt-3.5-turbo",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        stream: bool = False,
        **kwargs
    ) -> Any:
        """使用OpenAI SDK调用chat completion API"""
        
        try:
            # 创建OpenAI客户端，使用自定义base_url
            client = OpenAI(
                api_key=self.api_key,
                base_url=self.base_url.replace("/chat/completions", "")  # 移除具体endpoint，SDK会自动添加
            )
            
            response = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream,
                **kwargs
            )
            
            return response
        
        except Exception as e:
            print(f"OpenAI SDK错误: {e}")
            return None

def demo_requests_method():
    """演示使用requests方法调用API"""
    print("=== 使用requests方法调用API ===")
    
    client = OpenAICompatibleClient(BASE_URL, API_KEY)
    
    messages = [
        {"role": "system", "content": "你是一个有用的AI助手。"},
        {"role": "user", "content": "请解释什么是人工智能？"}
    ]
    
    response = client.chat_completion_requests(
        messages=messages,
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=500
    )
    
    if "error" not in response:
        print("响应成功:")
        print(json.dumps(response, indent=2, ensure_ascii=False))
        
        # 提取回复内容
        if "choices" in response and len(response["choices"]) > 0:
            content = response["choices"][0]["message"]["content"]
            print(f"\nAI回复: {content}")
    else:
        print(f"请求失败: {response['error']}")

def demo_openai_sdk_method():
    """演示使用OpenAI SDK方法调用API"""
    print("\n=== 使用OpenAI SDK方法调用API ===")
    
    client = OpenAICompatibleClient(BASE_URL, API_KEY)
    
    messages = [
        {"role": "system", "content": "你是一个有用的AI助手。"},
        {"role": "user", "content": "请用简短的话解释什么是机器学习？"}
    ]
    
    response = client.chat_completion_openai_sdk(
        messages=messages,
        model="gpt-3.5-turbo",
        temperature=0.7,
        max_tokens=300
    )
    
    if response:
        print("响应成功:")
        print(f"模型: {response.model}")
        print(f"创建时间: {response.created}")
        print(f"AI回复: {response.choices[0].message.content}")
        
        # 使用情况
        if hasattr(response, 'usage') and response.usage:
            print(f"Token使用情况: {response.usage}")
    else:
        print("请求失败")

def demo_streaming():
    """演示流式响应"""
    print("\n=== 演示流式响应 (仅requests方法) ===")
    
    client = OpenAICompatibleClient(BASE_URL, API_KEY)
    
    messages = [
        {"role": "user", "content": "请写一个关于春天的短诗"}
    ]
    
    # 注意：流式响应需要特殊处理
    payload = {
        "model": "gpt-3.5-turbo",
        "messages": messages,
        "temperature": 0.8,
        "stream": True
    }
    
    try:
        response = requests.post(
            client.base_url,
            headers=client.headers,
            json=payload,
            stream=True,
            timeout=30
        )
        response.raise_for_status()
        
        print("流式响应:")
        for line in response.iter_lines():
            if line:
                line_str = line.decode('utf-8')
                if line_str.startswith('data: '):
                    data_str = line_str[6:]  # 移除 'data: ' 前缀
                    if data_str.strip() == '[DONE]':
                        break
                    try:
                        data = json.loads(data_str)
                        if 'choices' in data and len(data['choices']) > 0:
                            delta = data['choices'][0].get('delta', {})
                            if 'content' in delta:
                                print(delta['content'], end='', flush=True)
                    except json.JSONDecodeError:
                        continue
        print("\n")
        
    except Exception as e:
        print(f"流式请求错误: {e}")

def main():
    """主函数"""
    print("OpenAI Compatible API 调用示例")
    print(f"Base URL: {BASE_URL}")
    print("-" * 50)
    
    # 演示requests方法
    demo_requests_method()
    
    # 演示OpenAI SDK方法
    demo_openai_sdk_method()
    
    # 演示流式响应
    demo_streaming()

if __name__ == "__main__":
    main()
