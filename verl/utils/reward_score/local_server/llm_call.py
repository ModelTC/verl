from math import e
import re
import os
from .build_model import APIModel


import re
import os
import json  # <-- 1. 需要导入json模块
from .build_model import APIModel

# --- 默认值 ---
# api_url 的默认值应为一个列表
default_api_url_list = ["http://10.119.22.24:8080/v1"]
model_name = "qwq32B"

# --- API URL 配置 ---
# 尝试从环境变量获取 api_url
env_api_url_str = os.getenv("api_url")

if env_api_url_str:
    try:
        # 2. 尝试将字符串解析为Python对象（这里我们期望它是一个列表）
        api_url = json.loads(env_api_url_str)
        
        # 3. 添加一个检查，确保解析出的是一个列表
        if not isinstance(api_url, list):
            print(f"Error: 环境变量 'api_url' 不是一个有效的JSON列表。使用默认值。")
            api_url = default_api_url_list
        else:
            api_url = [f"http://{url}:8080/v1" for url in api_url]
    except json.JSONDecodeError:
        # 4. 如果字符串不是有效的JSON（例如，它只是一个普通的URL），则处理异常
        print(f"Error: 无法解析 'api_url' 环境变量 (不是有效的JSON)。使用默认值。")
        print(f"获取到的值是: {env_api_url_str}")
        api_url = default_api_url_list
else:
    # 5. 如果环境变量未设置，使用默认值
    print(f"Info: 未设置 'api_url' 环境变量。使用默认值。")
    api_url = default_api_url_list

# --- Model Name 配置 ---
# 尝试从环境变量获取 model_name
env_model_name = os.getenv("model_name")
if env_model_name:
    model_name = env_model_name

# --- 打印最终配置 ---
print(f"Using Model: {model_name}")
print(f"Using API URLs: {api_url}")

print("="*100)
print(f"api_url: {api_url}, model_name: {model_name}")

# api_model = APIModel(api_url, model_name, address=os.getenv("address"))
api_model_list = [APIModel(url, model_name, address=os.getenv("address")) for url in api_url]

def generate_chat(messages, max_tokens=8192, temperature=0.0, idx=0):
    api_model = api_model_list[idx%len(api_model_list)]
    if os.getenv("is_self_reward"):
        response = api_model.generate_selfreward(messages, max_tokens, temperature)
    else: 
        response = api_model.generate_chat(messages, max_tokens, temperature)
    return response.strip()


def extract_chat(messages, max_tokens=8192, temperature=0.0, idx=0):
    api_model = api_model_list[idx%len(api_model_list)]
    if os.getenv("is_self_reward"):
        response = api_model.generate_selfreward(messages, max_tokens, temperature)
    else: 
        response = api_model.generate_chat(messages, max_tokens, temperature)
    return response.strip()


prompt_template = """
请判断以下文本是否满足给定的约束，仅回答是或否，不要输出其他内容。


原始指令：$I$

文本：$R$

约束：$C$

原始指令描述了基本的任务信息，给定的约束介绍了应该满足的具体的一个约束。
请判断以下文本是否满足给定的这个约束（仅仅判断是否满足给定的约束），仅回答是或否，不要输出其他内容。
"""


def llm_judge(instruction, response, constraint, idx=0):
    if isinstance(response, list):
        response = "\n\n".join(response)
    prompt = prompt_template.replace("$R$", response).replace("$C$", constraint).replace("$I", instruction)
    data = [
        {"role": "user", "content": prompt}
    ]
    response = generate_chat(data, idx=0)
    if "</think>" in response:
        response = response.split("</think>")[1].strip()
    # print(f"Prompt: {prompt}\nResponse: {response}")
    return response[0] == "是"


def llm_extract(instruction, response, specific_prompt, idx=0):
    prompt_suffix = "\n请直接输出文本中的原文信息，不要改写，不要添加任何额外的信息。"

    prompt = f"文本：{response}\n抽取要求：{specific_prompt}" + prompt_suffix
    data = [
        {"role": "user", "content": prompt}
    ]
    response = extract_chat(data, max_tokens=8192, idx=idx)
    if "</think>" in response:
        response = response.split("</think>")[1].strip()
    return response


def extract_score(text):
    match = re.search(r'\[\[(\d+)\]\]', text)
    try:
        return int(match.group(1))
    except:
        return 0


def llm_score(instruction, response, checkers, idx=0):
    prompt = f"""
    请判断给定的回复是否遵循指令中的约束，比如长度、风格、格式等约束。
    
    [指令]
    {instruction}

    [回复]
    {response}

    [约束]
    {checkers}

    请判断给定的回复是否遵循指令中的约束，比如长度、风格、格式等约束。
    请在回答的最开始用[[score]]格式输出你的分数。
    如果遵循所有的约束，请输出[[1]]，否则输出[[0]]
    """
    data = [
        {"role": "user", "content": prompt}
    ]
    response = generate_chat(data, max_tokens=4096*2, idx=idx)
    if "</think>" in response:
        response = response.split("</think>")[1].strip()
    # print(f"Prompt: {prompt}\nResponse: {response}")
    score = extract_score(response)
    return score



if __name__ == "__main__":
    result = llm_judge(
        "What is the speed of light, and how does it compare to the speed of sound in a vacuum? Please answer with a tone of excitement and wonder.The word 'light' should appear at least 3 times, and your response should contain exactly 3 sentences.",
        "Oh, the speed of light is a mind-blowing marvel of the universe, traveling at a staggering 299,792,458 meters per second (m/s)! 🌟 In comparison, the speed of sound in a vacuum is non-existent because sound needs a medium to travel, whereas light races through the void with unparalleled grace and swiftness. Imagine the thrill of light zooming across the cosmos, effortlessly outpacing any sound, and illuminating the mysteries of space with its incredible speed!",
        "Your response should contain exactly 3 sentences",
        idx=0
    )
    print(result)

    


