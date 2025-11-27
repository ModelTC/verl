from openai import OpenAI
import requests
import asyncio
import aiohttp
from uuid import uuid4


async def post_chat_completion(address: str, payload: dict) -> dict:
    timeout = aiohttp.ClientTimeout(total=None)
    async with aiohttp.ClientSession(timeout=timeout) as session:
        # 追加请求头
        extra_headers = payload.pop("extra_headers", {})
        request_id = extra_headers.get("x-request-id") or uuid4().hex
        extra_headers["x-request-id"] = request_id

        async with session.post(
            f"http://{address}/v1/chat/completions",
            headers={"Authorization": "Bearer token-abc123", **extra_headers},
            json=payload,
        ) as response:
            response.raise_for_status()
            return await response.json()

def _compute_score_async(
    judge_input: str,
    judge_model_url: list,
    model_name: str,
) -> float:
        
    response = asyncio.run(get_judge_output(judge_input, judge_model_url, model_name))

    return response

class APIModel:
    def __init__(self, base_url, model_name, api_key="EMPTY", address=None):
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
        )
        self.address = address
        self.model_name = model_name
    
    def search(self, query, max_tokens=512):
        return self.generate(query, max_tokens=max_tokens)

    def generate(self, query, max_tokens=1024, temperature=0.0):
        try:
            chat_completion = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": query
                    }
                ],
                model=self.model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            response = chat_completion.choices[0].message.content
        except Exception as e:
            print(e)
            response = "NA"
        return response

    def generate_chat(self, messages, max_tokens=1024, temperature=0.0):
        try:
            chat_completion = self.client.chat.completions.create(
                messages=messages,
                model=self.model_name,
                temperature=temperature,
                max_tokens=max_tokens
            )
            response = chat_completion.choices[0].message.content
            return response
        except Exception as e:
            print(e)
            response = "NA"
        return response
    
    def generate_selfreward(self, messages, max_tokens=1024, temperature=0.0):
        try:
            
            response = _compute_score_async(messages[0]["content"], self.address, self.model_name)
            # chat_completion = self.client.chat.completions.create(
            #     messages=messages,
            #     model=self.model_name,
            #     temperature=temperature,
            #     max_tokens=max_tokens
            # )
            # response = chat_completion.choices[0].message.content
            return response
        except Exception as e:
            print(e)
            response = "NA"
        return response

    def safety_check(self, query):
        try:
            chat_completion = self.client.moderations.create(
                model=self.model_name,
                input=query
            )
            import pdb; pdb.set_trace()
            return dict(chat_completion.results[0].categories), chat_completion.results[0].flagged
        except Exception as e:
            print(e)
            response = {"NA": 1}
        return response


class LocalAPIModel:
    def __init__(self, base_url, model_name, api_key="EMPTY"):
        self.base_url = base_url
        self.model_name = model_name
    
    def search(self, query, max_tokens=512):
        return self.generate(query, max_tokens=max_tokens)

    def generate(self, query, max_tokens=1024, temperature=0.0):
        return self.generate_chat(
            [{"role": "user", "content": query}],
            max_tokens=max_tokens,
            temperature=temperature
        )

    def generate_chat(self, messages, max_tokens=1024, temperature=0.0):
        request_data = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": 1024,
            "temperature": 0.0
        }
        response = requests.post(
            self.base_url,
            json=request_data,
            headers={"Content-Type": "application/json"},
            timeout=900 
        )
        if response.status_code == 200:
            resp_json = response.json()
            content = resp_json['choices'][0]['message']['content'].strip()
            return content
        else:
            print(
                f"Failed to fetch response: {response.status_code}, {response.text}"
            )
            return None
    
     