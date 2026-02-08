import os
from openai import OpenAI
from dotenv import load_dotenv
from typing import List
from openai.types.chat import ChatCompletionMessageParam


load_dotenv()


class HelloAgentsLLM:
    def __init__(self, model: str | None = None, apiKey: str | None = None, baseUrl: str | None = None, timeout: int | None = None):
        """
        优先使用传入参数，否则从环境变量加载
        """
        self.model = model or os.getenv("LLM_MODEL_ID")
        apiKey = apiKey or os.getenv("LLM_API_KEY")
        baseUrl = baseUrl or os.getenv("LLM_BASE_URL")
        timeout = timeout or int(os.getenv("LLM_TIMEOUT", 60))

        if not all([self.model, apiKey, baseUrl]):
            raise ValueError("环境变量需要定义ID,API和服务地址")

        self.client = OpenAI(api_key=apiKey, base_url=baseUrl, timeout=timeout)

    def think(self, messages: List[ChatCompletionMessageParam], temperature: float = 0) -> str:
        # 神坑，Pylance对所有数据类型都要求显性检查
        #
        current_model = self.model
        assert current_model is not None

        print(f"正在调用{self.model}模型")
        try:
            # non-stream
            response = self.client.chat.completions.create(
                model=current_model,
                messages=messages,
                temperature=temperature,
            )
            print("大模型响应成功")
            return str(response.choices[0].message.content)
        except Exception as e:
            print(f"调用时发生错误:{e}")
            return ""


if __name__ == "__main__":
    try:
        llmClient = HelloAgentsLLM()
        # 必须显性声明
        exampleMessages: List[ChatCompletionMessageParam] = [
            {"role": "system", "content": "You are a helpful assistant with Python"},
            {"role": "user", "content": "写一个快速排序算法"}
        ]
        responseText = llmClient.think(exampleMessages)
        if responseText:
            print(responseText)
    except ValueError as e:
        print(e)
