from typing import List, Dict, Any, Optional
from environment import HelloAgentsLLM
from dotenv import load_dotenv


class Memory:
    """
    使用一个列表 records 来按顺序存储每一次的行动和反思。
    add_record 方法负责向记忆中添加新的条目。
    get_trajectory 方法是核心，它将记忆轨迹“序列化”成一段文本，可以直接插入到后续的提示词中，为模型的反思和优化提供完整的上下文。
    get_last_execution 方便我们获取最新的“初稿”以供反思。
    """

    def __init__(self):
        self.records: List[Dict[str, Any]] = []

    def add_record(self, record_type: str, content: str):
        """
        :param record_type: execution, reflection
        :param content: 具体的内容
        """
        record = {"type": record_type, "content": content}
        self.records.append(record)
        print(f"记忆已经更新，新增一条'{record_type}'记录")

    def get_trajectory(self) -> str:
        trajectory_parts = []
        for record in self.records:
            if record['type'] == 'execution':
                trajectory_parts.append(
                    f"----上一轮尝试(代码)----\n{record['content']}")
            elif record['type'] == 'reflection':
                trajectory_parts.append(f"----评审员反馈----\n{record['content']}")

        return "\n\n".join(trajectory_parts)

    def get_last_execution(self) -> Optional[str]:
        for record in reversed(self.records):
            if record['type'] == 'execution':
                return record['content']
        return None


INITIAL_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。请根据以下要求，编写以一个Python函数。
你的代码必须包含完整的函数签名、文档字符串，并遵循PEP 8编码规范。

要求{task}

请直接输出代码，不要包含任何额外的解释。
"""

REFLECT_PROMPT_TEMPLATE = """
你是一位极其严格的代码评审专家和资深算法工程师，对代码的性能有极致的要求。
你的任务是审查以下代码。，并专注于找出其在<strong>算法效率</strong>上的主要瓶颈。

# 原始任务:
{task}

# 待审查的代码:

```python
{code}
```
请分析该代码的时间复杂度，并思考是否存在一种<strong>算法上更优</strong>的解决方案显著提升性能。
如果存在，请清晰指出当前算法的不足，并提出具体的、可行的改进算法建议（例如，使用筛法代替除法）。
如果代码在算法层面已经达到最优，才能回答"无需改进"。

请直接输出你的反馈，不要包含任何额外的解释。
"""

REFINE_PROMPT_TEMPLATE = """
你是一位资深的Python程序员。你正在根据一位代码评审专家的反馈来优化你的代码。

# 原始任务：
{task}

# 你上一轮尝试的代码
{last_code_attempt}
评审员的反馈:
{feedback}

请根据评审员的反馈，生成一个优化后的新版本代码。
你的代码必须包含完整的函数签名，文档字符串，并遵循PEP 8编码规范。
请直接输出优化后的代码，不要包含任何额外的解释。
"""


class ReflectionAgent:
    def __init__(self, llm_client, max_iterations=3):
        self.llm_client = llm_client
        self.memory = Memory()
        self.max_iterations = max_iterations

    def run(self, task: str):
        print(f"\n---开始执行任务---\n任务:{task}")

        initial_prompt = INITIAL_PROMPT_TEMPLATE.format(task=task)
        initial_code = self._get_llm_response(initial_prompt)
        self.memory.add_record("execution", initial_code)

        for i in range(self.max_iterations):
            print(f"\n----第{i+1}/{self.max_iterations}轮迭代----")
            print("\n->正在进行反思...")
            last_code = self.memory.get_last_execution()
            reflect_prompt = REFLECT_PROMPT_TEMPLATE.format(
                task=task, code=last_code)
            feedback = self._get_llm_response(reflect_prompt)
            self.memory.add_record("reflection", feedback)

            if "无需改进" in feedback:
                print("\n反思认为代码已经无需改进，任务完成")
                break

            print("\n->正在进行优化...")
            refine_prompt = REFINE_PROMPT_TEMPLATE.format(
                task=task,
                last_code_attempt=last_code,
                feedback=feedback
            )
            refined_code = self._get_llm_response(refine_prompt)
            self.memory.add_record("execution", refined_code)

        final_code = self.memory.get_last_execution()
        print(f"\n----任务完成----\n最终生成的代码\n```python\n{final_code}\n```")
        return final_code

    def _get_llm_response(self, prompt: str) -> str:
        messages = [{"role":"user","content":prompt}]
        response_text = self.llm_client.think(messages=messages) or ""
        return response_text
        
if __name__ == '__main__':
    load_dotenv()
    llm_client = HelloAgentsLLM()
    reflect = ReflectionAgent(llm_client)
    reflect.run("编写一个Python函数，找出1到n之间所有的素数 (prime numbers)。")