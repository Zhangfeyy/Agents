from serpapi import SerpApiClient
import os
from typing import Dict, Any, Callable
from dotenv import load_dotenv

load_dotenv()

def search(query: str) -> str:
    print(f"正在执行[SerpApi]网页搜索: {query}")
    try:
        api_key = os.getenv("SERPAPI_API_KEY")
        if not api_key:
            return "请先配置SERPAPI"

        params = {
            "engine": "google",
            "q": query,
            "api_key": api_key,
            "gl": "cn",  # 国家代码
            "hl": "zh-cn"  # 语言代码
        }

        client = SerpApiClient(params)
        results = client.get_dict()

        # 优先寻找最直接答案
        if "answer_box_list" in results:
            return "\n".join(results["answer_box_list"])
        if "answer_box" in results and "answer" in results["answer_box"]:
            return results["answer_box"]["answer"]
        if "knowledge_graph" in results and "description" in results["knowledge_graph"]:
            return results["knowledge_graph"]["description"]
        if "organic_results" in results and results["organic_results"]:
            snippets = [
                # res：获取'title'的值，否则返回空字符串
                f"[{i+1}] {res.get('title','')}\n{res.get('snippet','')}"
                for i, res in enumerate(results["organic_results"][:3])
            ]
            return "\n\n".join(snippets)
        return f"没有找到{query}的信息"

    except Exception as e:
        return f"搜索发生错误: {e}"

# 需要一个工具类调度工具


class ToolExecutor:
    def __init__(self):
        # 一个字典：键是字符串，值是任何值
        self.tools: Dict[str, Dict[str, Any]] = {}

    def registerTool(self, name: str, description: str, func: Callable):
        if name in self.tools:
            print(f"{name}已存在，数据将被覆盖")
        self.tools[name] = {'description': description, "func": func}
        print(f"{name}已注册")

    def getTool(self, name: str) -> Callable:
        func = self.tools.get(name, {}).get("func")
        assert func is not None

        return func

    # enumerate返回索引，items()专属字典，返回键
    def getAvailableTools(self) -> str:
        return "\n".join([
            f"-{name}:{info['description']}"
            for name, info in self.tools.items()
        ])


if __name__ == '__main__':
    toolExecutor = ToolExecutor()
    search_description = "网页搜索引擎"
    toolExecutor.registerTool("Search", search_description, search)
    
    print(toolExecutor.getAvailableTools())
    
    tool_name = "Search"
    tool_input = "Nvidia最新的GPU型号"
    tool_function = toolExecutor.getTool(tool_name)

    
    if tool_function:
            observation = tool_function(tool_input)
            print("----观察----")
            print(observation)
    else:
        print(f"未找到{tool_name}工具")