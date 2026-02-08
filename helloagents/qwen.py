import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "F:/Python/Models/Qwen1.5-0.5B-Chat"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

tokenizer = AutoTokenizer.from_pretrained(model_id)

#加载模型到指定设备
model = AutoModelForCausalLM.from_pretrained(model_id).to(device) # type: ignore

print("模型和分词器加载完成")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "你好，请介绍你自己。"}
]

# 使用分词器的模板格式化输入
text = tokenizer.apply_chat_template(
	messages,
	tokenize=False,
 	add_generation_prompt=True
)

# 编码输入文本
model_inputs = tokenizer([text],return_tensors="pt").to(device)

# print("编码后的输入文本：")
# print(model_inputs)

generation_config = {
	"do_sample":True, #必须先设置
	"temperature":0.7, # 对stochasticity影响比较大
	"top_p": 0.9,
	"top_k":40,
	"max_new_tokens":512
}


generated_ids = model.generate(
	model_inputs.input_ids,
	**generation_config
	
)

# 截掉输入
generated_ids = [
	output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
	# zip配对，：左切片
 
]

# 解码
response = tokenizer.batch_decode(generated_ids,skip_special_tokens=True)[0]

# print("\n模型回答")
# print(response)

#测试zero-shot-----------------------------------------------------------------
zero_messages_sentiment = [
    {"role": "system", "content": "You can help with sentiment classification"},
    {"role": "user", "content": "我非常地悲伤"}
]

zero_text_sentiment = tokenizer.apply_chat_template(
	zero_messages_sentiment,
	tokenize=False,
 	add_generation_prompt=True
)
zero_model_inputs = tokenizer([zero_text_sentiment],return_tensors="pt").to(device)

zero_generated_ids = model.generate(
	zero_model_inputs.input_ids,
	**generation_config
)

# 截掉输入
zero_generated_ids = [
	output_ids[len(input_ids):] for input_ids, output_ids in zip(zero_model_inputs.input_ids, zero_generated_ids)
	# zip配对，：左切片
]

# 解码
# zero_response = tokenizer.batch_decode(zero_generated_ids,skip_special_tokens=True)[0]
# print("zero shot test"+"-"*20)
# print(zero_response)

#测试few-shot-----------------------------------------------------------------
few_messages_sentiment = [
    {"role": "system", "content": "You can help with sentiment classification"},
    {"role": "user", "content": """
     模仿例子输出：
	文本：我很开心
	结果：积极
	---
	文本：我很难过
	结果：消极
	---
	文本：我想睡觉
	结果:""" # prompt对结尾空格敏感
     }
]
few_text_sentiment = tokenizer.apply_chat_template(
	few_messages_sentiment,
	tokenize=False,
 	add_generation_prompt=True
)
few_model_inputs = tokenizer([few_text_sentiment],return_tensors="pt").to(device)

few_generated_ids = model.generate(
	few_model_inputs.input_ids,
	**generation_config
)

# 截掉输入
few_generated_ids = [
	output_ids[len(input_ids):] for input_ids, output_ids in zip(few_model_inputs.input_ids, few_generated_ids)
	# zip配对，：左切片
]

# 解码
few_response = tokenizer.batch_decode(few_generated_ids,skip_special_tokens=True)[0]
# print("few shot test"+"-"*20)
# print(few_response)

#测试CoT-----------------------------------------------------------------
cot_messages_sentiment = [
    {"role": "system", "content": "You can help with sentiment classification"},
    {"role": "user", "content": """
     请分析文本的情感倾向，按照以下步骤思考：
     1. 提取关键词
     2. 分析关键词的情感含义
     3. 给出最终结果
     ---
	文本：我很开心
	分析: 关键词开心，很都表达了积极的情绪
	结果：积极
	---
	文本：我很难过
	分析: 关键词难过，很都表达了积极的情绪
	结果：消极
	---
	文本：我想睡觉
	分析:""" # prompt对结尾空格敏感
     }
]
cot_text_sentiment = tokenizer.apply_chat_template(
	cot_messages_sentiment,
	tokenize=False,
 	add_generation_prompt=True
)
cot_model_inputs = tokenizer([cot_text_sentiment],return_tensors="pt").to(device)

cot_generated_ids = model.generate(
	cot_model_inputs.input_ids,
	**generation_config
)

# 截掉输入
cot_generated_ids = [
	output_ids[len(input_ids):] for input_ids, output_ids in zip(cot_model_inputs.input_ids, cot_generated_ids)
	# zip配对，：左切片
]

# 解码
cot_response = tokenizer.batch_decode(cot_generated_ids,skip_special_tokens=True)[0]
print("cot shot test"+"-"*20)
print(cot_response)