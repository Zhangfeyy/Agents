import collections

corpus = "datawhale agent learns datawhale agent works"
tokens = corpus.split()
total_tokens=len(tokens)

# 1. 计算P(datawhale)
count_datawhale = tokens.count('datawhale')
p_datawhale = count_datawhale / total_tokens

# 2. 计算P(agent|datawhale)
"""
这行代码将tokens（分词后的列表）
和它自身向后偏移一位的列表配对，生成所有相邻的二元组（bigrams）
"""
bigrams = zip(tokens,tokens[1:])
"""
这行代码统计每种二元组在bigrams中的出现次数，
结果是一个Counter对象，键是二元组，值是出现次数
"""
bigrams_counts = collections.Counter(bigrams)
count_datawhale_agent = bigrams_counts[('datawhale','agent')]

p_agent_given_datawhale = count_datawhale_agent / count_datawhale

# 3. 计算P(learns|agent)
count_agent_learns = bigrams_counts[('agent','learns')]
count_agent = tokens.count('agent')
p_learns_given_agent = count_agent_learns / count_agent

# 4. 连乘
p_sentence = p_datawhale * p_agent_given_datawhale * p_learns_given_agent
print(p_sentence)