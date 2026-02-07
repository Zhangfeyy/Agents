import re
import collections


def get_stats(vocab):
    pairs = collections.defaultdict(int)
    for word, freq in vocab.items():
        # 按空格切分
        symbols = word.split()
        for i in range(len(symbols)-1):
            pairs[symbols[i], symbols[i+1]] += freq # 初始值为0，键是tuple

    return pairs


def merge_vocab(pair, v_in):
    v_out = {}
    bigram = re.escape(' '.join(pair)) # re.escape 转义
    p = re.compile(r'(?<!\S)' + bigram + r'(?!\S)') # 创建pattern, 前后不是非空白字符
    for word in v_in:
        w_out = p.sub(''.join(pair), word) #替换
        v_out[w_out] = v_in[word]
    return v_out


# 在每个词词尾加上</w>表示结束，并切分好字符
vocab = {'h u g </w>': 1, 'p u g </w>': 1, 'p u n </w>': 1, 'b u n </w>': 1}
num_merges = 4

for i in range(num_merges):
    pairs = get_stats(vocab)
    if not pairs:
        break
    best = max(pairs, key=pairs.get)
    # arg1：比较对象，arg2： 比较标准
    vocab= merge_vocab(best,vocab)
    print(f"第{i+1}次合并：{best}->{''.join(best)}")
    print(f"新词表：{list(vocab.keys())}")
    print("-"*20)
    
    
