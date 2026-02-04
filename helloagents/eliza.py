import re
import random

# 定义规则库，rules是一个字典，键是正则表达式，值是列表
rules = {
    r'I need (.*)': [
        "Why do you need {0}? I know you are already {age} years old",
        "Would it really help you to get {0}?",
        "Are you sure need {0}?"
    ],
    r'Why don\'t you (.*)\?': [
        "Do you really think I don't {0}?",
        "Perhaps eventually I will {0}.",
        "Do you really want me to {0}?"
    ],
    r'Why can\'t I (.*)\?': [
        "Do you think you should be able to {0}?",
        "If you could {0}, what would you do?",
        "I don't know -- why can't you {0}?"
    ],
    r'I am (.*)': [
        "Do you come to me because you are {0}?",
        "How long have you been {0}?",
        "How do you feel about being {0}, {name}?"
    ],
    r'.* mother .*': [
        "Tell me more about your mother.",
        "What was your relationship with your mother like?",
        "How do you feel about your mother?"
    ],
    r'.* father .*': [
        "Tell me more about your father.",
        "How did your father make you feel?",
        "What has your father taught you, {name}?"
    ],
    r'.* work .*': [
        "How do you feel about your work? As a {job}",
        "Do you like your work?",
        "Can you tell me about your work?"
    ],
    r'.*': [
        "Please tell me more.",
        "Let's change focus a bit...Tell me about your family.",
        "Can you elaborate on that?"
    ]
}

# 定义代词转换规则，dict
pronoun_swap = {
    "i": "you", "you": "i", "me": "you", "my": "your",
    "am": "are", "are": "am", "was": "were", "i'd": "you would",
    "i've": "you have", "i'll": "you will", "yours": "mine", "mine": "yours"
}


def swap_pronouns(phrase):
    words = phrase.lower().split()
    swapped_words = [pronoun_swap.get(word, word)
                     for word in words]  # List Comprehension
    return " ".join(str(word) for word in swapped_words)


def respond(user_input):
    """
    根据规则库生成响应
    """
    for attribute, templates in attributes.items():
        for template in templates:
            match = re.search(template, user_input, re.IGNORECASE)
            if match:
                info = match.group(1) if match.groups() else ' '
                user_profile[attribute] = info
                print("ok, so I know you more now")

    for pattern, responses in rules.items():
        match = re.search(pattern, user_input, re.IGNORECASE)
        if match:
            captured_group = match.group(1) if match.groups() else ' '
            swapped_group = swap_pronouns(captured_group)
            response = random.choice(responses)
            try:
                return response.format(swapped_group, **user_profile)
            except (KeyError, IndexError):
                return response.format(swapped_group)

    return random.choice(rules[r'.*'])  # 字典索引


# 自定义记忆库
user_profile = {"name": "friend", "age": "30", "job": "student"}

attributes = {
    "name": [r'i am called (.*)',
             r'my name is (.*)',
             r'you can call me (.*)'],
    "age": [r'i am (.*) years old',
            r'my age is (.*)'],
    "job": [r'i am a (.*)',
            r'i work as a (.*)',
            r'i am doing a (.*)']
}

if __name__ == '__main__':
    print("Therapist: Hello! How can I help you today?")
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["quit", "exit", "bye"]:
            print("Therapist: Goodbye. It was nice talking to you.")
            break
        response = respond(user_input)
        print(f"Therapist: {response}")
