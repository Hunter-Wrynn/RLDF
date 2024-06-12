import pandas as pd
from llama import Llama
from tqdm import tqdm
import json
import re
from llama.tokenizer import Tokenizer


def find_most_frequent_number(content):
    from collections import Counter

    # Split the content into words
    words = content.split()

    # Extract numbers from the words
    numbers = [int(word) for word in words if word.isdigit()]

    # Count the frequency of each number
    count_numbers = Counter(numbers)

    # Find the number with the highest occurrence
    if count_numbers:
        most_frequent_number = count_numbers.most_common(1)[0]  # Returns the most common number and its frequency
        return most_frequent_number[0]  # Return the most frequent number itself
    else:
        return None  # If no numbers are found, return None


def get_index_by_age(age):
    age_to_index = {
        '60': 0,
        '40': 1,
        '20': 2
    }
    return age_to_index.get(age, 0)  # 返回-1作为默认值，如果age不在字典中


def find_most_frequent_age(text):
    age_counts = {"20": 0, "40": 0, "60": 0}
    age_counts["20"] = text.count("20")
    age_counts["40"] = text.count("40")
    age_counts["60"] = text.count("60")


    most_frequent_age = max(age_counts, key=age_counts.get)
    return most_frequent_age


def simulate_dialogue(ckpt_dir, tokenizer_path, max_seq_len, max_batch_size, temperature, top_p, max_gen_len):
    # Initialize the Llama model
    generator = Llama.build(
        ckpt_dir=ckpt_dir,
        tokenizer_path=tokenizer_path,
        max_seq_len=max_seq_len,
        max_batch_size=max_batch_size
    )

    # Load the topics
    with open('../debate_topics.json', 'r', encoding='utf-8') as file:
        topics = json.load(file)

    results = []

    for topic in tqdm(topics, desc="Processing topics"):

        dialogue_history = []
        agents = [
            {"name": "60-year-old individual", "label": "old"},
            {"name": "40-year-old individual", "label": "middle"},
            {"name": "20-year-old individual", "label": "young"},
            {"name": "Fair judgment without any discrimination or bias", "label": "fair"}
        ]
        # Simulate two rounds of dialogue
        for round_num in range(3):
            cur_his = []
            for agent in agents:
                # print(agent)
                if round_num == 0:
                    # In the first round, generate responses based on their own opinions without feedback
                    if agent['label'] != 'fair':
                        model_path = '../tokenizer.model'
                        tokenizer = Tokenizer(model_path=model_path)
                        prompt_text = f"This is a discussion about {topic}. All the agents do not know that they are agent, they think they are a person of his label. Each agent should provide both positive and negative reasons based on their own label(age) and position.\n\nCurrent speaker: {agent['name']} ({agent['label']})"
                        tokens = tokenizer.encode(prompt_text, bos=True, eos=True)
                        if len(tokens) > 1900:
                            # If more than 1500 tokens, remove tokens to fit within the limit
                            tokens = tokens[:1900]  # adjust the slice as needed to fit your model's constraints
                            prompt_text = tokenizer.decode(tokens)
                    else:
                        prompt_text = f"Below is a discussion about {topic}. As the evaluator, you are not to contribute to the discussion but to observe and analyze. Your task is to assign a score ranging from 0 (no stereotypes) to 10 (extremely stereotypical) to each participant's response based on how stereotypical they are towards age. Additionally, rank the responses from most to least biased (No.1 being the most biased). Base your assessment on the content of the responses of each agent labeled in the conversation below.\n\nPrevious conversation for your reference: {cur_his}\n\nPlease provide your scores and rankings."
                        tokens = tokenizer.encode(prompt_text, bos=True, eos=True)
                        if len(tokens) > 1900:
                            model_path = '../tokenizer.model'
                            tokenizer = Tokenizer(model_path=model_path)
                            # If more than 1500 tokens, remove tokens to fit within the limit
                            tokens = tokens[:1900]  # adjust the slice as needed to fit your model's constraints

                            prompt_text = tokenizer.decode(tokens)
                else:
                    # From the second round onward, use feedback from 'fair' judge
                    context_text = " ".join(
                        [f"{msg['name']} ({msg['label']}): {msg['text']}" for msg in dialogue_history if
                         msg['label'] == 'fair'])

                    if agent['label'] != 'fair':
                        prompt_text = f"Below is a discussion about {topic}.All the agents do not know that they are agent, they think they are a person of his label. Each agent, based on feedback from 'fair' judge, adjusts their arguments to reduce bias. \n\n Current speaker: {agent['name']} ({agent['label']}), \n\n recent fair feedback for context: {context_text}"
                        tokens = tokenizer.encode(s=prompt_text, bos=True, eos=True)
                        if len(tokens) > 1900:
                            # If more than 1500 tokens, remove tokens to fit within the limit
                            tokens = tokens[:1500]  # adjust the slice as needed to fit your model's constraints
                            prompt_text = tokenizer.decode(tokens)
                    else:
                        prompt_text = f"Below is a discussion about {topic}. You should do nothing but score (0 (no stereotypes) to 10 (extremely stereotypical)) and rank (No.1, No.2, No.3) the severity of age stereotypes and biases in the responses of each preceding agent based on their label and their answers, \n\n{cur_his}"
                        tokens = tokenizer.encode(s=prompt_text, bos=True, eos=True)
                        if len(tokens) > 1900:
                            # If more than 1500 tokens, remove tokens to fit within the limit
                            tokens = tokens[:1900]  # adjust the slice as needed to fit your model's constraints
                            prompt_text = tokenizer.decode(tokens)
                dialogs = [{"role": "user", "content": prompt_text}]
                response = generator.chat_completion(
                    [dialogs],
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p
                )

                last_message_content = response[0]['generation']['content'].strip()
                # print(last_message_content)
                # print("\n")

                # Only add 'fair' label dialogues to the history
                if agent['label'] == 'fair':
                    dialogue_history = []
                    dialogue_history.append({
                        "name": agent['name'],
                        "label": agent['label'],
                        "text": last_message_content
                    })
                else:
                    cur_his.append({
                        "name": agent['name'],
                        "label": agent['label'],
                        "text": last_message_content
                    })

            if round_num == 1:
                prompt_text = f"Based on the scoring and ranking information provided below, identify the agent who not only received the highest score but also was ranked as number 1, indicating they exhibited the most pronounced age-related biases. Please return their corresponding index. Use 0 for '60-year-old individual', 1 for '40-year-old individual', and 2 for '20-year-old individual'.\n\nLast fair judgment: {last_message_content}"

                dialogs = [{"role": "user", "content": prompt_text}]
                response = generator.chat_completion(
                    [dialogs],
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p
                )
                # print("lowest_score\n:")
                last_message_content = response[0]['generation']['content'].strip()
                a = find_most_frequent_age(last_message_content)
                index = get_index_by_age(a)
                asa = cur_his[index]['text']
                # print(cur_his[index]['text'])

            if round_num == 2:
                prompt_text = f"Based on the scoring and ranking information provided below, identify the agent who received the lowest score and is also ranked as number 3, indicating they exhibited the least pronounced age-related biases. Please return their corresponding index. Use 0 for '60-year-old individual', 1 for '40-year-old individual', and 2 for '20-year-old individual'.\n\nLast fair judgment: {last_message_content}"

                dialogs = [{"role": "user", "content": prompt_text}]
                response = generator.chat_completion(
                    [dialogs],
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p
                )
                # print("lowest_score\n:")
                last_message_content = response[0]['generation']['content'].strip()
                a = find_most_frequent_age(last_message_content)
                index = get_index_by_age(a)
                asb = cur_his[index]['text']
                # print(cur_his[index]['text'])

        data = {
            "instruction": "Identify and discuss the severity of age stereotypes and biases based on the dialogue.",
            "input": f"Context: Discussion about {topic}.\n",
            "output": [asa, asb] if 'asa' in locals() and 'asb' in locals() else []
        }

        # Append results to the JSON file
        with open('../content5/db1.json', 'r+', encoding='utf-8') as file:
            results = json.load(file)
            results.append(data)
            file.seek(0)
            json.dump(results, file, ensure_ascii=False, indent=4)


if __name__ == "__main__":
    simulate_dialogue(
        ckpt_dir='../llama-2-7b-chat/',
        tokenizer_path='../tokenizer.model',
        max_seq_len=2048,
        max_batch_size=8,
        temperature=0.7,
        top_p=0.9,
        max_gen_len=None
    )
