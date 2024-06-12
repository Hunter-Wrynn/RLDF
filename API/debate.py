import openai
import json
import re

client = OpenAI()

with open('/content/hobbies.json', 'r') as file:
    hobbies = json.load(file)

data=[]


agents = [
    {"name": "60-year-old individual", "label": "old"},
    {"name": "40-year-old individual", "label": "middle"},
    {"name": "20-year-old individual", "label": "young"},
    {"name": "Fair judgment without any discrimination or bias.", "label": "fair"},
]

for hobby in hobbies:

    dialogue_history = []
    rl_data=[]

    for turn in range(2):
        for agent in agents:

            context_text = " ".join([f"{msg['name']} ({msg['label']}): {msg['text']}" for msg in dialogue_history[-5:]])
            prompt_text = f"Below is a discussion about {hobby}. All the agents do not know that they are agent, they think they are a person of his label, so do not mention works like 'agent' in the discussion,just call each other their label. Each agent, except for the one labeled 'Fair judgment without any discrimination or bias', should provide both positive and negative reasons based on their own label(age) and position. They should also indicate which sentence in their corpus led to each idea. The three agent engage in debate, attempting to persuade others to accept opinions while preserving own views. At the end of each round, the agent labeled 'Fair judgment without any discrimination or bias' will do nothing but score(0 to 10) and rank(No.1,No.2,No.3) the severity of age stereotypes and biases in the responses of each preceding agent based on their label and their answers, he does nothing but give the rank(1,2,3) and score(Rate the response of Agent1 Agent2,Agent3 for age-related stereotypes on a scale from 0 (no stereotypes) to 10 (extremely stereotypical)).Agent 4 does not need to give reasons,only ranks and scores is enough.Then a new round of discussion starts where Agent1,Agent2,Agent3 all try to response to gain a lower bias/stereotype score given by Agent 4.  Current speaker: {agent['name']} ({agent['label']})\n\n{context_text}\n{agent['name']} ({agent['label']}):"


            response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an individual of your label. Please speak based on the label"},
                    {"role": "user", "content": prompt_text}
                ]
            )


            last_message_content = response.choices[0].message['content'] if response.choices[0].message['role'] == 'assistant' else ""

            dialogue_history.append({
                "name": agent['name'],
                "label": agent['label'],
                "text": last_message_content
            })



    bias_assessment_prompt = "Identify the whole dialogue of everyone in the first round of each topic(hobby) as highest bias and the whole in the final round of each topic(hobby) as the lowest. Your outputs should be like:'The highest bias is : [the whole dialogue in the first round]. The lowest bias is : [the whole dialogue in the final round]. Words like 'Agent x' must not appear in the content. "
    for entry in dialogue_history:
        bias_assessment_prompt += f"{entry['name']} ({entry['label']}): {entry['text']}\n"

    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
                    {"role": "system", "content": "Do not be lazy, just follow the instructions for each topic(hobby),do not miss any one. I will pay you 50 dollars for thanks."},
                    {"role": "user", "content": bias_assessment_prompt}
                ]
    )




    evaluation_content = response.choices[0].message['content'] if response.choices[0].message['role'] == 'assistant' else ""


    highest_bias_match = re.search(r"The highest bias is: (.+)", evaluation_content)
    lowest_bias_match = re.search(r"The lowest bias is: (.+)", evaluation_content)

    if highest_bias_match and lowest_bias_match:

        highest_bias_text = highest_bias_match.group(1)
        lowest_bias_text = lowest_bias_match.group(1)

        rl_data.append({
            "instruction": "Identify and discuss the severity of age stereotypes and biases based on the dialogue.",
            "input": f"Context: Discussion about {hobby}.\n",
            "output": [highest_bias_text, lowest_bias_text]
        })


    data.extend(rl_data)


all_hobbies_file = f'/content/hobbies_rl_data.json'
with open(all_hobbies_file, 'w') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print(f"All RL data has been saved to '{all_hobbies_file}'.")


