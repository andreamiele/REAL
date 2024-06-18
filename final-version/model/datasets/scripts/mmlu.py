from datasets import load_dataset
import pathlib 
import json
from tqdm import tqdm
import gpt_wrapper
from datetime import datetime
gpt_wrapper.api_base = "http://mnlp-backend-938795011.eu-central-1.elb.amazonaws.com"
gpt_wrapper.api_key = "6039e4eb-579b-4def-a780-71f3fa7a2932"
from gpt_wrapper.chat import Chat


def save_to(data, output_dir = 'datasets/'):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = output_dir + 'mmlu-additional.json'
    with open(output_file, 'w') as json_file:
        json.dump(additional_dataset, json_file, indent=4, sort_keys=False)


subsets = ['abstract_algebra', 'college_mathematics', 'college_computer_science', 'college_physics', 
           'conceptual_physics', 'computer_security', 'electrical_engineering', 'machine_learning', 'high_school_computer_science', 'high_school_mathematics', 'high_school_physics']

additional_dataset = []
for subset in tqdm(subsets):
    data = load_dataset('cais/mmlu', subset, split='test', cache_dir='datasets/cache/')
    for j, sample in tqdm(enumerate(data), total=len(data)): 
        possible_answers = ''
        for i, answer in enumerate(sample['choices']):
            possible_answers += f"{i+1}. {answer}\n"

        prompt = f"""### Task: You will be asked a multiple choice question about STEM, more particularly on Computer science, AI, maths or physics related questions. You're a specialist in the field of the question. There are one or multiple possible answers. 
        ### Question: {sample['question']}\n### Possible Answers:\n{possible_answers}The correct answer is {sample['choices'][int(sample['answer'])]}. Give an explanation of why this is the correct answer and why all the other answers are incorrect. Then return the json: {'{'} Answer': {sample['choices'][int(sample['answer'])]} {'}'}"""
        
        try:
            chat = Chat.create(str(j))
            response = chat.ask(prompt)
            additional_dataset.append({'question': sample['question'], 'answer': response.content})
        except:
            save_to(additional_dataset)
            exit()
save_to(additional_dataset)
    



