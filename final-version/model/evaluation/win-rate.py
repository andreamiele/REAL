from datetime import datetime
from tqdm import tqdm
from datasets import load_dataset
import pathlib 
import json
from tqdm import tqdm
import gpt_wrapper
import argparse
gpt_wrapper.api_base = "http://mnlp-backend-938795011.eu-central-1.elb.amazonaws.com"
gpt_wrapper.api_key = "6039e4eb-579b-4def-a780-71f3fa7a2932"
from gpt_wrapper.chat import Chat


def parse_args():
    parser = argparse.ArgumentParser() 
    parser.add_argument('--test-path', type=str, default='data/dpo_prepared/dpo_small.json')
    parser.add_argument('--reference-generations-path', default='data/inference/reference.json', type=str) ### the json should be a list of the responses for question from the same test set
    parser.add_argument('--policy-generations-path', default='data/inference/policy,json', type=str)
    
    return parser.parse_args()

def open_json(path):
    with open(path, 'r') as f:
        return json.load(f)
    
def process_gpt_output(output_text):
    try:
        result_dict = json.loads(output_text)
        return result_dict
    except json.JSONDecodeError:
        start_index = output_text.find('{')
        end_index = output_text.rfind('}')
        if start_index != -1 and end_index != -1:
            extracted_dict_text = output_text[start_index:end_index + 1]
            try:
                result_dict = json.loads(extracted_dict_text)
                return result_dict
            except json.JSONDecodeError:
                print("Failed to extract a valid dictionary from the text.")
                return None
        else:
            print("No dictionary found in the text.")
            return None
        
def main():
    args = parse_args()
    test_data = open_json(args.test_path) ## the test data can 
    reference_generations = open_json(args.reference_generations_path)
    policy_generations = open_json(args.policy_generations_path)

    
    for i, sample in enumerate(test_data):
        question = sample['prompt']
        reference = reference_generations[i]
        policy = policy_generations[i] 

        chat = Chat.create(str(i))

        prompt = f"""You will be asked a question about STEM, more particularly on Computer science, AI, maths or physics related questions. You're a specialist in the field of the question.
        To the best of your abilities, you must select one of the following answers based on correctness and explanation. If both answers are equivalent, you may return 'tie', as number 3.
        ### Question: {question}\n 
        ### Answer 1: {reference}\n
        ### Answer 2: {policy}\nThe best answer is number ..."""

        response =  chat.ask(prompt, max_tokens=1)
        try:
            answer = int(response)
        except:
            exit()
        parsed = process_gpt_output(response.content)
        print(answer) 
        results = {'reference': 0, 'policy': 0, 'tie': 0}
        if answer == 1:
            results['reference'] += 1
        elif answer == 2:
            results['policy'] += 1
        elif answer == 3:
            results['tie'] += 1

        print(results)
        save_to(results, output_dir = 'data/inference/')




def save_to(data, output_dir = 'datasets/'):
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    output_file = output_dir + 'win-rate.json'
    with open(output_file, 'w') as json_file:
        json.dump(data, json_file, indent=4, sort_keys=False)





