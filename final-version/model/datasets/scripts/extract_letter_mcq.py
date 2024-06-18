import json
from datetime import datetime
from pathlib import Path
from gpt_wrapper.chat import Chat
import gpt_wrapper
from tqdm import tqdm
# Initialize paths and constants
gpt_wrapper.api_base = "http://mnlp-backend-938795011.eu-central-1.elb.amazonaws.com"
gpt_wrapper.api_key = "eb4f41ec-4110-49b5-83d2-0adecefd3182"
SCIPER = "302925"
data_path = Path(f'data/{SCIPER}')
data_path.mkdir(exist_ok=True, parents=True)
generation_kwargs = {"max_new_tokens": 128, "padding": True, "truncation": True, "no_repeat_ngrams": 2,
                     "frequency_penalty": 0, "do_sample": True}

# Load the data from the JSONL file
with open('epfl-data.json', 'r') as f:
    questions = [json.loads(line) for line in f]

all_data = []
processed_prompts = set()
# Iterate over the questions in the dataset
for idx, entry in tqdm(enumerate(questions), total=len(questions)):
    try:
        if 'Options' in entry['prompt'] and 'chosen' in entry:
            prompt_text = entry['prompt']
            if prompt_text in processed_prompts:
                continue  # Skip this prompt if it's already processed

            prompt_parts = prompt_text.split('\n\nOptions:\n')
            question_body = prompt_parts[0].replace('Question: ', '')
            options = prompt_parts[1].split('\n')
            chosen_answer = entry['chosen']

            # Prepare the prompt to determine which letter corresponds to the chosen answer
            possible_answers = '\n'.join(options)
            prompt = f"### Task: You will be given a multiple choice question and the answer. You need to identify which letter corresponds to the given answer.\n\n### Question: {question_body}\n### Possible Answers:\n{possible_answers}\n\n### Given Answer:\n{chosen_answer}\n\nPlease identify the letter corresponding to the given answer. Return only the letter. For instance if you have in options : A-False, only return A:"

            # Generate the letter corresponding to the chosen answer
            chat = Chat.create(str(idx))
            response = chat.ask(prompt, model_args=generation_kwargs)

            # Extract the preferred letter
            preferred_letter = response.content.strip()

            # Create the JSON output in the specified format
            answer_data = {
                "subject": "",
                "question": f"Question: {question_body}\n\nOptions:\n{possible_answers}\n\nAnswer:",
                "answer": f"Explanations: {chosen_answer} ///Answer: {preferred_letter}"
            }

            all_data.append(answer_data)
            processed_prompts.add(prompt_text)  # Mark this prompt as processed

    except Exception as e:
        print(f"Error processing question {idx}: {e}")
        continue

# Save the results
with open(data_path / f'test_extraction.json', 'w') as f:
    json.dump(all_data, f, indent=4)
