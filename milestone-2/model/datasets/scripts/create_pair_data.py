import pandas as pd
from datasets import load_dataset
from gpt_wrapper.chat import Chat
import json
from datetime import datetime
from pathlib import Path

# Initialize paths and constants
SCIPER = "302925"
data_path = Path(f'data/{SCIPER}')
data_path.mkdir(exist_ok=True, parents=True)
generation_kwargs = {"max_new_tokens": 128, "padding": True, "truncation": True, "no_repeat_ngrams": 2,
                     "frequency_penalty": 0, "do_sample": True}

# Load the MathQA dataset
dataset = load_dataset("allenai/math_qa")
questions = dataset['train']

all_data = []

# Iterate over the first 5 questions in the dataset
for idx, entry in enumerate(questions):
    if idx >= 5:
        break

    try:
        question_id = entry['Problem']
        question_body = entry['Problem']
        question_options = entry['options']

        # Prepare prompts
        possible_answers = '\n'.join([f"{chr(65 + i)}. {option}" for i, option in enumerate(question_options)])
        prompt = f"### Task: You will be asked a multiple choice question about STEM, more particularly on computer science, AI, maths or physics related questions. You're a specialist in the field of the question. There are one or multiple possible answers. Your task is to answer the question to the best of your abilities by explaining if an answer is correct or incorrect and why. \
                    You must parse every possible answer and explain if it is correct or incorrect and why. \
                    You must complete the task step by step and give your final answer by completing the following json: {{'Correct Answer(s)': <>}}.\
                You must explain your reasoning process. Ensure that your answer is detailed, accurate, and logical, demonstrating a deep understanding of the topic. Remember, clarity, coherence, and accuracy are key components of a successful response.\
                        ### Question: {question_body}\n ### Possible Answers: \n{possible_answers}\n\nAnswer: "

        # Generate answers from two different chats
        chat_A = Chat.create(str(question_id))
        chat_B = Chat.create(str(question_id))
        response_A = chat_A.ask(prompt, model_args=generation_kwargs)
        response_B = chat_B.ask(prompt, model_args=generation_kwargs)

        # Ranking prompt
        prompt2 = f"""### Task: You will be given a question about STEM, more particularly on computer science, AI, maths or physics related questions. You are also provided with two answers. You must rank the answers based on the following criteria: correctness, relevance, clarity, completeness. 
        For example, if answer A is more correct than answer B, you should select correctness: A. If both are equivalent you can select AB. However, for the overall score, you must select only A or B. You must return a json in the format:
            "ranking_criteria": {{
                        "correctness": <>,
                        "relevance":  <>,
                        "clarity": <>,
                        "completeness": <>,
                        "other": "Conciseness: <>; Engagement: <>",
                        "overall": <>,
                        }}
        ### Question: {question_body}      
        ### Answer A: {response_A.content}
        ### Answer B: {response_B.content}
        Now return only the JSON response."""

        chat_ranking = Chat.create(str(question_id))
        ranking_response = chat_ranking.ask(prompt2, model_args=generation_kwargs)
        ranking_result = json.loads(ranking_response.content)

        # Determine chosen and rejected based on overall ranking
        chosen_answer = response_A.content if ranking_result['ranking_criteria']['overall'] == 'A' else response_B.content
        rejected_answer = response_B.content if ranking_result['ranking_criteria']['overall'] == 'A' else response_A.content

        # Collect all data in DPO format
        answer_data = {
            "prompt": question_body,
            "chosen": chosen_answer,
            "rejected": rejected_answer
        }

        all_data.append(answer_data)
    except Exception as e:
        print(f"Error processing question {question_id}: {e}")
        continue

# Save the results
with open(data_path / f'annotated_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json', 'w') as f:
    json.dump(all_data, f, indent=4)
