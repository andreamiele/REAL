import pandas as pd 
import gpt_wrapper
from datetime import datetime
from termcolor import colored, cprint
gpt_wrapper.api_base = "http://mnlp-backend-938795011.eu-central-1.elb.amazonaws.com"
gpt_wrapper.api_key = "38ddc971-7d5a-4525-be76-ebdc490fd7b7"
from gpt_wrapper.chat import Chat
import json
from pathlib import Path
from utils import process_gpt_output
SCIPER = "369939"

Path(f'data/{SCIPER}').mkdir(exist_ok=True, parents=True)
generation_kwargs = {"max_new_tokens": 128, "padding": True, "truncation": True, "no_repeat_ngrams": 2, "frequency_penalty": 0, "do_sample": True,}
df = pd.read_json(f"data/{SCIPER}.json")

all_data = []
directory_path = Path(f"data/{SCIPER}")
file_list = [file.name for file in directory_path.iterdir() if file.is_file()]

question_ids = []

annotated_dict = {}
if file_list:
    last_file_annotated = sorted(file_list)[-1]
    with open(f"data/{SCIPER}/{last_file_annotated}", 'r') as json_file:
        data = json.load(json_file)

    for sample in data:
        try:
            for question in sample:
                question_id = question.get('question_id')
                if question_id is not None:
                    annotated_dict[question_id] = sample[0]
                    question_ids.append(question_id)
        except:
            question_ids.append(sample['question_id'])
            annotated_dict[sample['question_id']] = sample


for i, entry in df.iterrows():
    try:
        course_id = entry.course_id
        question_id = entry.question_id
        question_body = entry.question_body
        question_options = entry.question_options 
        if question_id in question_ids:
            print("YOU HAVE ALREADY ANNOTATED THIS QUESTION")
            cprint("QUESTION: " + question_body, 'light_blue')
            print("DO YOU WANT TO ANNOTATE AGAIN? (Y/N)")
            annotating_again = True
            ans = input()
            if ans.lower() == 'n':
                annotating_again = False
                all_data.append(annotated_dict[question_id])
                continue
        
        cprint("QUESTION: " + question_body, 'blue')
        if question_id in question_ids:
            print("YOU ")
            continue
        chat_A = Chat.create(str(question_id)) 
        chat_B = Chat.create(str(question_id)) 

        if question_options == None:
            prompt = "### Task: You will be asked a question about STEM, more particularly on Computer science, AI, maths or physics related questions. You're a specialist in the field of the question. Your task is to answer the question to the best of your abilities. You must complete the task step by step and give your final answer by completing the following json: {'Answer': <>}.\
                    You must explain your reasoning process.  Ensure that your answer is detailed, accurate, and logical, demonstrating a deep understanding of the topic. Remember, clarity, coherence, and accuracy are key components of a successful response.\
                        ### Question: " + question_body + "\n\nAnswer: "
        else:
            possible_answers = ''
            for i, answer in enumerate(question_options):
                possible_answers += f"{i+1}. {answer}\n"
            cprint("POSSIBLE ANSWERS: \n" + possible_answers, 'blue')

            prompt = "### Task: You will be asked a multiple choice question about STEM, more particularly on Computer science, AI, maths or physics related questions. You're a specialist in the field of the question. There are one or multiple possible answers. Your task is to answer the question to the best of your abilities by explaining if an answer is correct or incorrect and why. \
                    You must parse every possible answer and explain if it is correct or incorrect and why. \
                    You must complete the task step by step and give your final answer by completing the following json: {'Correct Answer(s)': <>}.\
                You must explain your reasoning process. Ensure that your answer is detailed, accurate, and logical, demonstrating a deep understanding of the topic. Remember, clarity, coherence, and accuracy are key components of a successful response.\
                        ### Question: " + question_body + "\n ### Possible Answers: \n" + possible_answers + "\n\nAnswer: "
            prompt2 = "### Task: You will be given a question about STEM, more particularly on Computer science and AI related questions. You are also provided with an answer. You must decide whether the answer is correct or incorrect and explain why in a step-by-step manner. \
                        Question: " + question_body + "\n ### Answer: " + question_options[0] + "\n\nAnswer: "
            prompt_answers = f"""### Task: You will be give a multiple choice question about computer science, physics or AI. With the question you will be given the choices for the question. Your task is to evaluate the answers and decide which are correct or incorrect I give you and answer if it is correct or incorrect. \
                ### Question: {question_body}\n ###Possible Answers: {possible_answers}\n\nReturn a json stating each choice, whether it is correct or incorrect: {'{'}"choice_1": <>, "choice_2": <>,  ... {'}'}"""
            
            response = chat_A.ask(prompt_answers, model_args=generation_kwargs)
            print(response.content)


        response_A = chat_A.ask(prompt, model_args=generation_kwargs)
        response_B = chat_B.ask(prompt, model_args=generation_kwargs)
        cprint("#"* 50 + "RESPONSE A" + "#"* 50, 'red')
        cprint(response_A, 'green')
        cprint("#"* 50 + "RESPONSE B" + "#"* 50, 'red')
        cprint(response_B, 'light_green')
        print("#"* 100)
        print(type(response_A.content))
        prompt2 = f"""### Task: You will be given a question about STEM, more particularly on Computer science, AI, maths or physics related questions. You are also provided with two answers. You must rank the answers based on the following criteria: correctness, relevance, clarity, completeness. 
        For example, if answer A is more correct than answer B, you should select correctness: A. If both are equivalent you can select AB. However, for the overall score, you must select only A or B. You must return a json in the format:
            "ranking_criteria": {'{'}
                        "correctness": <>,
                        "relevance":  <>,
                        "clarity": <>,
                        "completeness": <>,
                        "other": "Conciseness: <>; Engagement: <>"
                        "overall": <>,
                        {'}'}"

        ### Question: {question_body}      
        ### Answer A: {response_A.content}
        ### Answer B: {response_B.content}
        Now return only the JSON response.    
            """
        chat_ranking = Chat.create(str(question_id))
        ranking_response = chat_ranking.ask(prompt2, model_args=generation_kwargs)
        print("#"* 50 + "RANKING RESPONSE" + "#"* 50)
        print(ranking_response)
        print(process_gpt_output(ranking_response.content))
        answer = {
                    "course_id": course_id, 
                    "question_id": question_id, 
                    "question" : question_body, 
                    "A_chat_id": response_A.chat_id, 
                    "B_chat_id": response_B.chat_id,
                    "A": response_A.content,
                    "B": response_B.content,
                    "ranking_criteria": {
                        "overall": input("Overall: "),
                        "correctness": input("Correctness: "),
                        "relevance":  input("Relevance: "),
                        "clarity": input("Clarity: "),
                        "completeness": input("Completeness: "),
                        "other": f"Conciseness: {input('Conciseness:')}; Engagement: {input('Engagement:')}"
                        }
                },
    
        all_data.append(answer)
    except KeyboardInterrupt:
        break
    

with open(f'data/{SCIPER}/annotated_{datetime.now()}.json', 'w') as f:
    json.dump(all_data, f, indent=4)