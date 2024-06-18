import json
from datasets import load_dataset
from datasets import concatenate_datasets
from datasets import Dataset
from os import listdir
from tqdm import tqdm
from os.path import isfile, join
import gpt_wrapper
gpt_wrapper.api_base = "http://mnlp-backend-938795011.eu-central-1.elb.amazonaws.com"
gpt_wrapper.api_key = "38ddc971-7d5a-4525-be76-ebdc490fd7b7"
from gpt_wrapper.chat import Chat
import random

huggingface_read_token = "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"

# Input file
# The script also uses all other json files in the folder as datasets
course_data_file = "M1_preference_data_15052024.json"
# Output files
mcq_dataset_file = "mcq_prepared/mcq_dataset.json"

"""
Parse the course data file.
Input: Course data file name.
Output: Dataset containing the question and answer.
Selection is based on best overall performance.
"""
def parseCourseData(course_data_file):
    print("Loading: " + course_data_file)
    with open(course_data_file) as f:
        file_contents = f.read()
    
    parsed_json = json.loads(file_contents)

    pairs = []
    for q in tqdm(parsed_json):
        if("Options:" in q["question_complete"]):
            answers = []
            for i, pref in tqdm(enumerate(q["preference"]), leave=False):
                chatInstance = Chat.create(str(q["question_id"]) + "-" + str(i))
                # Select based on overall preference
                pref_answer = pref[pref["overall"]]
                if pref_answer is None:
                    exit("Error, no preferred answer found for a given entry.")

                prompt = "### Task: You will see a multiple choice question and an answer. Your task is very simple, out of the proposed answers you need to find which answer was actually chosen.\
                    To be clear, we are not asking you to answer the question yourself, just to find which answer was chosen. Reply using a single letter, A, B, C or D. Do not add any other information to your answer, just a single letter.\
                    The question that was asked was the following: " + q["question_complete"] + ". The chosen answer was the following: " + pref_answer + ".\
                    Which option was chosen? (remember, reply with only a single letter)"
                reply = chatInstance.ask(prompt)
                letter = reply.content[0]
                answers.append(letter)

            most_freq_answer = max(set(answers), key=answers.count)
            pair_dict = {'subject': '', 'question': q["question_complete"], 'answer': most_freq_answer}
            pairs.append(pair_dict)

    return Dataset.from_list(pairs)
    

"""
Parse external datasets from Huggingface.
Input: Huggingface dataset name, name of the question column, list of answer columns (including correct one), and name of the answer column in the dataset.
Output: Dataset containing the question with options appended and answer.
"""
def load_external_dataset(name, question_column, answer_columns, correct_column, config_name=None):
    print("Loading and parsing: " + name)
    ext_ds = load_dataset(name, split='train', token=huggingface_read_token, name=config_name)
    ext_ds_parsed = []
    for entry in ext_ds:
        answer_options = [entry[a] for a in answer_columns]
        random.shuffle(answer_options)
        options = [chr(65 + i) + ". " + o for i, o in enumerate(answer_options)]
        full_qstn = entry[question_column] + "\n\nOptions:\n" + "\n".join(options)
        correct_answer = chr(65 + answer_options.index(entry[correct_column]))

        ext_ds_parsed.append({'subject': '', 'question': full_qstn, 'answer': correct_answer})

    return Dataset.from_list(ext_ds_parsed)

"""
Save a dataset to jsonlines.
The dataset is expected to contain the question and answer columns.
Input: dataset and file name to save the dataset to (in json format).
"""
def write_jsonlines(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


datasets = []
# datasets.append(parseCourseData(course_data_file))


datasets.append(load_external_dataset('allenai/sciq', 'question', ['distractor1', 'distractor2', 'distractor3', 'correct_answer'], 'correct_answer'))
datasets.append(load_external_dataset('Idavidrein/gpqa', 'Question', ['Incorrect Answer 1', 'Incorrect Answer 2', 'Incorrect Answer 3', 'Correct Answer'], 'Correct Answer', "gpqa_extended"))

# ######## Consider all json files as data files
# data_files = [f for f in listdir(".") if isfile(join(".", f)) and f.endswith(".json")]

# for f in data_files:
#     if f != course_data_file:
#         print("Loading and parsing: " + f)
#         additional_data = load_dataset('json', data_files=f, split='train')
#         datasets.append(additional_data)

total_data = concatenate_datasets(datasets)

######## Split data and save
# print("Creating train & test splits")
# dataset = total_data.train_test_split(test_size = 0.2, train_size = 0.8)

print("Saving train split")
write_jsonlines(total_data, mcq_dataset_file)
