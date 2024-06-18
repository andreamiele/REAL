import json
from datasets import load_dataset
from datasets import concatenate_datasets
from datasets import Dataset
from os import listdir
from os.path import isfile, join
import itertools
import random

# Input file
# The script also uses all other json files in the folder as datasets
course_data_file = "M1_preference_data_15052024.json"
# Output files

folder = "dpo_prepared/"
dpo_output_file = folder + "all_dpo.json"
dpo_small_file = folder + "dpo_small.json"
small_size = 30


min_word_limit = 20
max_answers_per_question = 4

# train_file = "dpo_prepared/dpo_train_split.json"
# test_file = "dpo_prepared/dpo_test_split.json"

def word_count(text):
    return len(text.split())

"""
Parse the course data file.
Input: Course data file name.
Output: List of dictionaries containing the prompt, chosen answer and rejected answer. 
        First returning argument is for training, the second is the remaining pairs.
Selection is based on best overall performance.
"""
def parseCourseData(course_data_file):
    print("Loading: " + course_data_file)
    with open(course_data_file) as f:
        file_contents = f.read()
    
    parsed_json = json.loads(file_contents)

    pairs = []
    remaining_pairs = []
    for q in parsed_json:
        i = 0
        random.shuffle(q['preference'])
        for pref in q["preference"]:
            # Select based on overall preference
            pref_answer = pref[pref["overall"]]
            if pref_answer is None:
                exit("Error, no preferred answer found for a given entry.")
            dispref_answer = pref["A"] if pref["overall"] == "B" else pref["B"] # Dispreferred is the other option
            if dispref_answer is None:
                exit("Error, no rejected answer found for a given entry.")
            pair_dict = {'prompt': q["question_complete"], 'chosen': pref_answer, 'rejected': dispref_answer}

            if(word_count(pair_dict['chosen']) >= min_word_limit and word_count(pair_dict['rejected']) >= min_word_limit):
                if i >= max_answers_per_question:
                    remaining_pairs.append(pair_dict)
                else:
                    pairs.append(pair_dict)
                    i+=1

    return pairs, remaining_pairs
    

"""
Parse external datasets from Huggingface.
Input: Huggingface dataset name, name of the question column and name of the answer column in the dataset.
Output: Dataset containing the question and answer.
"""
def load_external_dataset(name, question_column, chosen_column, rejected_column):
    print("Loading and parsing: " + name)
    math_pref = load_dataset(name, split='train')
    math_pref_parsed = []
    for entry in math_pref: 
        question = entry[question_column]
        chosen_response = entry[chosen_column]
        dispref_answer = entry[rejected_column]
        math_pref_parsed.append({'prompt': question, 'chosen': chosen_response, 'rejected': dispref_answer})

    return math_pref_parsed

"""
Save a dataset to jsonlines file.
The dataset is expected to be a list of dictionaries.
Input: dataset and file name to save the dataset to.
"""
def write_jsonlines(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


datasets = {}
datasets["epfl-data"], epfl_test_data = parseCourseData(course_data_file)
datasets["argilla-distilabel-math-preference-dpo"] = load_external_dataset('argilla/distilabel-math-preference-dpo', 'instruction', 'chosen_response', 'rejected_response')

# Add chatgpt-dpo-data
with open("chatgpt-dpo-data.json", 'r') as f:
    print("Loading and parsing: chatgpt-dpo-data.json")
    extra_data = json.load(f)
    datasets['gpt-dpo-data'] = extra_data

total_data = list(itertools.chain.from_iterable(datasets.values()))

for name, ds in datasets.items():
    random.shuffle(ds)
    write_jsonlines(ds, folder + name + ".json")

random.shuffle(epfl_test_data)
write_jsonlines(epfl_test_data, folder + "epfl_test_data" + ".json")

random.shuffle(total_data)
write_jsonlines(total_data, dpo_output_file)

# Create a smaller file for testing
#subset = random.sample(total_data, small_size)
subset=total_data[-small_size:]
write_jsonlines(subset, dpo_small_file)


