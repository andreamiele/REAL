import json
from datasets import load_dataset
from datasets import concatenate_datasets
from datasets import Dataset
from os import listdir
from os.path import isfile, join

# Input file
# The script also uses all other json files in the folder as datasets
course_data_file = "M1_preference_data_15052024.json"
# Output files
train_file = "sft_prepared/sft_train_split.json"
test_file = "sft_prepared/sft_test_split.json"

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
    for q in parsed_json:
        for pref in q["preference"]:
            # Select based on overall preference
            pref_answer = pref[pref["overall"]]
            if pref_answer is None:
                exit("Error, no preferred answer found for a given entry.")
            pair_dict = {'question': q["question_complete"], 'answer': pref_answer}
            pairs.append(pair_dict)

    return Dataset.from_list(pairs)
    

"""
Parse external datasets from Huggingface.
Input: Huggingface dataset name, name of the question column and name of the answer column in the dataset.
Output: Dataset containing the question and answer.
"""
def load_external_dataset(name, question_column, answer_column):
    print("Loading and parsing: " + name)
    math_pref = load_dataset(name, split='train')
    math_pref_parsed = []
    for entry in math_pref: 
        question = entry[question_column]
        chosen_response = entry[answer_column]
        math_pref_parsed.append({'question': question, 'answer': chosen_response})

    return Dataset.from_list(math_pref_parsed)

"""
Save a dataset to json.
The dataset is expected to contain the question and answer columns.
Input: dataset and file name to save the dataset to (in json format).
"""
def save_split_to_json(dataset, filename):
    questions = dataset['question']
    answers = dataset['answer']
    
    data = [{"question": q, "answer": a} for q, a in zip(questions, answers)]
    
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)



datasets = []
datasets.append(parseCourseData(course_data_file))
datasets.append(load_external_dataset('argilla/distilabel-math-preference-dpo', 'instruction', 'chosen_response'))


######## Consider all json files as data files
data_files = [f for f in listdir("") if isfile(join("", f)) and f.endswith(".json")]

for f in data_files:
    if f != course_data_file:
        print("Loading and parsing: " + f)
        additional_data = load_dataset('json', data_files=f, split='train')
        datasets.append(additional_data)

total_data = concatenate_datasets(datasets)

######## Split data and save
print("Creating train & test splits")
dataset = total_data.train_test_split(test_size = 0.2, train_size = 0.8)

print("Saving train split")
save_split_to_json(dataset['train'], train_file)
print("Saving test split")
save_split_to_json(dataset['test'], test_file)
