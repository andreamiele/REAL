import json
from datasets import load_dataset, Dataset
import random

# Input file
course_data_file = "M1_preference_data_15052024.json"

# Output files
train_file = "dpo_prepared/epfl_train_data_.json"
test_file = "dpo_prepared/epfl_test_data_.json"

# Minimum word limit for the answers
min_word_limit = 20


def word_count(text):
    return len(text.split())


def parseCourseData(course_data_file):
    print("Loading: " + course_data_file)
    with open(course_data_file) as f:
        file_contents = f.read()

    parsed_json = json.loads(file_contents)

    pairs = []
    for q in parsed_json:
        i = 0
        random.shuffle(q['preference'])
        for pref in q["preference"]:
            # Select based on overall preference
            pref_answer = pref[pref["overall"]]
            if pref_answer is None:
                exit("Error, no preferred answer found for a given entry.")
            dispref_answer = pref["A"] if pref["overall"] == "B" else pref["B"]  # Dispreferred is the other option
            if dispref_answer is None:
                exit("Error, no rejected answer found for a given entry.")
            pair_dict = {'prompt': q["question_complete"], 'chosen': pref_answer, 'rejected': dispref_answer}

            if (word_count(pair_dict['chosen']) >= min_word_limit and word_count(
                    pair_dict['rejected']) >= min_word_limit):

                pairs.append(pair_dict)


    return pairs


def write_jsonlines(data, filename):
    with open(filename, 'w') as f:
        for item in data:
            f.write(json.dumps(item) + "\n")


# Load and parse the course data
epfl_data = parseCourseData(course_data_file)

# Shuffle the data
random.shuffle(epfl_data)

# Split into 80/20 train/test
split_index = int(len(epfl_data) * 0.8)
train_data = epfl_data[:split_index]
test_data = epfl_data[split_index:]

# Save the train and test datasets
write_jsonlines(train_data, train_file)
write_jsonlines(test_data, test_file)

print(f"Train data saved to {train_file}")
print(f"Test data saved to {test_file}")