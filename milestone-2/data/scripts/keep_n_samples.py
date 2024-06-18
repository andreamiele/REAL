import json
# Define the path to your JSONL file
input_file_path = 'dpo_prepared/epfl_test_data.json'
output_file_path = 'dpo_prepared/epfl_test_data_100.json'

# Read the first 100 lines from the JSONL file
lines = []
with open(input_file_path, 'r') as file:
    for i, line in enumerate(file):
        if i < 100:
            lines.append(json.loads(line))
        else:
            break

# Save the collected lines into a JSON file
with open(output_file_path, 'w') as outfile:
    json.dump(lines, outfile, indent=4)

print(f'Saved first 100 lines to {output_file_path}')