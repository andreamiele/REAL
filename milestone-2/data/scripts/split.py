import pandas as pd
import json
from datasets import load_dataset
#df_dpo = pd.read_json('/scratch/izar/mouchel/project-m2-2024-real/data/dpo_prepared/dpo.json')
sft_test = pd.read_json('/scratch/izar/mouchel/project-m2-2024-real/data/sft_prepared/sft_test_split.json')
dpo = load_dataset('json', data_files='/scratch/izar/mouchel/project-m2-2024-real/data/dpo_prepared/all_dpo.json', split='train')

dpo_test_set = []
new_dpo_train_set = []

dpo = pd.DataFrame(dpo)

from tqdm import tqdm
for i, entry in tqdm(dpo.iterrows(), total=len(dpo)):
    question = entry['prompt']
    sample = {'prompt': question, 'chosen': entry['chosen'], 'rejected': entry['rejected']}
    if question in sft_test['question'].values and entry['chosen'] in sft_test['answer'].values:
        dpo_test_set.append(sample)
    else:
        new_dpo_train_set.append(sample)

print(len(sft_test))
print(len(dpo_test_set))
print(len(new_dpo_train_set))



with open('/scratch/izar/mouchel/project-m2-2024-real/data/dpo_prepared/dpo_train.json', 'w') as f:
    json.dump(new_dpo_train_set, f, indent=4)

with open('/scratch/izar/mouchel/project-m2-2024-real/data/dpo_prepared/dpo_test.json', 'w') as f:
    json.dump(dpo_test_set, f, indent=4)