from transformers import AutoTokenizer, RagRetriever, RagSequenceForGeneration, RagTokenForGeneration, RagTokenizer, pipeline
import torch
from datasets import load_dataset
import json 
import os
from tqdm import tqdm
import pandas as pd


####### Questions for retrieval ####### 

print('M1 formatted -> extracting questions')

file_path = 'data/M1_reformatted.json'

df = pd.read_json(file_path)
unique_questions = set(df['question'].unique())

print('number of questions: ', len(unique_questions))
# print("First 3 questions in the set:")
# for question in list(unique_questions)[:3]:
#     print(question)

####################################################################################################


######## Load the dataset and model ########

print('Loading the dataset and model for retrieval')


rag_dataset = load_dataset("wiki_dpr", 'psgs_w100.multiset.compressed', split='train', cache_dir='cache', trust_remote_code=True)
print('Dataset loaded')
retriever = RagRetriever.from_pretrained("facebook/rag-token-nq", indexed_dataset=rag_dataset)
print('Retriever loaded')
rag_model = RagSequenceForGeneration.from_pretrained('facebook/rag-token-nq', indexed_dataset=rag_dataset)
print('Model loaded')
rag_tokenizer = AutoTokenizer.from_pretrained("facebook/rag-token-nq")
print('Tokenizer loaded')
summarizer = pipeline('summarization', max_length=80)
print('Summarizer loaded')


####################################################################################################


######## Retrieve documents for each question ########  

def get_documents(prompt):
    inputs = rag_tokenizer(prompt, max_length=80, padding=True, return_tensors='pt')
    input_ids = inputs['input_ids']
    question_hidden_states = rag_model.question_encoder(input_ids)[0]
    docs_dict = retriever(input_ids.numpy(), question_hidden_states.detach().numpy(), return_tensors='pt')
    doc_ids = docs_dict["doc_ids"]
    doc_embeds = docs_dict["retrieved_doc_embeds"].numpy()  # Get document embeddings
    
    # Check the number of retrieved documents
    num_docs = len(doc_ids)
    documents = []
    embeddings = []
    for i in range(min(3, num_docs)):  # Retrieve up to 3 documents and their embeddings
        doc_text = rag_dataset[doc_ids[i]]['text']
        doc_embedding = doc_embeds[i].tolist()
        doc_title = rag_dataset[doc_ids[i]]['title']
        doc_id = doc_ids[i]

        for text, title, id, embedding in zip(doc_text, doc_title, doc_id, doc_embedding):
            documents.append({
                'id': str(id.item()), 
                'text': text, 
                'title': title, 
                'embeddings': embedding
            })
    return documents


# Prepare data to be saved
retrieved_documents = []
save_path = 'documents/retrieved_documents_bis.json'

# Process each unique question
for question in tqdm(unique_questions, total=len(unique_questions)):
    try:
        documents = get_documents(question)
        retrieved_documents.append({'question': question, 'documents': documents})
    except Exception as e:
        print(f"Error processing question: {e}")
        continue

# Save the retrieved documents to a JSON file
with open(save_path, 'w') as f:
    json.dump(retrieved_documents, f, indent=4)

print(f"Retrieved documents saved to {save_path}")