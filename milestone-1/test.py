import gpt_wrapper
import json
from gpt_wrapper.chat import Chat

#gpt_wrapper.api_base = ### INSERT HERE
#gpt_wrapper.api_key = ### INSERT HERE

def generate_dual_answers(questions_json):
    # Load the questions from the JSON file
    with open(questions_json, 'r') as file:
        questions = json.load(file)
    count =0
    for question in questions:
        if 'question_options' in question and question['question_options'] is not None:
            # Format the question with options if options are available
            content = question['question_body'] + "\nOptions: " + ", here are the options. generate the rationale behind its choice. ".join(question['question_options'])
        else:
            content = question['question_body']

        # Create two separate chat sessions
        chat_session1 = Chat.create("EPFL Course Questions Session 1")
        chat_session2 = Chat.create("EPFL Course Questions Session 2")

        # Generate the first answer from the first chat session
        answer1 = chat_session1.ask(content=content)

        # Generate the second answer from the second chat session
        answer2 = chat_session2.ask(content=content)

        # Print or store the responses
        print("Question ID:", question['question_id'])
        print("Question:", content)
        print("Answer 1:", answer1 if answer1 else "No response")
        print("Answer 2:", answer2 if answer2 else "No response")
        print("\n" + "-" * 80 + "\n")
        count += 1
        if count==1:
            break

# Path to your JSON file containing the questions
questions_json = 'data/STUDENTID.json'
generate_dual_answers(questions_json)