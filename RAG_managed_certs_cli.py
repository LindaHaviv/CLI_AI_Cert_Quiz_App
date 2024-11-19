import boto3
import re

# ANSI escape codes for coloring text
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

# Initialize Bedrock Agent Runtime client
bedrock_agent_runtime_client = boto3.client('bedrock-agent-runtime', region_name='us-east-1')

# Function to query the knowledge base
def query_knowledge_base(knowledge_base_id, model_arn, question):
    try:
        #print(f"DEBUG: Querying knowledge base with ID: {knowledge_base_id}")
        #print(f"DEBUG: Using model ARN: {model_arn}")
        response = bedrock_agent_runtime_client.retrieve_and_generate(
            input={
                'text': question
            },
            retrieveAndGenerateConfiguration={
                'type': 'KNOWLEDGE_BASE',
                'knowledgeBaseConfiguration': {
                    'knowledgeBaseId': knowledge_base_id,
                    'modelArn': model_arn
                }
            }
        )
        return response['output']['text']
    except Exception as e:
        print(f"Error querying knowledge base: {e}")
        return None

# Function to parse questions and answers
def parse_questions_and_answers(response_text):
    questions = []
    answers = []

    # Regular expression to identify question blocks
    question_blocks = re.findall(
        r"(.*?)\n(A\..*?B\..*?C\..*?D\..*?)\nCorrect Answer: ([A-D])",
        response_text,
        re.DOTALL
    )

    for block in question_blocks:
        question = block[0].strip()
        options = block[1].strip()
        correct_answer = block[2].strip().lower()
        questions.append(f"{question}\n{options}")
        answers.append(correct_answer)

    return questions, answers

# Function to ask questions
def ask_questions(questions, answers):
    score = 0
    total_questions = len(questions)

    for i, (question, correct_answer) in enumerate(zip(questions, answers), 1):
        print(f"\nQuestion {i}/{total_questions}: {question}")
        while True:
            user_answer = input("Your Answer (A/B/C/D or type 'exit' to quit): ").strip().lower()
            if user_answer == 'exit':
                print("\nExiting the quiz. Goodbye!")
                return score
            if user_answer not in ['a', 'b', 'c', 'd']:
                print(f"{RED}Invalid input. Please enter A, B, C, or D.{RESET}")
                continue

            if user_answer == correct_answer:
                print(f"{GREEN}Correct!{RESET}")
                score += 1
            else:
                print(f"{RED}Incorrect! The correct answer was: {correct_answer.upper()}{RESET}")
            break

    return score

# Main function to run the quiz
def run_quiz(knowledge_base_id, model_arn):
    print("Generating quiz questions...")
    question_prompt = (
        "Using the knowledge base, generate 5 multiple-choice questions about the AWS AI Practitioner certification. "
        "Each question should: 1. Clearly state the question. 2. Provide four answer options labeled A, B, C, and D. "
        "3. Indicate the correct answer as 'Correct Answer: A/B/C/D'. Ensure the questions are based on content from the knowledge base."
    )

    quiz_text = query_knowledge_base(knowledge_base_id, model_arn, question_prompt)

    if not quiz_text:
        print("No quiz generated. Please try again.")
        return

    #print("DEBUG: Raw Response Text:")
    #print(quiz_text)

    questions, answers = parse_questions_and_answers(quiz_text)
    if not questions:
        print("No valid questions found. Please try again.")
        return

    print("\nStarting the quiz. Answer each question one at a time.")
    score = ask_questions(questions, answers)

    print(f"\nYour final score: {score}/{len(questions)}")

if __name__ == "__main__":
    knowledge_base_id = ""  # Replace with your knowledge base ID
    model_arn = ""  # Replace with your model ARN

    run_quiz(knowledge_base_id, model_arn)