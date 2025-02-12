import boto3
import re
from datetime import datetime
import hashlib
import json

# ANSI escape codes for coloring text
GREEN = '\033[92m'
RED = '\033[91m'
RESET = '\033[0m'

# Initialize AWS clients
bedrock_agent_runtime_client = boto3.client('bedrock-agent-runtime', region_name='us-east-1')
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')

# DynamoDB table name
QUIZ_HISTORY_TABLE = 'QuizHistory'

# Create the DynamoDB table if it doesn't exist
try:
    table = dynamodb.create_table(
        TableName=QUIZ_HISTORY_TABLE,
        KeySchema=[
            {'AttributeName': 'question_hash', 'KeyType': 'HASH'},
        ],
        AttributeDefinitions=[
            {'AttributeName': 'question_hash', 'AttributeType': 'S'},
        ],
        ProvisionedThroughput={
            'ReadCapacityUnits': 5,
            'WriteCapacityUnits': 5
        }
    )
    table.meta.client.get_waiter('table_exists').wait(TableName=QUIZ_HISTORY_TABLE)
except dynamodb.meta.client.exceptions.ResourceInUseException:
    table = dynamodb.Table(QUIZ_HISTORY_TABLE)

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
def get_question_hash(question):
    """Generate a hash for a question to use as a unique identifier."""
    return hashlib.md5(question.encode()).hexdigest()

def is_question_used(question):
    """Check if a question has been used before."""
    question_hash = get_question_hash(question)
    table = dynamodb.Table(QUIZ_HISTORY_TABLE)
    response = table.get_item(
        Key={'question_hash': question_hash}
    )
    return 'Item' in response

def store_quiz_result(questions, score):
    """Store quiz results in DynamoDB."""
    table = dynamodb.Table(QUIZ_HISTORY_TABLE)
    timestamp = datetime.now().isoformat()
    
    for question in questions:
        question_hash = get_question_hash(question)
        table.put_item(
            Item={
                'question_hash': question_hash,
                'question_text': question,
                'timestamp': timestamp,
                'quiz_score': score
            }
        )

def run_quiz(knowledge_base_id, model_arn):
    print("Generating quiz questions...")
    max_attempts = 3
    attempts = 0
    required_questions = 5
    final_questions = []
    final_answers = []

    while len(final_questions) < required_questions and attempts < max_attempts:
        question_prompt = (
            f"Using the knowledge base, generate {required_questions - len(final_questions)} "
            "multiple-choice questions about the AWS AI Practitioner certification. "
            "Each question should: 1. Clearly state the question. 2. Provide four answer options labeled A, B, C, and D. "
            "3. Indicate the correct answer as 'Correct Answer: A/B/C/D'. Ensure the questions are based on content from the knowledge base "
            "and are different from previous questions."
        )

        quiz_text = query_knowledge_base(knowledge_base_id, model_arn, question_prompt)

        if not quiz_text:
            print("Failed to generate questions. Retrying...")
            attempts += 1
            continue

        questions, answers = parse_questions_and_answers(quiz_text)
        
        for question, answer in zip(questions, answers):
            if len(final_questions) >= required_questions:
                break
            if not is_question_used(question):
                final_questions.append(question)
                final_answers.append(answer)

        attempts += 1

    if not final_questions:
        print("Could not generate enough unique questions. Please try again later.")
        return

    print("\nStarting the quiz. Answer each question one at a time.")
    score = ask_questions(final_questions, final_answers)

    # Store the results
    store_quiz_result(final_questions, score)

    print(f"\nYour final score: {score}/{len(final_questions)}")
    print("Quiz results have been stored.")

if __name__ == "__main__":
    knowledge_base_id = "A1OPRG4XPK"  # Replace with your knowledge base ID
    model_arn = "arn:aws:bedrock:us-east-1::foundation-model/anthropic.claude-v2:1" # Replace with your model ARN

    run_quiz(knowledge_base_id, model_arn)