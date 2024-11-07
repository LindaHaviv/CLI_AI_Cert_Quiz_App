import boto3
import json
import re
from decimal import Decimal
import random
import datetime
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import ConversationalRetrievalChain
from langchain_aws import BedrockLLM, BedrockEmbeddings
from langchain_community.document_loaders import PyPDFLoader

# Initialize DynamoDB and Bedrock clients
dynamodb = boto3.resource('dynamodb', region_name='us-east-1')
table = dynamodb.Table('QuizResults')

# Initialize Bedrock client
bedrock_client = boto3.client('bedrock-runtime', region_name='us-east-1')

# ANSI escape codes for coloring text
GREEN = '\033[92m'  # Bright Green
RED = '\033[91m'    # Bright Red
BLUE = '\033[94m'   # Bright Blue
RESET = '\033[0m'   # Reset to default color

def load_documents(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    if not documents:
        print("No documents loaded. Please check the file path or document format.")
    return documents

def create_vector_store(documents):
    embeddings = BedrockEmbeddings()
    vector_store = FAISS.from_documents(documents, embeddings)
    return vector_store

def generate_quiz_questions_with_rag(vector_store):
    retriever = vector_store.as_retriever()
    bedrock_llm = BedrockLLM(model_id="meta.llama3-70b-instruct-v1:0")

    prompt_template = PromptTemplate(
        input_variables=["context", "question"],
        template="Use the following context to answer the question:\n\n{context}\n\nQuestion: {question}\nAnswer:"
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=bedrock_llm,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": prompt_template}
    )

    today = datetime.datetime.now().strftime("%Y-%m-%d")
    query = f"Generate 5 multiple choice AWS AI Practitioner quiz questions for {today}, excluding other question types."

    input_data = {
        "question": query,
        "chat_history": []
    }

    response = qa_chain.invoke(input_data)

    if response:
        print("Model response:", response)  # Print to check format
    else:
        print("No quiz generated. Please try again.")
        return None

    questions = response.get("answer", "No questions generated.")
    return questions

# Function to generate AI explanations for correct/incorrect answers
def generate_explanation(bedrock_client, question, correct_answer):
    model_id = "meta.llama3-70b-instruct-v1:0"
    prompt = f"Explain why the correct answer to the following question is {correct_answer}:\n\nQuestion: {question}"
    
    try:
        response = bedrock_client.invoke_model(
            modelId=model_id,
            body=json.dumps({"prompt": prompt, "temperature": 0.5, "max_gen_len": 256})
        )
        result = json.loads(response['body'].read())
        return result.get('generation', "No explanation available.")
    
    except Exception as e:
        print(f"Error invoking Bedrock for explanation: {str(e)}")
        return "No explanation available."

def store_quiz_result(score):
    try:
        table.put_item(
            Item={
                'ResultID': 'latest',
                'Score': int(score)
            }
        )
        print("Quiz result stored successfully.")
    except Exception as e:
        print(f"Error storing result in DynamoDB: {str(e)}")

def get_quiz_result():
    try:
        response = table.get_item(Key={'ResultID': 'latest'})
        if 'Item' in response:
            item = response['Item']
            return {'Score': int(item['Score']), 'ResultID': item['ResultID']}
        return None
    except Exception as e:
        print(f"Error retrieving result from DynamoDB: {str(e)}")
        return None

def split_questions_and_answers(quiz_text):
    # Use regular expressions to identify each question and its answer options
    question_answer_pairs = re.findall(r"\*\*Question \d+.*?Answer: .*?\*\*", quiz_text, re.DOTALL)

    if not question_answer_pairs:
        print("No questions found.")
        return [], []

    questions = []
    answers = []

    for pair in question_answer_pairs:
        # Split into question part and answer part
        match = re.search(r"(Question \d+.*?)(Answer: [A-D].*?)\*\*", pair, re.DOTALL)
        if match:
            question_part = match.group(1).strip()
            answer_part = match.group(2).strip()
            
            # Clean up formatting
            question_text = re.sub(r"\*\*", "", question_part).replace("\n", " ").strip()
            answer_text = re.sub(r"\*\*", "", answer_part).replace("Answer:", "").strip()
            
            questions.append(question_text)
            answers.append(answer_text)

    return questions, answers

def ask_questions(questions, answers, bedrock_client):
    score = 0
    # Ensure we reset score and check for valid questions and answers
    if not questions or not answers or len(questions) != len(answers):
        print("Error: Incomplete quiz. Please try again.")
        return 0

    # Limit the quiz to 5 questions
    for i, question in enumerate(questions[:5], 1):  # Only ask the first 5 questions
        print(f"\nQuestion {i}: {question}")
        user_answer = input("Your Answer (type 'exit' to quit): ").strip().lower()

        if user_answer == "exit":
            print("\nExiting the quiz. Goodbye!")
            break

        # Extract only the letter of the correct answer from the answers list
        correct_answer = answers[i - 1].split(')')[0].strip().lower()  # Just the letter, e.g., 'b'

        if user_answer == correct_answer:
            print(f"{GREEN}Correct!{RESET}")
            score += 1
        else:
            print(f"{RED}Incorrect! The correct answer was: {answers[i - 1]}{RESET}")

        # Generate AI explanation if applicable (optional)
        # Updated call to generate_explanation in ask_questions function
        explanation = generate_explanation(bedrock_client, question, answers[i - 1])
        print(f"\n{BLUE}Explanation: {explanation}{RESET}")

    return score

def run_quiz(bedrock_client):
    print("Loading documents and creating vector store...")
    documents = load_documents("/path / to / notes.pdf")
    vector_store = create_vector_store(documents)

    print("Generating quiz questions...")
    quiz_text = generate_quiz_questions_with_rag(vector_store)
    
    if quiz_text is None:
        print("No quiz generated. Please try again.")
        return
    
    print("\nProcessing the quiz questions...")
    questions, answers = split_questions_and_answers(quiz_text)

    if not questions:
        print("No valid questions found. Please try again.")
        return

    print("\nStarting the quiz. Answer each question one at a time.")
    score = ask_questions(questions, answers, bedrock_client)

    store_quiz_result(score)
    result = get_quiz_result()
    if result:
        print(f"\nYour stored result: {result}")
    else:
        print("\nNo previous result found.")

if __name__ == "__main__":
    run_quiz(bedrock_client)
