from langchain_chroma.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from dotenv import load_dotenv
import os, torch
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langdetect import detect
from sentence_transformers import CrossEncoder

load_dotenv()

GROQ_TRANSLATE_API_KEY = os.getenv("GROQ_TRANSLATE_API_KEY")
llm_translate = ChatGroq(api_key=GROQ_TRANSLATE_API_KEY, model="meta-llama/llama-4-maverick-17b-128e-instruct")

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
llm = ChatGroq(api_key=GROQ_API_KEY, model="meta-llama/llama-4-maverick-17b-128e-instruct")

# Build Retrieval-QA Chain
embeddings = HuggingFaceEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cuda" if torch.cuda.is_available() else "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
vectordb = Chroma(persist_directory="vector_store", embedding_function=embeddings)
retriever = vectordb.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 10,
        "fetch_k": 20,
        "lambda_mult": 0.5
    }
)

combine_prompt = ChatPromptTemplate.from_messages([
    ("system",
     "You must answer strictly using the provided context below. "
     "If the answer is clearly stated in the context, quote it directly. "
     "If it's not present, clearly say: 'My database does not contain this information.' "
     "Do NOT suggest websites or general contacts unless the context is empty. "
     "Important: Do not include any metadata about the source of the information. "
     ),
    ("user", "Context:\n{context}\n\nQuestion:\n{input}")
])

# Chain composition: prompt → LLM → output parser
combine_docs_chain = (
    combine_prompt
    | llm
    | StrOutputParser()
)

# Create retrieval chain
chain = create_retrieval_chain(
    retriever,
    combine_docs_chain
)

def my_combine_docs_run(inputs: dict) -> str:
    user_question = inputs.get("input", "")
    docs = inputs.get("context", [])

    context_text = ""
    for i, doc in enumerate(docs):
        context_text += f"Document {i+1}:\n{doc.page_content}\n\n"

    prompt_text = (
        "answer based on context"
        "Context:\n"
        f"{context_text}\n\n"
        "User question:\n"
        f"{user_question}\n\n"
        "Provide a clear, concise, helpful answer"
        "If you cannot answer from the context, respond: \"My database does not contain this information.\""
    )

    response = llm.invoke(prompt_text)
    return response.content

def detect_and_translate_question(user_input: str):
    prompt = (
        "You are a translator/detector."
        "Detect the language of the following user question and then provide an English translation. "
        "Output exactly in the following two-line format (no extra explanation):\n"
        "LANGUAGE_CODE: <language name in small case> if its hinglish then 'hinglish'\n"
        "TRANSLATION: <English translation of the question>\n\n"
        f"User question: {user_input}"
    )
    resp = llm_translate.invoke(prompt)
    result = resp.content.strip()
    lang_code = "english"
    translation = user_input
    try:
        lines = result.splitlines()
        for line in lines:
            if line.startswith("LANGUAGE_CODE:"):
                lang_code = line.split(":",1)[1].strip()
            elif line.startswith("TRANSLATION:"):
                translation = line.split(":",1)[1].strip()
    except Exception as e:
        lang_code = "english"
        translation = user_input
    return lang_code, translation

def translate_answer_back(answer: str, target_language_code: str):
    if target_language_code == "english":
        return answer
    prompt = (
        "You are a translator. Translate the following answer into the language "
        f"{target_language_code} if and only if it is an Indian Language or Hinglish, else output in English.\n"
        "Do not add explanations, output only the translated answer:\n\n"
        f"{answer}\n"
    )
    resp = llm_translate.invoke(prompt)
    return resp.content.strip()

def ask_user(user_input: str) -> str:
    lang_code, user_input_for_model = detect_and_translate_question(user_input)
    
    response = chain.invoke({"input": user_input_for_model})
    answer = response["answer"] if isinstance(response, dict) and "answer" in response else str(response)

    if lang_code != "english" and detect(answer) == 'en':
        answer = translate_answer_back(answer, lang_code)

    return answer

# Chat loop for testing (Terminal-based)
if __name__ == "__main__":
    print("\n Multilingual Chatbot is ready!")
    print("Type your question (type 'exit' to quit)\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        answer = ask_user(user_input)
        print(f"Bot: {answer}\n")
