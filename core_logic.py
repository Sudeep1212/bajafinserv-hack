import requests
import google.generativeai as genai
import logging
import time
import gc
import fitz  # PyMuPDF
from typing import List
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Core RAG Functions using Semantic Search ---

def download_pdf(url: str, timeout: int = 30) -> bytes:
    """Downloads the PDF content from a URL."""
    logger.info(f"Downloading PDF from {url}")
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download PDF: {e}")
        raise

def extract_text_from_pdf(pdf_bytes: bytes) -> str:
    """Extracts text from PDF bytes using fitz."""
    logger.info("Extracting text from PDF with fitz...")
    text = ""
    with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
        for page in doc:
            text += page.get_text("text") + "\n"
    logger.info(f"Successfully extracted {len(text)} characters.")
    return text

def get_text_chunks(text: str) -> List[str]:
    """Splits text into manageable chunks."""
    logger.info("Splitting text into chunks...")
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text)
    logger.info(f"Created {len(chunks)} chunks.")
    return chunks

# --- Main System Class ---

class HackRxSystem:
    def __init__(self):
        """Initializes the system, API keys, and pre-loads all necessary models."""
        self.api_keys = os.getenv("GEMINI_API_KEYS", "").split(',')
        if not self.api_keys or self.api_keys == ['']:
            raise ValueError("GEMINI_API_KEYS environment variable not set or empty.")
        
        self.current_api_key_index = 0
        self.model = None
        self.embeddings = None
        self._initialize_model()
        self._initialize_embeddings()

    def _initialize_embeddings(self):
        """Loads the sentence-transformer model into memory."""
        try:
            logger.info("Loading sentence-transformer model... (This happens once)")
            self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            logger.info("Sentence-transformer model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load embeddings model: {e}", exc_info=True)
            # This is a fatal error, so we raise it.
            raise e

    def _initialize_model(self):
        """Initializes the Gemini model with the current API key."""
        try:
            api_key = self.api_keys[self.current_api_key_index]
            genai.configure(api_key=api_key)
            
            # Initialize the generative model
            self.model = genai.GenerativeModel('gemini-2.5-flash-lite')
            
            logger.info(f"Initialized Gemini model with API key index: {self.current_api_key_index}")
        except Exception as e:
            logger.error(f"Failed to initialize model with key index {self.current_api_key_index}: {e}")
            self._rotate_api_key()

    def _rotate_api_key(self):
        """Rotates to the next API key."""
        self.current_api_key_index = (self.current_api_key_index + 1) % len(self.api_keys)
        logger.warning(f"Rotating to API key index: {self.current_api_key_index}")
        self._initialize_model()

    def generate_batch_answers(self, questions: list[str], context: str) -> list[str]:
        prompt_template = """
        Based *only* on the context provided below, answer the following {num_questions} questions.
        For each question, provide a concise and direct answer from the text.
        If the answer is not found in the context, state 'Answer not found in context'.

        CONTEXT:
        ---
        {context}
        ---

        QUESTIONS:
        {questions_formatted}

        Provide your response as a numbered list that corresponds exactly to the questions asked.
        For example:
        1. [Answer to question 1]
        2. [Answer to question 2]
        ...
        """

        questions_formatted = "\n".join([f"{i+1}. {q}" for i, q in enumerate(questions)])
        
        prompt = prompt_template.format(
            num_questions=len(questions),
            context=context,
            questions_formatted=questions_formatted
        )

        # Loop for retries with key rotation
        for _ in range(len(self.api_keys)):
            try:
                response = self.model.generate_content(prompt)
                raw_text = response.text
                
                # Parse the numbered list response
                answers = re.split(r'\n\d+\.\s*', raw_text)
                cleaned_answers = [ans.strip() for ans in answers if ans.strip()]
                
                if len(cleaned_answers) >= len(questions):
                    return cleaned_answers[:len(questions)] # Return parsed answers
                else:
                    logger.warning(f"Could not parse batch response correctly. Expected {len(questions)} answers, got {len(cleaned_answers)}. The API call was successful, but the format is wrong. Not retrying.")
                    return ["Could not parse answer from batch response."] * len(questions)

            except Exception as e:
                logger.warning(f"API call failed for batch: {e}. Rotating key and retrying.")
                self._rotate_api_key()
                time.sleep(1) # Wait a second before retrying with the new key

        # If the loop completes, all keys have failed for this batch
        logger.error("All API keys failed for this batch.")
        return ["API call failed for this batch after multiple retries."] * len(questions)

    def process_questions(self, pdf_url: str, questions: List[str]) -> dict:
        """Main processing pipeline for handling a batch of questions."""
        try:
            pdf_bytes = download_pdf(pdf_url)
            document_text = extract_text_from_pdf(pdf_bytes)
            text_chunks = get_text_chunks(document_text)
            
            del document_text
            gc.collect()

            logger.info("Creating vector store...")
            if not self.embeddings:
                raise Exception("Embeddings model not initialized.")

            vector_store = FAISS.from_texts(text_chunks, self.embeddings)
            retriever = vector_store.as_retriever(search_kwargs={"k": 3})
            logger.info("Vector store created successfully.")

            all_answers = []
            batch_size = 10
            for i in range(0, len(questions), batch_size):
                batch_questions = questions[i:i + batch_size]
                logger.info(f"Processing batch {i//batch_size + 1}: {len(batch_questions)} questions.")

                # Gather context for all questions in the batch
                batch_context_docs = set()
                for question in batch_questions:
                    docs = retriever.invoke(question)
                    for doc in docs:
                        batch_context_docs.add(doc.page_content)
                
                combined_context = "\n\n---\n\n".join(batch_context_docs)

                # Generate answers for the batch in a single API call
                batch_answers = self.generate_batch_answers(batch_questions, combined_context)
                all_answers.extend(batch_answers)
                
                # Optional: small delay between batches just in case
                time.sleep(1)

            return {"answers": all_answers}

        except Exception as e:
            logger.error(f"An error occurred during processing: {e}", exc_info=True)
            # Return an error message for all questions if a fatal error occurs
            return {"answers": ["An error occurred while processing the document. Please try again."] * len(questions)}

# Global instance to be used by the API
hackrx_system = HackRxSystem()