import os
import json
from typing import List, Dict, Any
import fitz  # PyMuPDF
import numpy as np

# Text Processing and Chunking
import re
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.schema import Document
from sklearn.metrics.pairwise import cosine_similarity

# Ollama for Embeddings
import ollama


class SimplePDFProcessor:
    """
    Simple PDF processor that extracts text directly
    """

    def __init__(self, pdf_path: str):
        self.pdf_path = pdf_path

    def extract_text(self) -> str:
        """
        Extract text from PDF using PyMuPDF (fitz)
        """
        try:
            doc = fitz.open(self.pdf_path)
            text = ""
            for page_num in range(len(doc)):
                page = doc.load_page(page_num)
                text += page.get_text()
            doc.close()
            print(f"✅ Extracted {len(text)} characters from PDF")
            return text
        except Exception as e:
            print(f"❌ Failed to extract text: {e}")
            return ""

    def clean_text(self, text: str) -> str:
        """
        Clean and normalize extracted text
        """
        # Remove excessive whitespace
        text = re.sub(r'\s+', ' ', text)

        # Remove page numbers and headers/footers
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)

        # Remove excessive newlines
        text = re.sub(r'\n\s*\n', '\n\n', text)

        return text.strip()


class IntelligentTextSplitter:
    """
    Advanced text splitting with semantic awareness and context preservation
    """

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # Initialize LangChain's recursive splitter
        self.recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", "! ", "? ", "; ", ", ", " ", ""]
        )

    def split_with_overlap(self, text: str) -> List[str]:
        """
        Split text with intelligent overlap to preserve context
        """
        chunks = self.recursive_splitter.split_text(text)

        # Post-process chunks to ensure quality
        processed_chunks = []
        for chunk in chunks:
            chunk = chunk.strip()
            if len(chunk) > 50:  # Only keep substantial chunks
                processed_chunks.append(chunk)

        return processed_chunks

    def create_documents_with_metadata(self, chunks: List[str], source: str = "hec_outline.pdf") -> List[Document]:
        """
        Create Document objects with metadata for each chunk
        """
        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source": source,
                    "chunk_id": i,
                    "chunk_size": len(chunk),
                    "total_chunks": len(chunks)
                }
            )
            documents.append(doc)

        return documents


class OllamaEmbeddingGenerator:
    """
    Generate embeddings using Ollama with multiple model options
    """

    def __init__(self, model_name: str = "mxbai-embed-large"):
        self.model_name = model_name
        self.embeddings_cache = {}

    def generate_embedding(self, text: str) -> List[float]:
        """
        Generate embedding for a single text using Ollama
        """
        try:
            # Check cache first
            if text in self.embeddings_cache:
                return self.embeddings_cache[text]

            # Generate embedding using Ollama
            response = ollama.embeddings(
                model=self.model_name,
                prompt=text
            )

            embedding = response['embedding']

            # Cache the result
            self.embeddings_cache[text] = embedding

            return embedding
        except Exception as e:
            print(f"❌ Error generating embedding: {e}")
            # Return a zero vector as fallback
            return [0.0] * 768  # Default embedding dimension

    def generate_embeddings_batch(self, texts: List[str], batch_size: int = 10) -> List[List[float]]:
        """
        Generate embeddings for multiple texts in batches
        """
        embeddings = []
        total_texts = len(texts)

        print(f"🔄 Generating embeddings for {total_texts} texts...")

        for i in range(0, total_texts, batch_size):
            batch = texts[i:i + batch_size]
            batch_embeddings = []

            for text in batch:
                embedding = self.generate_embedding(text)
                batch_embeddings.append(embedding)

            embeddings.extend(batch_embeddings)

            # Progress update
            processed = min(i + batch_size, total_texts)
            print(f"📊 Progress: {processed}/{total_texts} embeddings generated")

        print("✅ All embeddings generated successfully!")
        return embeddings

    def save_embeddings(self, embeddings: List[List[float]], texts: List[str], 
                       filename: str = "embeddings.json"):
        """
        Save embeddings and texts to disk
        """
        data = {
            'embeddings': embeddings,
            'texts': texts,
            'model_name': self.model_name
        }

        with open(filename, 'w') as f:
            json.dump(data, f)

        print(f"💾 Embeddings saved to {filename}")

    def load_embeddings(self, filename: str = "embeddings.json") -> Dict[str, Any]:
        """
        Load embeddings from disk
        """
        try:
            with open(filename, 'r') as f:
                data = json.load(f)
            print(f"📂 Embeddings loaded from {filename}")
            return data
        except Exception as e:
            print(f"❌ Error loading embeddings: {e}")
            return {}

class SimpleVectorSearch:
    """
    Simple vector search using cosine similarity
    """

    def __init__(self, embeddings, texts):
        self.embeddings = np.array(embeddings)
        self.texts = texts
        print(f"✅ Vector search initialized with {len(embeddings)} documents")

    def search(self, query_embedding, top_k=5):
        """
        Search for similar documents using cosine similarity
        """
        # Calculate cosine similarities
        similarities = cosine_similarity([query_embedding], self.embeddings)[0]

        # Get top-k most similar documents
        top_indices = np.argsort(similarities)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                'text': self.texts[idx],
                'similarity': similarities[idx],
                'index': idx
            })

        return results

    def search_by_text(self, query_text, embedding_generator, top_k=5):
        """
        Search using text query (generates embedding first)
        """
        print(f"🔍 Searching for: {query_text}")
        # Generate embedding for query
        query_embedding = embedding_generator.generate_embedding(query_text)

        print(f"🔍 Query embedding: {query_embedding}")

        # Search using embedding
        return self.search(query_embedding, top_k)
