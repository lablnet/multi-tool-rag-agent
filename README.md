# Multi-Tool Agent with RAG

A multi-purpose AI agent built with LangGraph and Google's Gemini AI that combines real-time data access with intelligent document search capabilities. The agent can help with weather information, cryptocurrency prices, and search through educational knowledge bases.

## Features

- 🌤️ **Weather Information**: Get current weather data for any location worldwide
- 💰 **Cryptocurrency Prices**: Check real-time prices for various cryptocurrencies
- 📚 **Knowledge Base Search**: Intelligent search through educational documents using RAG (Retrieval-Augmented Generation)
- 🧠 **Semantic Search**: Advanced vector-based search with semantic understanding
- 📄 **PDF Processing**: Extract and process text from PDF documents
- 🔍 **Intelligent Chunking**: Smart text splitting with context preservation

## Technologies Used

- **LangGraph**: For building the agent framework
- **Google Gemini AI**: As the language model
- **Ollama**: For generating embeddings and local AI capabilities
- **LangChain**: For document processing and text splitting
- **PyMuPDF (fitz)**: For PDF text extraction
- **scikit-learn**: For vector similarity calculations
- **Geopy**: For geocoding and location services
- **Open-Meteo API**: For weather data
- **CoinGecko API**: For cryptocurrency prices

## Setup

1. **Install Dependencies**:
   ```bash
   pip install ollama PyMuPDF langchain langchain-community scikit-learn
   pip install langgraph langchain-google-genai geopy requests
   ```

2. **Install Ollama**:
   - Download and install Ollama from [ollama.ai](https://ollama.ai)
   - Pull the embedding model: `ollama pull mxbai-embed-large`

3. **API Key Configuration**:
   - Get a Google Gemini API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Set your API key in the notebook environment

4. **Initialize RAG System**:
   - Run `rag.ipynb` to process PDF documents and generate embeddings
   - This creates the knowledge base for semantic search

5. **Run the Agent**:
   - Open `agent.ipynb` in Jupyter Notebook or JupyterLab
   - Execute all cells to initialize the multi-tool agent

## Usage

The agent can respond to natural language queries like:

### Weather Queries
- "What's the weather in Islamabad?"
- "How's the weather in New York?"
- "Tell me the current weather in London"

### Cryptocurrency Queries
- "What's the price of Ethereum?"
- "What's the current Bitcoin price?"
- "How much is Litecoin worth?"

### Knowledge Base Queries
- "What is the prerequisite for MSCS?"
- "Tell me about computer science curriculum requirements"
- "What courses are required for software engineering?"
- "Explain the admission criteria for IT programs"

## Project Structure

- `agent.ipynb`: Main notebook containing the multi-tool agent implementation
- `rag.ipynb`: RAG system notebook for processing PDF documents and generating embeddings
- `rag.py`: Core RAG implementation with PDF processing, text splitting, and vector search
- `hec_outline.pdf`: Educational document (Pakistan Universities curriculum outline)
- `hec_outline_embeddings.json`: Pre-generated embeddings for the knowledge base
- `README.md`: This documentation file

## Notes

- The agent uses a ReAct (Reasoning and Acting) pattern for tool selection
- Weather data is provided by Open-Meteo API (free tier)
- Cryptocurrency prices are fetched from CoinGecko API
- The agent includes error handling for API failures and invalid inputs

### RAG Implementation Details

- **PDF Processing**: Uses PyMuPDF for efficient text extraction from PDF documents
- **Text Chunking**: Implements intelligent text splitting with 1000-character chunks and 200-character overlap
- **Embeddings**: Uses Ollama with `mxbai-embed-large` model for generating high-quality embeddings
- **Vector Search**: Employs cosine similarity for semantic document retrieval
- **Knowledge Base**: Currently contains Pakistan Universities curriculum outline (HEC document)
- **Caching**: Embeddings are cached and saved to disk for faster subsequent searches
