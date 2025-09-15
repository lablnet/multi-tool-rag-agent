# Multi-Tool Agent with RAG

> **⚠️ Educational Project**: This project is created for learning and educational purposes. It demonstrates the implementation of RAG (Retrieval-Augmented Generation) systems, multi-tool AI agents.

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

### Educational Purpose
This project is designed for educational and learning purposes to demonstrate:
- Implementation of RAG (Retrieval-Augmented Generation) systems
- Multi-tool AI agent development using LangGraph
- Integration of various APIs and AI services
- Document processing and vector search techniques
- Best practices in AI application development

### Technical Implementation
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

## TODO - Future Enhancements

### ✅ Completed Features
- [x] **RAG System**: Document processing and semantic search implementation
- [x] **Multi-Tool Agent**: Weather, cryptocurrency, and knowledge base tools
- [x] **Agent Framework**: LangGraph-based agent with ReAct pattern

### 🚀 Advanced Features to Implement
#### Agent using function calling
- [ ] **Agent using function calling**: Agent that can use function calling to perform tasks without using any agentic framework like LangGraph or LangChain.

#### Multi-Agent Systems
- [ ] **Agent Collaboration**: Multiple specialized agents working together
- [ ] **Agent Communication**: Inter-agent messaging and coordination protocols
- [ ] **Agent Orchestration**: Central coordinator managing multiple agents

#### Advanced RAG Enhancements
- [ ] **Hybrid Search**: Combine semantic and keyword-based search
- [ ] **Multi-Modal RAG**: Support for images, audio, and video documents
- [ ] **Dynamic Retrieval**: Adaptive retrieval based on query complexity

#### Agent Intelligence
- [ ] **Memory Systems**: Long-term and short-term memory for agents
- [ ] **Learning Agents**: Agents that improve from interactions
- [ ] **Planning Agents**: Advanced planning and goal decomposition
- [ ] **Reasoning Chains**: Multi-step logical reasoning capabilities
- [ ] **Self-Reflection**: Agents that can evaluate and improve their own performance

### 🎯 Learning Objectives
These enhancements will help explore:
- Advanced AI agent architectures
- Distributed systems and microservices
- Multi-agent coordination algorithms
- Advanced RAG techniques and evaluation
- Human-AI collaboration patterns
- Scalable AI system design
