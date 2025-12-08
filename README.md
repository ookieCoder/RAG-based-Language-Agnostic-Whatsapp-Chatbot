# WhatsApp Chatbot using RAG Chain

This project is a multilingual WhatsApp Chatbot that uses Retrieval-Augmented Generation (RAG) to answer user queries based on a knowledge base. It leverages Groq's LLMs for fast inference and translation, ChromaDB for vector storage, and Flask for handling WhatsApp webhooks.

## Features

-   **RAG-based Answers**: Retrieves relevant information from your document knowledge base to answer queries.
-   **Multilingual Support**: Automatically detects the user's language, translates the query to English for processing, and translates the answer back to the user's language.
-   **WhatsApp Integration**: Works directly within WhatsApp using the Meta Cloud API.
-   **Document Ingestion**: Supports various file formats (PDF, TXT, DOCX, XLSX, Images) for building the knowledge base.
-   **Smart Updates**: Only processes new or modified files to update the vector store efficiently.

## Prerequisites

-   Python 3.8+
-   [Groq API Key](https://console.groq.com/)
-   [Meta Developer Account](https://developers.facebook.com/) (for WhatsApp Business API)
-   A configured WhatsApp App in the Meta Developer Dashboard.

## Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/ookieCoder/RAG-based-Language-Agnostic-Whatsapp-Chatbot.git
    cd RAG-based-Language-Agnostic-Whatsapp-Chatbot
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Set up Environment Variables:**
    Create a `.env` file in the root directory
    
    Fill in your API keys and tokens in `.env`:
    ```env
    API_KEY=your_api_key
    ACCESS_TOKEN=your_whatsapp_access_token
    VERIFY_TOKEN=your_custom_verify_token
    ```

## Usage

### 1. Prepare the Knowledge Base
Place your documents (PDFs, text files, etc.) in the `data/` directory.

### 2. Update the Vector Store
Run the update script to ingest documents and create embeddings:
```bash
python update.py
```
This will create a `vector_store` directory containing the ChromaDB database.

### 3. Run the Chatbot
You can run the chatbot in two modes:

**Terminal Mode (Testing):**
Interact with the bot directly in your terminal to test the RAG pipeline.
```bash
python main.py
```

**WhatsApp Webhook Mode (Production):**
Start the Flask server to handle WhatsApp messages.
```bash
python whatsapp.py
```
The server will start on `http://0.0.0.0:5000`.

### 4. Connect to WhatsApp
1.  Expose your local server to the internet using a tool like [ngrok](https://ngrok.com/):
    ```bash
    ngrok http 5000
    ```
2.  Copy the HTTPS URL provided by ngrok (e.g., `https://your-url.ngrok-free.app`).
3.  Go to your Meta Developer Dashboard -> WhatsApp -> Configuration.
4.  In the **Callback URL** field, enter: `https://your-url.ngrok-free.app/webhook`
5.  In the **Verify Token** field, enter the `VERIFY_TOKEN` you defined in your `.env` file.
6.  Verify and Save.

## Project Structure

-   `main.py`: Core RAG logic and terminal-based chat interface.
-   `whatsapp.py`: Flask server for handling WhatsApp webhooks.
-   `update.py`: Script to process documents and update the vector database.
-   `requirements.txt`: Python dependencies.
-   `data/`: Directory to store your knowledge base documents.
-   `vector_store/`: Directory where ChromaDB stores embeddings.

## License

[MIT](LICENSE)
