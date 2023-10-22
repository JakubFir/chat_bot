# Console ChatBot for asking question about given YouTube video

### Requirements:
 * python
 * API keys for OpenAI and Pinecone
 
### Set up:
    1. git clone repository

    2. install required Python packages:
        * pip install pinecone-client,
        * pip install langchain
        * pip install python-dotenv,
        * pip install logger,
        * pip install youtube-transcript
        * pip install openai,
        * pip install tiktoken
 
    3. set up API keys

    4. Create a file named .env in the project directory with the following content:
        * OPENAI_API_KEY=your-openai-api-key
        * PINECONE_API_KEY=your-pinecone-api-key
    5.Creat Pinecone database
        * name : "chat"
        * Dimensions: 1536
 

## Run the ChatBot application:

   `python main.py`