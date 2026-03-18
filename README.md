### RAG Demostration


    * This project demostrates how RAG documents pipeline works
    * This also demostrates the complete RAG pipeline which consists of
        * Data Ingestion pipeline
        * Query Retrieval Pipeline
        * Chunking
        * Embedding (Text to Vector)
        * Store to VectorDB(ChromeDB)
        * Query Embedding
        * Context generation
        * LLM finetuning
        * Output

### Prerequisites
    Python3

### Install dependencies

    pip install -r requirement.txt

### RAGBuilder - Each notebook file does the following

###### RAG Pipelines - Data Ingestion to Vector DB Pipeline
###### RAG Pipelines - embedding And vectorStoreDB
###### RAG Pipelines - Retriever Pipeline From VectorStore
###### RAG Pipelines - VectorDB To LLM Output Generation


###### Go to Notebook folder and execute each files seperately

    ## pdf_loader.ipynb
    ## confluence_loader.ipynb
    ## excel_loader.ipynb
    ## document_loader.ipynb

### Refer to flow_images folder to visualize the pipeline flow