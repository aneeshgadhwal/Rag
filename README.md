# Retrieval-Augmented Generation (RAG) application


[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)
![OpenAI](https://img.shields.io/badge/OpenAI-412991.svg?style=for-the-badge&logo=OpenAI&logoColor=white)
![Python3](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)
![Markdown](https://img.shields.io/badge/markdown-%23000000.svg?style=for-the-badge&logo=markdown&logoColor=white)
![Wiki](https://img.shields.io/badge/Wikipedia-000000.svg?style=for-the-badge&logo=Wikipedia&logoColor=white)



Construction of a Retrieval-Augmented Generation (RAG) application using various language processing tools and frameworks.

Let's break down the key components and functionalities of the code:

1. **Task 1: Embeddings and Similarity Search**:
   - Defines a list of documents and prompts the user to input a query.
   - Utilizes Hugging Face embeddings to embed both the documents and the query.
   - Implements a cosine similarity function to perform a similarity search between the query and each document.
   - Prints the documents along with their embeddings and the cosine similarity scores relative to the query.

2. **Task 2: Real Documents**:
   - Utilizes a document loader to load text from a Markdown file.
   - Splits the text into smaller chunks using a text splitter.
   - Embeds the chunks using Hugging Face embeddings.
   - Calculates cosine similarity between the query and each document chunk.
   - Selects the top documents based on similarity scores.

3. **Task 3: Indexing with Retrieval**:
   - Repeats the document loading, splitting, embedding, and retrieval steps.
   - Utilizes a vector database to store document embeddings.
   - Performs retrieval using a cosine similarity search and retrieves the top relevant documents.

4. **Task 4: Generation using Prompt Template**:
   - Construct a prompt template for the RAG application.
   - Initializes an OpenAI language model (LLM).
   - Builds a chain for RAG, including context retrieval, prompting, LLM interaction, and output parsing.
