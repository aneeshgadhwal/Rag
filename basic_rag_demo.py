''' 
Task 1: Demonstrate the 
build a rag also known as reterival augmented generation for interaction with your data using openai embedding
and then search for similar document based on your

Task 1: Documents + query -> embeddings of both using the opensource embeddings such as  -> similarity search using the cosine similiarity


'''
# list of string
documents = ['my name is ak','i live in paris','i work at the google','my job role includes build a rag']

query = input()

def similarity_search (document_embeddings, query_embeddings) -> int:
    # similarity search based on the cosine similaarity
    from numpy import dot, linalg
    dot_product = dot(document_embeddings, query_embeddings)
    norm1 = linalg.norm(document_embeddings)
    norm2 = linalg.norm(query_embeddings)
    return dot_product/(norm1*norm2)

from langchain_community.embeddings import HuggingFaceEmbeddings
HuggingFaceEmbeddings_ = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

# to embed a list of the text
documents_embed = HuggingFaceEmbeddings_.embed_documents(documents)

query_embed = HuggingFaceEmbeddings_.embed_query(query)

most_related_ranking = []
for i in range(len(documents)):
    # prints the documents with their correspoinding embeddings
    print(documents[i], documents_embed[i])
    value = similarity_search(documents_embed[i], query_embed)
    print(i, " ->>>> ", value)
    most_related_ranking.append(value)

for i in range(len(most_related_ranking)):
    print(i, most_related_ranking[i])




# task 2 use the real documents
# step 1 use the document loader to load the documents
from langchain_community.document_loaders import TextLoader
text_loader = TextLoader('index.md')
text_loader = text_loader.load()

# step 2 use the text splitter to split the text in the documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=40)
splits = text_splitter.split_documents(text_loader)

# step 3 get the embeddings for the documents and the query question

query_embed = query_embed
from langchain_community.embeddings import HuggingFaceEmbeddings
HuggingFaceEmbeddings_ = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')
split_document = []
for i in range(len(splits)):
    split_document.append(splits[i].page_content)

splits_embed = HuggingFaceEmbeddings_.embed_documents(split_document)

# check the cosine similarity betweeen the documents and select the document with the highest cosine similarity
for i in range(len(splits_embed)):
    print(i, similarity_search(query_embed, splits_embed[i]))

# to select top 5 document most similar to the query are
value_index = []
for i in range(len(splits_embed)):
   value_index.append((similarity_search(query_embed, splits_embed[i]),i))

value_index.sort(key=lambda a: a[0], reverse=True)
#print(value_index)


# task 3 indexing with the reterival
# step 1 use the document loader to load the documents
from langchain_community.document_loaders import TextLoader
text_loader = TextLoader('index.md')
text_loader = text_loader.load()

# step 2 use the text splitter to split the text in the documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
splits = text_splitter.split_documents(text_loader)

# step 3 get the embeddings for the documents and the query question

query_embed = query_embed
from langchain_community.embeddings import HuggingFaceEmbeddings
HuggingFaceEmbeddings_ = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

# step 4 vector database to store the data
from langchain_community.vectorstores import Chroma

chroma_vector_database = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings_)
retriever = chroma_vector_database.as_retriever(search_kwargs={"k":5})

#print(retriever.get_relevant_documents(query))


# Task 4 generation using the prompt template
# step 1 use the document loader to load the documents
from langchain_community.document_loaders import TextLoader
text_loader = TextLoader('index.md')
text_loader = text_loader.load()

# step 2 use the text splitter to split the text in the documents
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=20)
splits = text_splitter.split_documents(text_loader)

# step 3 get the embeddings for the documents and the query question

query_embed = query_embed
from langchain_community.embeddings import HuggingFaceEmbeddings
HuggingFaceEmbeddings_ = HuggingFaceEmbeddings(model_name='all-MiniLM-L6-v2')

# step 4 vector database to store the data
from langchain_community.vectorstores import Chroma

chroma_vector_database = Chroma.from_documents(documents=splits, embedding=HuggingFaceEmbeddings_)
retriever = chroma_vector_database.as_retriever(search_kwargs={"k":5})

#print(retriever.get_relevant_documents(query))

# step 5 prompt using the ChatPromptTemplate

from langchain.prompts import ChatPromptTemplate
template_ = """ Answer the question based only on the following context: {context}
Question: {question}
"""
prompt_ = ChatPromptTemplate.from_template(template_, verbose=True)
#print(prompt_)

# step 6 chatmodel llm openai
import os
os.environ['OPENAI_API_KEY'] = <your-openai-api-key>
from langchain_openai import ChatOpenAI
llm = ChatOpenAI(model='gpt-3.5-turbo', temperature=0.0)

# step 7 build a chain and invoke it
chain = prompt_ | llm

# before invoking i want to ckeck the number of token to be passed to the openai model api
# for that we can use tiktoken -> 490 ~~ 500 (which is chunk_size*numberofdocumetsreterived) == 100*5
import tiktoken
#print(len(tiktoken.get_encoding("cl100k_base").encode_batch(retriever.get_relevant_documents(query)[0].page_content)) * 5)

top_doc = retriever.get_relevant_documents(query)
query_answer = chain.invoke({"context":top_doc, "question":query})
print(query_answer.content)

# step 8 build a complete rag chains
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
rag_chain = ({'context': retriever, 'question':RunnablePassthrough()} 
             | prompt_
             | llm
             | StrOutputParser())


query_answer_rag_chain = rag_chain.invoke("give a quick summary of the x company")
print(query_answer_rag_chain)










