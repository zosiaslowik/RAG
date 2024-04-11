from langchain.document_loaders.csv_loader import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
import os
import shutil
from langchain.vectorstores.chroma import Chroma

# Load the CSV file
loader = CSVLoader('medium.csv', encoding='utf-8', source_column="Text")
articles = loader.load()

# Split the text of the articles into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=250)
splitted_articles = text_splitter.split_documents(articles)

# Check if the chroma path exists and delete
if os.path.isdir('chroma'):
    shutil.rmtree('chroma')

# Create database from article chunks
articles_database = Chroma.from_documents(splitted_articles, OpenAIEmbeddings(), persist_directory='chroma')

