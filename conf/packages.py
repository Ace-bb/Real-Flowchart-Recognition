# from langchain.chains import RetrievalQA
# from langchain.llms import OpenAI
# from langchain.vectorstores import FAISS, Chroma, Pinecone
# from langchain.embeddings.openai import OpenAIEmbeddings
# from langchain.embeddings import HuggingFaceEmbeddings, OpenAIEmbeddings
# from langchain.document_loaders import PyMuPDFLoader
# from langchain.retrievers.multi_query import MultiQueryRetriever
# from langchain.text_splitter import RecursiveCharacterTextSplitter, SpacyTextSplitter, TokenTextSplitter, CharacterTextSplitter
from rich.console import Console
from retry import retry
import os
import json
import getpass
# import camelot
import warnings
import copy
import time
import datetime
import numpy as np
import random
from tqdm import tqdm
import shutil
import threading