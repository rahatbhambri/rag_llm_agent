import openai, langchain, os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter 
from langchain_community.embeddings import SentenceTransformerEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_community.llms import OpenAI 
from dotenv import load_dotenv 
from sentence_transformers import SentenceTransformer  

model = SentenceTransformer('paraphrase-MiniLM-L3-v2')
embeddings = SentenceTransformerEmbeddings(model_name='paraphrase-MiniLM-L3-v2')

load_dotenv()


def read_pdf(path): 
    file_loader = PyPDFDirectoryLoader(path)
    documents = file_loader.load() 
    return documents

def chunk_data(docs, chunk_size=800, chunk_overlap=50):
    text_spliter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    doc = text_spliter.split_documents(docs)
    print(type(doc), type(doc[0]), len(doc))
    
    return doc

data = read_pdf('./static')
doc_list = chunk_data(docs=data)


vector = model.encode("This is a sample sentence")   
# print(vector)


api_key=""
os.environ["PINECONE_API_KEY"] = api_key
pc = Pinecone(api_key=api_key)

index_name = "lcvector"

vectorstore_from_docs = PineconeVectorStore.from_documents(
        doc_list,
        index_name=index_name,
        embedding=embeddings
)


def retrieve_query(query, k=2):
    matching_results = vectorstore_from_docs.similarity_search(query, k=k
                                                               )
    return matching_results


from langchain.chains.question_answering import load_qa_chain
from langchain_community.llms import Ollama

# Initialize the local LLM via Ollama
llm = Ollama(model="mistral", temperature=0.5)

# Load QA chain
chain = load_qa_chain(llm, chain_type="stuff")

def retreive_answers(query):
    doc_search=retrieve_query(query)
    response= chain.run(input_documents=doc_search, question=query) 
    return response 


our_query= "What are key technical skills of Rahat?"
answer=retreive_answers(our_query)
print(answer)