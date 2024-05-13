import os
import tempfile
from dotenv import load_dotenv
import streamlit as st
from langchain.chains.summarize import load_summarize_chain
# from langchain.document_loaders import PyPDFLoader
from langchain_openai import AzureOpenAI
from langchain_community.document_loaders import PyPDFLoader

# Load environment variables from .env file
load_dotenv()

OPENAI_API_VERSION = "2024-02-01"
azure_openai_api_key = os.environ.get('AZURE_OPENAI_API_KEY')

llm = AzureOpenAI(
    deployment_name="gpt-35-turbo-instruct",
    api_version=OPENAI_API_VERSION,
    azure_endpoint="https://d2912.openai.azure.com/",
    temperature=1
)

def summarize_pdfs_from_folder(pdfs_folder):
    summaries = []
    for pdf_file in pdfs_folder:
        try:
            with tempfile.NamedTemporaryFile(delete=False) as temp_file:
                temp_path = temp_file.name
                temp_file.write(pdf_file.read())
            
            loader = PyPDFLoader(temp_path)
            docs = loader.load_and_split()
            chain = load_summarize_chain(llm, chain_type="map_reduce")
            summary = chain.run(docs)
            summaries.append(summary)

        except Exception as e:
            st.error(f"Error summarizing PDF: {e}")
        finally:
            # Delete the temporary file
            os.remove(temp_path)

    return summaries

# Streamlit App
st.title("Multiple PDF Summarizer")

# Allow user to upload PDF files
pdf_files = st.file_uploader("Upload PDF files", type="pdf", accept_multiple_files=True)

if pdf_files:
    # Generate summaries when the "Generate Summary" button is clicked
    if st.button("Generate Summary"):
        st.write("Summaries:")
        summaries = summarize_pdfs_from_folder(pdf_files)
        for i, summary in enumerate(summaries):
            st.write(f"Summary for PDF {i+1}:")
            st.write(summary)

