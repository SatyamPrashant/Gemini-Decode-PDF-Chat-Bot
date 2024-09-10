import os
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
from io import BytesIO
import base64
import time

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Extract text from PDFs
def get_pdf_text(pdf_docs):
    if not pdf_docs:
        st.error("No PDFs were uploaded. Please upload a PDF and try again.")
        return None
    
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(BytesIO(pdf.read()))
        for page in pdf_reader.pages:
            text += page.extract_text()
    
    return text if text else None

# Split text into chunks
def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return text_splitter.split_text(text)

# Convert chunks to vectors and store them
def get_vector_store(text_chunks):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")

# Create conversational chain for QA
def get_conversational_chain():
    prompt_template = """Analyze the PDF context and answer the question as detailed as possible from the provided context. 
    If the answer is not in the context, say 'The answer is not available in the context.' 
    Do not provide a wrong answer.
    
    Context: \n{context}?\n
    Question: \n{question}\n

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-1.5-pro", temperature=0.9)
    prompt = PromptTemplate(template=prompt_template, input_variables=['context', 'question'])
    return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# Process user input and fetch the response
def user_input(user_question):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")
        db = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
        docs = db.similarity_search(user_question)
        
        if not docs:
            st.warning("The question does not match the context of the uploaded PDFs.")
            return None
        
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response.get("output_text", "No answer available.")
    except Exception as e:
        st.error(f"Error: {str(e)}")
        return None

# Streamlit app
def main():
    st.set_page_config(page_title="PDF Chat Bot")

    # Set background image
    def set_background(image_file):
        with open(image_file, "rb") as image:
            b64_image = base64.b64encode(image.read()).decode("utf-8")
        css = f"""
        <style>
        .stApp {{
            background: url(data:image/png;base64,{b64_image});
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """
        st.markdown(css, unsafe_allow_html=True)

    set_background("background_image.png")

    st.title("Gemini-Decode: PDF Chat Bot App 💬📄:memo:")
    st.markdown("### Generate Summaries and Documentation with Gemini AI")

    # Sidebar: Upload PDF files
    with st.sidebar:
        st.image("logo.png", use_column_width=True)
        st.image("sidebar_image.png", use_column_width=True)
        pdf_docs = st.file_uploader("Upload Your PDF Files", accept_multiple_files=True, type="pdf")
        
        if st.button("Submit & Process"):
            if not pdf_docs:
                st.error("Please upload at least one PDF.")
            else:
                with st.spinner("Processing PDFs..."):
                    raw_text = get_pdf_text(pdf_docs)
                    if raw_text:
                        text_chunks = get_text_chunks(raw_text)
                        get_vector_store(text_chunks)
                        st.success("PDFs processed successfully.")
        
        st.image("sidebar_image1.png", use_column_width=True)

    # Main user interaction area
    user_question = st.text_input("Ask any Question from the PDF Files")

    # Generate result button
    if user_question and st.button("Generate Result :rocket:"):
        with st.spinner("Fetching results... :hourglass:"):
            response = user_input(user_question)
            if response:
                st.markdown("## Progress :hourglass:")
                progress_bar = st.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.01)
                    progress_bar.progress(percent_complete+1)
                st.success("Results Generated! :tada:")
                
                # Formatting the response with sections and key points
                st.markdown("### Here is the Response:")
                st.markdown(f"**Question Asked:** {user_question}")
                st.markdown("### Answer:")
                st.markdown(response)
                st.markdown("---")

if __name__ == "__main__":
    main()

# Footer or additional instructions
st.markdown("---")
st.markdown("#### Developed by Satyam Prashant | Powered by Google Gemini AI & Streamlit")