import streamlit as st
import tempfile
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA

# ------------------ PAGE CONFIG ------------------
st.set_page_config(page_title="Multi-PDF RAG", layout="wide")
st.title("📚 Multi-PDF RAG (OpenAI + LangChain + FAISS)")

# ------------------ OPENAI KEY ------------------
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# ------------------ SESSION STATE ------------------
if "documents" not in st.session_state:
    st.session_state.documents = []

if "vectorstore" not in st.session_state:
    st.session_state.vectorstore = None

# ------------------ PDF UPLOAD ------------------
uploaded_files = st.file_uploader(
    "Upload one or more PDFs",
    type="pdf",
    accept_multiple_files=True
)

if uploaded_files:
    st.session_state.documents = []  # reset on re-upload

    for pdf in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(pdf.read())
            loader = PyPDFLoader(tmp.name)
            st.session_state.documents.extend(loader.load())

    st.success(f"{len(uploaded_files)} PDFs loaded successfully.")

# ------------------ CREATE EMBEDDINGS BUTTON ------------------
if st.button("🚀 Create Embeddings"):
    if not st.session_state.documents:
        st.warning("Please upload PDFs first.")
    else:
        with st.spinner("Creating embeddings..."):

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150
            )

            chunks = splitter.split_documents(st.session_state.documents)

            embeddings = OpenAIEmbeddings(
                model="text-embedding-3-small"
            )

            st.session_state.vectorstore = FAISS.from_documents(
                chunks,
                embeddings
            )

            st.success(f"Embeddings created for {len(chunks)} chunks.")

# ------------------ QUERY SECTION ------------------
if st.session_state.vectorstore:
    st.markdown("---")
    st.subheader("🔎 Ask Questions")

    query = st.text_input("Ask something across all uploaded PDFs")

    if query:
        with st.spinner("Thinking..."):
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0
            )

            qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=st.session_state.vectorstore.as_retriever(
                    search_kwargs={"k": 4}
                ),
                return_source_documents=True
            )

            result = qa_chain(query)

            st.subheader("✅ Answer")
            st.write(result["result"])

            with st.expander("📄 Source Chunks"):
                for doc in result["source_documents"]:
                    st.write(doc.page_content)
                    st.markdown("---")
