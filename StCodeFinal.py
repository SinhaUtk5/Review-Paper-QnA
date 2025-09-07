import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings

# ---------------------------
# STATIC CONFIG
# ---------------------------
PDF_PATH = r"InvitedReviewPaper.pdf"  # Keep the PDF in repo root

# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("📄 QnA on Invited Review Paper")

# Show reference info
st.markdown(
    """
**Ref:** Review of physics-informed machine learning (PIML) methods applications in subsurface engineering  
**Authors:** U Sinha, B Dindoruk  
**Journal:** Geoenergy Science and Engineering 250, 213713  
[View Paper](https://www.sciencedirect.com/science/article/abs/pii/S2949891025000715)
"""
)

# Input API key
openai_api_key = st.text_input("🔑 Enter your OpenAI API Key:", type="password")

# Input question
query = st.text_input("❓ Ask a question related to the paper:")

# Only process if key + query are provided
if openai_api_key and query:
    os.environ["OPENAI_API_KEY"] = openai_api_key

    # Step 1: Load PDF
    loader = PyPDFLoader(PDF_PATH)
    documents = loader.load()

    # Step 2: Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=400)
    chunks = text_splitter.split_documents(documents)

    # Step 3: Create embeddings
    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

    # Step 4: Store in FAISS
    texts = [doc.page_content for doc in chunks]
    vectorstore = FAISS.from_texts(texts, embeddings)

    # Step 5: Query with scores
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=7)
    docs_with_scores = sorted(docs_with_scores, key=lambda x: x[1])

    # Weighted context
    context = ""
    for i, (doc, score) in enumerate(docs_with_scores):
        weight = 1 / (score + 1e-6)
        repeat = max(1, int(weight * 3))
        context += (doc.page_content + "\n") * repeat

    # Step 6: Run LLM
    llm = ChatOpenAI(model="gpt-4o-mini", openai_api_key=openai_api_key)
    response = llm.predict(f"Answer based on:\n{context}\n\nQuestion: {query}")

    # Debug chunks
    st.subheader("🔎 Retrieved Chunks")
    for i, (doc, score) in enumerate(docs_with_scores):
        with st.expander(f"Chunk {i+1} (score={score:.4f})"):
            st.write(doc.page_content)

    # Final answer
    st.subheader("💡 Answer")
    st.write(response)

