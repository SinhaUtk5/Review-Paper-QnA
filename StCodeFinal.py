import os
import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from PIL import Image
from langchain_openai import ChatOpenAI, OpenAIEmbeddings


# Function to load and resize images to square
def load_square_image(path, size=220):
    try:
        img = Image.open(path)
        # Crop to square if needed
        min_side = min(img.size)
        img = img.crop((0, 0, min_side, min_side))
        # Resize
        img = img.resize((size, size))
        return img
    except Exception as e:
        # Fallback placeholder
        return Image.open(
            "https://upload.wikimedia.org/wikipedia/commons/8/88/Placeholder_avatar.png"
        ).resize((size, size))


# ---------------------------
# CONFIG
# ---------------------------
PDF_PATH = r"C:\Users\Utkarsh Sinha\Desktop\Videos and Notes\QnA ChatBot\Invited Review Paper 1-s2.0-S2949891025000715-main (2).pdf"

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


# --- Author section with pictures ---
col1, col2 = st.columns(2)

with col1:
    st.image(
        load_square_image("birol.jpg"), caption="Dr. Birol Dindoruk", width=180
    )  # <-- replace with correct path/URL
    st.markdown(
        """
        **Birol Dindoruk**  
        Professor  
        Harold Vance Department of Petroleum Engineering,  
        Texas A&M University  
        """
    )

with col2:
    st.image(
        load_square_image("utkarsh.jpg"), caption="Utkarsh Sinha", width=180
    )  # <-- replace with correct path/URL
    st.markdown(
        """
        **Utkarsh Sinha**  
        Remote Collaborator  
        Harold Vance Department of Petroleum Engineering,
        Texas A&M University  
        """
    )
# API key input
openai_api_key = st.text_input("🔑 Enter your OpenAI API Key:", type="password")
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key


# ---------------------------
# CACHE PDF + FAISS
# ---------------------------
@st.cache_resource(show_spinner=True)
def load_faiss(pdf_path: str):
    """Load PDF, split into chunks, and build FAISS vector store (cached)."""
    loader = PyPDFLoader(pdf_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=400)
    chunks = text_splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings()
    texts = [doc.page_content for doc in chunks]

    vectorstore = FAISS.from_texts(texts, embeddings)
    return vectorstore


# ---------------------------
# INPUT QUESTION
# ---------------------------
query = st.text_input("❓ Ask a question related to the paper:")

if query and openai_api_key:
    # Load FAISS (cached after first run)
    vectorstore = load_faiss(PDF_PATH)

    # Retrieve relevant docs
    docs_with_scores = vectorstore.similarity_search_with_score(query, k=10)
    docs_with_scores = sorted(docs_with_scores, key=lambda x: x[1])

    # Build grounded context
    context_sections = []
    for i, (doc, score) in enumerate(docs_with_scores, start=1):
        context_sections.append(
            f"Source {i} (score={score:.4f}):\n{doc.page_content.strip()}\n"
        )
    grounded_context = "\n\n".join(context_sections)

    # Optimized prompt
    system_prompt = """You are a helpful research assistant. 
Use ONLY the provided sources to answer the question. 
If unsure, say "Not enough information in the provided sources."
Always cite sources like (Sinha and Dindoruk, 2025) instead of "Source 1".
Also mention the paper title if available in the reference.
"""

    user_prompt = f"""
Question: {query}

Context Sources:
{grounded_context}
"""

    # Run LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    response = llm.predict(f"{system_prompt}\n\n{user_prompt}")

    # Display results
    st.subheader("💡 Answer")
    st.write(response)

    st.subheader("🔎 Retrieved Chunks")
    for i, (doc, score) in enumerate(docs_with_scores):
        with st.expander(f"Chunk {i+1} (score={score:.4f})"):
            st.write(doc.page_content)


