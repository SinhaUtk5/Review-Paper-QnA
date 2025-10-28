import os
import io
import sys
import traceback
import urllib.request
from typing import List, Tuple, Optional

import streamlit as st

# --- Safe, version-agnostic imports -------------------------------------------------
# Text splitter: prefer new package; fall back to older paths.
try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter  # >=0.0.1
except Exception:
    try:
        from langchain.text_splitters import RecursiveCharacterTextSplitter  # legacy
    except Exception as e:
        st.error("Could not import RecursiveCharacterTextSplitter. Please install langchain-text-splitters or a compatible LangChain.")
        st.stop()

# LLM + Embeddings (new, split packages first; fallback to legacy).
ChatOpenAI = None
OpenAIEmbeddings = None
_openai_embeddings_kwargs = {}

try:
    from langchain_openai import ChatOpenAI as _ChatOpenAI, OpenAIEmbeddings as _OpenAIEmbeddings
    ChatOpenAI = _ChatOpenAI
    OpenAIEmbeddings = _OpenAIEmbeddings
    _openai_embeddings_kwargs = {}
except Exception:
    try:
        # Legacy imports
        from langchain.chat_models import ChatOpenAI as _ChatOpenAI
        from langchain.embeddings import OpenAIEmbeddings as _OpenAIEmbeddings
        ChatOpenAI = _ChatOpenAI
        OpenAIEmbeddings = _OpenAIEmbeddings
        _openai_embeddings_kwargs = {}
    except Exception:
        st.error("Could not import OpenAI LLM/Embeddings. Install langchain-openai or a compatible LangChain version.")
        st.stop()

# Document loader (PDF)
try:
    from langchain_community.document_loaders import PyPDFLoader
except Exception:
    PyPDFLoader = None

# Vector stores: prefer FAISS; fallback to DocArrayInMemorySearch to avoid faiss-cpu dep.
_FAISS_AVAILABLE = True
try:
    from langchain_community.vectorstores import FAISS
except Exception:
    _FAISS_AVAILABLE = False
    FAISS = None

_DA_AVAILABLE = True
try:
    from langchain_community.vectorstores import DocArrayInMemorySearch
except Exception:
    _DA_AVAILABLE = False
    DocArrayInMemorySearch = None

# Schemas
try:
    from langchain.schema import Document
except Exception:
    try:
        from langchain_core.documents import Document  # newer core naming
    except Exception:
        # Last resort minimal Document shim
        class Document:  # type: ignore
            def __init__(self, page_content: str, metadata: Optional[dict] = None):
                self.page_content = page_content
                self.metadata = metadata or {}

from PIL import Image, ImageOps


# --- App Config --------------------------------------------------------------------
st.set_page_config(page_title="PIML Invited Review — Q&A", page_icon="📄", layout="wide")

# --- Workaround: some hosts inject proxy envs that break older openai clients ---
for _k in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
    os.environ.pop(_k, None)
    
# Constants (update paths/titles as needed)
PDF_PATH_DEFAULT = "InvitedReviewPaper.pdf"
PAPER_TITLE = "Review of physics-informed machine learning (PIML) methods in subsurface engineering"
PAPER_CITATION = "Sinha and Dindoruk (2025), Geoenergy Science and Engineering 250, 213713"
PAPER_URL = "https://www.sciencedirect.com/science/article/abs/pii/S2949891025000715"

# --- Utilities ---------------------------------------------------------------------
def load_square_image(path_or_url: str, size: int = 200) -> Optional[Image.Image]:
    """Open local path or URL, crop center-square, and resize. Returns None on failure."""
    try:
        if path_or_url.startswith("http://") or path_or_url.startswith("https://"):
            with urllib.request.urlopen(path_or_url, timeout=8) as resp:
                data = resp.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
        else:
            img = Image.open(path_or_url).convert("RGB")
        # Center-crop square
        img = ImageOps.fit(img, (size, size), method=Image.BICUBIC, centering=(0.5, 0.5))
        return img
    except Exception:
        return None


def safe_warning(msg: str):
    st.warning(msg, icon="⚠️")


def safe_info(msg: str):
    st.info(msg, icon="ℹ️")


def show_exception(e: Exception):
    with st.expander("Show error details"):
        st.code("".join(traceback.format_exception(type(e), e, e.__traceback__)), language="python")


def call_llm(llm, prompt: str) -> str:
    """Compatible invocation across LangChain versions."""
    # Newer LC uses .invoke; older exposes .predict
    try:
        # Some ChatOpenAI implementations treat input as dict
        return llm.invoke(prompt) if hasattr(llm, "invoke") else llm.predict(prompt)  # type: ignore
    except TypeError:
        try:
            return llm.invoke({"input": prompt})  # type: ignore
        except Exception as e:
            raise e

# --- START OF FIX ---
def build_embeddings(api_key: Optional[str]):
    """Instantiate embeddings with widest compatibility, explicitly overriding proxies."""
    kwargs = {}
    if api_key:
        kwargs["openai_api_key"] = api_key
    
    # FIX: Define client_kwargs separately to prevent it from being included in the 
    # internal arguments passed by the vector store during embedding generation.
    client_kwargs = {"proxies": None}

    # Prefer modern default model where supported; fall back silently.
    for model in ["text-embedding-3-small", "text-embedding-3-large", "text-embedding-ada-002"]:
        try:
            # Pass client_kwargs explicitly as a keyword argument to the constructor
            return OpenAIEmbeddings(model=model, client_kwargs=client_kwargs, **kwargs)
        except TypeError:
            # Fallback for older versions of LangChain that don't take client_kwargs
            try:
                return OpenAIEmbeddings(model=model, **kwargs)
            except Exception:
                continue
        except Exception:
            continue
            
    # Last resort: try default constructor
    return OpenAIEmbeddings(**kwargs)


def build_llm(api_key: Optional[str], model_choice: Optional[str] = None, temperature: float = 0.0):
    """Instantiate ChatOpenAI across versions, explicitly overriding proxies."""
    kwargs = {"temperature": temperature}
    if api_key:
        kwargs["openai_api_key"] = api_key

    # FIX: Define client_kwargs separately
    client_kwargs = {"proxies": None}

    # Prefer gpt-4o-mini; fall back through some stable models.
    candidates = [model_choice] if model_choice else ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-3.5-turbo"]
    for m in candidates:
        try:
            # Pass client_kwargs explicitly as a keyword argument to the constructor
            return ChatOpenAI(model=m, client_kwargs=client_kwargs, **kwargs)
        except TypeError:
            # Fallback for older versions of LangChain that don't take client_kwargs
            try:
                return ChatOpenAI(model=m, **kwargs)
            except Exception:
                continue
        except Exception:
            continue
            
    # Last resort: try default constructor
    return ChatOpenAI(**kwargs)
# --- END OF FIX ---


def load_pdf_documents(pdf_path: str) -> List[Document]:
    """Load PDF into LangChain Documents, attaching page numbers."""
    if not PyPDFLoader:
        raise RuntimeError("PyPDFLoader not available. Install langchain-community and pypdf.")
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    # Ensure page numbers in metadata for citation
    for i, d in enumerate(docs):
        d.metadata = d.metadata or {}
        d.metadata.setdefault("page", d.metadata.get("page", i + 1))
        d.metadata.setdefault("source", os.path.basename(pdf_path))
    return docs


def chunk_documents(docs: List[Document], chunk_size: int = 800, chunk_overlap: int = 200) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)


def build_vectorstore(chunks: List[Document], embeddings, prefer_faiss: bool = True):
    """
    Build a vector store with graceful fallbacks:
      1) FAISS (requires faiss-cpu)
      2) DocArrayInMemorySearch (pure-Python; no ANN dependency)
    """
    texts = [c.page_content for c in chunks]
    metadatas = [c.metadata for c in chunks]

    if prefer_faiss and _FAISS_AVAILABLE:
        try:
            return FAISS.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas), "FAISS"
        except Exception:
            safe_warning("FAISS not available or failed to build. Falling back to DocArrayInMemorySearch.")

    if _DA_AVAILABLE:
        try:
            # This call should now work cleanly as the embeddings object was initialized properly
            return DocArrayInMemorySearch.from_texts(texts=texts, embedding=embeddings, metadatas=metadatas), "DocArrayInMemorySearch"
        except Exception as e:
            # Added a better error message for clarity if the error persists
            raise RuntimeError(f"Failed to build DocArrayInMemorySearch vectorstore. Check docarray installation: {e}") from e

    raise RuntimeError("No usable vectorstore backend found. Install faiss-cpu or ensure DocArrayInMemorySearch is available.")


def format_sources(docs_with_scores: List[Tuple[Document, float]]) -> str:
    """Make a compact, grounded context block with scores and page numbers."""
    lines = []
    for i, (doc, score) in enumerate(docs_with_scores, start=1):
        page = doc.metadata.get("page", "?")
        # Keep snippets neat
        snippet = doc.page_content.strip().replace("\n", " ")
        if len(snippet) > 1200:
            snippet = snippet[:1200] + "…"
        lines.append(f"Source {i} (p.{page}, score={score:.4f}):\n{snippet}")
    return "\n\n".join(lines)


def build_system_prompt() -> str:
    return (
        "You are a helpful research assistant.\n"
        "Use ONLY the provided sources to answer the question.\n"
        'If unsure, reply exactly: "Not enough information in the provided sources."\n'
        f'Always cite like "{PAPER_CITATION}" and mention the paper title when possible.\n'
        "Only include statements grounded in the sources and avoid speculation.\n"
    )


def build_user_prompt(query: str, grounded_context: str) -> str:
    return f"Question:\n{query}\n\nContext Sources:\n{grounded_context}"


# --- Streamlit UI ------------------------------------------------------------------
st.title("📄 Q&A on Invited Review Paper")

st.markdown(
    f"""
**Ref:** {PAPER_TITLE}  
**Authors:** U. Sinha, B. Dindoruk  
**Journal:** Geoenergy Science and Engineering 250, 213713  
[View Paper]({PAPER_URL})
"""
)

# Authors section
c1, c2 = st.columns(2)
with c1:
    img1 = load_square_image("birol.jpg") or load_square_image(
        "https://upload.wikimedia.org/wikipedia/commons/8/88/Placeholder_avatar.png"
    )
    if img1:
        st.image(img1, caption="Dr. Birol Dindoruk", width=150)
    st.markdown(
        "**Dr. Birol Dindoruk** \n"
        "Professor  \n"
        "Harold Vance Department of Petroleum Engineering,  \n"
        "Texas A&M University"
    )
with c2:
    img2 = load_square_image("utkarsh.jpg") or load_square_image(
        "https://upload.wikimedia.org/wikipedia/commons/8/88/Placeholder_avatar.png"
    )
    if img2:
        st.image(img2, caption="Utkarsh Sinha", width=150)
    st.markdown(
        "**Utkarsh Sinha** \n"
        "Remote Collaborator  \n"
        "Harold Vance Department of Petroleum Engineering,  \n"
        "Texas A&M University"
    )

# Controls
st.divider()

# Fixed PDF path (hidden from UI)
pdf_path = PDF_PATH_DEFAULT

openai_api_key = st.text_input("🔑 OpenAI API Key", type="password", help="Key is used only in your session.")
model_choice = st.selectbox("🤖 Model", ["gpt-4o-mini", "gpt-4o", "gpt-4.1-mini", "gpt-3.5-turbo"], index=0)
query = st.text_input("❓ Ask a question related to the paper:")

# Set env for libs that read from it; many versions still expect env var
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

# Cache heavy stuff
@st.cache_resource(show_spinner=True)
def _cached_embeddings(api_key: Optional[str]):
    return build_embeddings(api_key)

@st.cache_resource(show_spinner=True)
def _cached_docs(pdf_path_: str):
    return load_pdf_documents(pdf_path_)

@st.cache_resource(show_spinner=True)
def _cached_chunks(pdf_path_: str, chunk_size: int, chunk_overlap: int):
    docs_ = _cached_docs(pdf_path_)
    return chunk_documents(docs_, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

@st.cache_resource(show_spinner=True)
def _cached_vectorstore(pdf_path_: str, api_key: Optional[str], chunk_size: int, chunk_overlap: int):
    embs = _cached_embeddings(api_key)
    chunks_ = _cached_chunks(pdf_path_, chunk_size, chunk_overlap)
    return build_vectorstore(chunks_, embs, prefer_faiss=True)

# Sliders for power users (optional)
with st.expander("⚙️ Advanced settings"):
    chunk_size = st.slider("Chunk size", 300, 1500, 800, step=50)
    chunk_overlap = st.slider("Chunk overlap", 0, 600, 200, step=25)
    k_results = st.slider("Top-k retrieved chunks", 3, 20, 10, step=1)
    temperature = st.slider("LLM Temperature", 0.0, 1.0, 0.0, step=0.1)

go = st.button("▶️ Run Q&A", type="primary")

# --- Main Flow --------------------------------------------------------------------
if go:
    if not openai_api_key:
        safe_warning("Please enter your OpenAI API key.")
        st.stop()
    
    # Check if a query was entered
    if not query:
        safe_warning("Please enter a question to run the Q&A.")
        st.stop()

    try:
        # Load vectorstore (cached)
        with st.spinner("Building / loading vector store…"):
            vectorstore, backend = _cached_vectorstore(pdf_path, openai_api_key, chunk_size, chunk_overlap)

        st.caption(f"Vector backend: **{backend}** | chunks: {_cached_chunks(pdf_path, chunk_size, chunk_overlap).__len__()}")

        # Retrieve
        with st.spinner("Retrieving relevant chunks…"):
            try:
                # Works for both FAISS and DocArrayInMemorySearch
                docs_scores: List[Tuple[Document, float]] = vectorstore.similarity_search_with_score(query, k=k_results)
            except AttributeError:
                # Fallback if only similarity_search exists
                docs = vectorstore.similarity_search(query, k=k_results)
                docs_scores = [(d, 0.0) for d in docs]

            # Sort by ascending distance/score if scores are present and numeric
            try:
                docs_scores = sorted(docs_scores, key=lambda x: float(x[1]))
            except Exception:
                pass

            if not docs_scores:
                safe_warning("No chunks retrieved. Try broadening your question or adjusting chunk size/overlap.")
                st.stop()

            grounded_context = format_sources(docs_scores)

            # LLM
            with st.spinner("Generating answer…"):
                llm = build_llm(openai_api_key, model_choice=model_choice, temperature=temperature)
                system_prompt = build_system_prompt()
                user_prompt = build_user_prompt(query, grounded_context)
                prompt = f"{system_prompt}\n\n{user_prompt}"
                answer = call_llm(llm, prompt)

            # Display
            st.subheader("💡 Answer")
            st.markdown(answer)

            st.subheader("🔎 Retrieved Chunks")
            for i, (doc, score) in enumerate(docs_scores, start=1):
                page = doc.metadata.get("page", "?")
                with st.expander(f"Chunk {i} — p.{page} (score={score:.4f})"):
                    st.write(doc.page_content)
                    if doc.metadata:
                        st.caption(f"Metadata: {doc.metadata}")

            st.divider()
            st.caption(f"Citation format reminder: {PAPER_CITATION}")

    except FileNotFoundError as e:
        safe_warning(str(e))
        show_exception(e)
    except Exception as e:
        st.error("Something went wrong while running the Q&A. See details below.")
        show_exception(e)
