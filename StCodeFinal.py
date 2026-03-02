import os
import io
import sys
import re
import traceback
import urllib.request
from typing import List, Tuple, Optional

from PIL import Image, ImageOps
import streamlit as st

# Core SDK + LangChain 1.x split packages
import openai
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# --- Document schema (prefer 1.x) ---
try:
    from langchain_core.documents import Document  # 1.x location
except Exception:
    try:
        from langchain.schema import Document  # legacy location
    except Exception:

        class Document:  # type: ignore
            def __init__(self, page_content: str, metadata: Optional[dict] = None):
                self.page_content = page_content
                self.metadata = metadata or {}


# --- Vectorstore backends (FAISS preferred, DocArray fallback) ---
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


# --- App Config --------------------------------------------------------------------
st.set_page_config(
    page_title="Q&A Chatbot on Physics-Informed Machine Learning (Sinha and Dindoruk, 2025)",
    page_icon="📄",
    layout="wide",
)

# --- Workaround: Clear proxy environment variables ---
for _k in (
    "HTTP_PROXY",
    "HTTPS_PROXY",
    "ALL_PROXY",
    "http_proxy",
    "https_proxy",
    "all_proxy",
):
    os.environ.pop(_k, None)

# Constants
PDF_PATH_DEFAULT = "InvitedReviewPaper.pdf"
PAPER_TITLE = "Review of physics-informed machine learning (PIML) methods in subsurface engineering"
PAPER_CITATION = (
    "Sinha and Dindoruk (2025), Geoenergy Science and Engineering 250, 213713"
)
PAPER_URL = "https://www.sciencedirect.com/science/article/abs/pii/S2949891025000715"


# --- Utilities ---------------------------------------------------------------------
def load_square_image(path_or_url: str, size: int = 200) -> Optional[Image.Image]:
    """Open local path or URL, crop more from the bottom (keep top of head), then resize."""
    try:
        # Load image from URL or local path
        if path_or_url.startswith(("http://", "https://")):
            with urllib.request.urlopen(path_or_url, timeout=8) as resp:
                data = resp.read()
            img = Image.open(io.BytesIO(data)).convert("RGB")
        else:
            img = Image.open(path_or_url).convert("RGB")

        w, h = img.size
        min_side = min(w, h)

        # Adjust cropping: keep ~70% of top region, only crop bottom part
        left = (w - min_side) // 2
        top = int(
            (h - min_side) * 0.1
        )  # only crop 10% from top (keep almost entire head)
        top = max(0, top)
        bottom = top + min_side
        if bottom > h:
            bottom = h
            top = h - min_side

        img = img.crop((left, top, left + min_side, bottom))
        img = img.resize((size, size), Image.LANCZOS)
        return img

    except Exception:
        return None


def safe_warning(msg: str):
    st.warning(msg, icon="⚠️")


def safe_info(msg: str):
    st.info(msg, icon="ℹ️")


def show_exception(e: Exception):
    with st.expander("Show error details"):
        st.code(
            "".join(traceback.format_exception(type(e), e, e.__traceback__)),
            language="python",
        )


def call_llm(llm, prompt: str) -> str:
    """Return just the text content, regardless of LC version/object shape."""
    try:
        out = llm.invoke(prompt) if hasattr(llm, "invoke") else llm.predict(prompt)  # type: ignore
    except TypeError:
        out = llm.invoke({"input": prompt})  # type: ignore

    # If it's already a string
    if isinstance(out, str):
        return out

    # LangChain AIMessage / BaseMessage
    content = getattr(out, "content", None)
    if isinstance(content, str):
        return content
    if isinstance(content, list):  # sometimes structured content
        parts = []
        for part in content:
            if isinstance(part, dict) and part.get("type") == "text":
                parts.append(part.get("text", ""))
        if parts:
            return "\n".join(parts)

    # Dict-shaped fallback
    if isinstance(out, dict) and "content" in out:
        return str(out["content"])

    # Last resort: stringify
    return str(out)


# --- Embeddings: adapter to bridge LC -> OpenAI v2 ---------------------------------
def build_embeddings(api_key: Optional[str]):
    """
    Adapter so langchain_openai (expects client.create(...)) works with
    OpenAI SDK v2 (client.embeddings.create(...)).
    """
    client = openai.OpenAI(api_key=api_key)

    class _EmbeddingsClientAdapter:
        def __init__(self, inner):
            self._inner = inner

        def create(self, **kwargs):
            # Forward LC's .create(...) to v2 embeddings endpoint
            return self._inner.embeddings.create(**kwargs)

    adapter = _EmbeddingsClientAdapter(client)

    for model in [
        "text-embedding-3-small",
        "text-embedding-3-large",
        "text-embedding-ada-002",
    ]:
        try:
            return OpenAIEmbeddings(model=model, client=adapter)
        except Exception:
            continue

    return OpenAIEmbeddings(client=adapter)


def build_llm(
    api_key: Optional[str], model_choice: Optional[str] = None, temperature: float = 0.0
):
    """Instantiate ChatOpenAI without passing a custom OpenAI client."""
    kwargs = {"temperature": temperature}
    candidates = (
        [model_choice]
        if model_choice
        else [
            "gpt-5-mini",
            "gpt-5",
            "gpt-4o-mini",
            "gpt-4o",
            "gpt-4.1-mini",
            "gpt-3.5-turbo",
        ]
    )

    for m in candidates:
        try:
            return ChatOpenAI(model=m, api_key=api_key, **kwargs)
        except TypeError:
            try:
                return ChatOpenAI(model=m, openai_api_key=api_key, **kwargs)
            except Exception:
                continue
        except Exception:
            continue

    try:
        return ChatOpenAI(api_key=api_key, **kwargs)
    except TypeError:
        return ChatOpenAI(openai_api_key=api_key, **kwargs)


def load_pdf_documents(pdf_path: str) -> List[Document]:
    """Load PDF into LangChain Documents, attaching page numbers."""
    if not PyPDFLoader:
        raise RuntimeError(
            "PyPDFLoader not available. Install langchain-community and pypdf."
        )
    if not os.path.exists(pdf_path):
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    for i, d in enumerate(docs):
        d.metadata = d.metadata or {}
        d.metadata.setdefault("page", d.metadata.get("page", i + 1))
        d.metadata.setdefault("source", os.path.basename(pdf_path))
    return docs


def chunk_documents(
    docs: List[Document], chunk_size: int = 800, chunk_overlap: int = 200
) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
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
            return (
                FAISS.from_texts(
                    texts=texts, embedding=embeddings, metadatas=metadatas
                ),
                "FAISS",
            )
        except Exception:
            safe_warning(
                "FAISS not available or failed to build. Falling back to DocArrayInMemorySearch."
            )

    if _DA_AVAILABLE:
        try:
            return (
                DocArrayInMemorySearch.from_texts(
                    texts=texts, embedding=embeddings, metadatas=metadatas
                ),
                "DocArrayInMemorySearch",
            )
        except Exception as e:
            raise RuntimeError(
                f"Failed to build DocArrayInMemorySearch vectorstore. Check docarray installation: {e}"
            ) from e

    raise RuntimeError(
        "No usable vectorstore backend found. Install faiss-cpu or ensure DocArrayInMemorySearch is available."
    )


def format_sources(docs_with_scores: List[Tuple[Document, float]]) -> str:
    """Make a compact, grounded context block with scores and page numbers."""
    lines = []
    for i, (doc, score) in enumerate(docs_with_scores, start=1):
        page = doc.metadata.get("page", "?")
        snippet = doc.page_content.strip().replace("\n", " ")
        if len(snippet) > 1200:
            snippet = snippet[:1200] + "…"
        lines.append(f"Source {i} (p.{page}, score={score:.4f}):\n{snippet}")
    return "\n\n".join(lines)


def build_system_prompt() -> str:
    return (
        "You are a technical research assistant.\n"
        "Use ONLY the provided sources to answer the question.\n"
        'If unsure, reply exactly: "Not enough information in the provided sources."\n'
        "\n"
        "Scientific content rules:\n"
        "- Focus strictly on the scientific, mathematical, algorithmic, and engineering substance.\n"
        "- Do NOT describe the structure of the review paper.\n"
        "- Do NOT mention how many papers are reviewed or how the paper is organized.\n"
        "- Do NOT say phrases like 'the review discusses' or 'the paper mentions'.\n"
        "- If asked about a referenced work, explain the actual methodology, equations, assumptions, results, and limitations of that referenced work directly.\n"
        "- Treat referenced works as primary technical contributions and describe them scientifically.\n"
        "\n"
        "Formatting rules:\n"
        "- Do NOT add citations at the end of every sentence or bullet.\n"
        "- Provide exactly ONE citation line at the very end under a heading 'Reference:'.\n"
        f"- That single citation must be exactly: {PAPER_CITATION}\n"
        "- Do NOT write phrases like 'title not provided in the supplied excerpts'.\n"
        "- Do NOT mention missing metadata.\n"
        "\n"
        "Grounding rules:\n"
        "- Only include statements grounded in the provided sources.\n"
        "- Avoid speculation.\n"
    )



    



def build_user_prompt(query: str, grounded_context: str) -> str:
    return f"Question:\n{query}\n\nContext Sources:\n{grounded_context}"


# --- Streamlit UI ------------------------------------------------------------------
st.title("📄 Q&A Chatbot on Physics-Informed Machine Learning")

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
    img2 = load_square_image("utkarsh.jpg") or load_square_image(
        "https://upload.wikimedia.org/wikipedia/commons/8/88/Placeholder_avatar.png"
    )
    if img2:
        st.image(img2, caption="Utkarsh Sinha", width=150)
    st.markdown(
        """
        **Utkarsh Sinha**<br>
        Volunteer Research Associate <br>
        Interaction of Phase-Behavior and Flow (IPB&F) Consortium<br>
        """,
        unsafe_allow_html=True,
    )
with c2:
    img1 = load_square_image("birol.jpg") or load_square_image(
        "https://upload.wikimedia.org/wikipedia/commons/8/88/Placeholder_avatar.png"
    )
    if img1:
        st.image(img1, caption="Dr. Birol Dindoruk", width=150)
    st.markdown(
        """
        **Dr. Birol Dindoruk**<br>
        Professor<br>
        Harold Vance Department of Petroleum Engineering,<br>
        Texas A&M University
        """,
        unsafe_allow_html=True,
    )


st.divider()

# Fixed PDF path (hidden from UI)
pdf_path = PDF_PATH_DEFAULT

openai_api_key = st.text_input(
    "🔑 OpenAI API Key", type="password", help="Key is used only in your session."
)
model_choice = model_choice = st.selectbox(
    "🤖 Model",
    [
        "gpt-5-mini",  # ⭐ recommended default
        "gpt-5",
        "gpt-4o-mini",
        "gpt-4o",
        "gpt-4.1-mini",
        "gpt-3.5-turbo",
    ],
    index=0,
)
query = st.text_input("❓ Ask a question related to the paper:")

# Set env for libs that read from it; many versions still expect env var
if openai_api_key:
    os.environ["OPENAI_API_KEY"] = openai_api_key

# Handy cache reset (prevents stale objects after code changes)
if st.button("🔄 Reset engine cache"):
    st.cache_resource.clear()
    st.success("Caches cleared.")

# Cache heavy stuff (include a compat tag so we can invalidate cleanly)
_COMPAT_TAG = "v2_emb_adapter_v1"


@st.cache_resource(show_spinner=True)
def _cached_embeddings(api_key: Optional[str], compat_tag: str):
    return build_embeddings(api_key)


@st.cache_resource(show_spinner=True)
def _cached_docs(pdf_path_: str):
    return load_pdf_documents(pdf_path_)


@st.cache_resource(show_spinner=True)
def _cached_chunks(pdf_path_: str, chunk_size: int, chunk_overlap: int):
    docs_ = _cached_docs(pdf_path_)
    return chunk_documents(docs_, chunk_size=chunk_size, chunk_overlap=chunk_overlap)


@st.cache_resource(show_spinner=True)
def _cached_vectorstore(
    pdf_path_: str,
    api_key: Optional[str],
    chunk_size: int,
    chunk_overlap: int,
    compat_tag: str,
):
    embs = _cached_embeddings(api_key, compat_tag)
    chunks_ = _cached_chunks(pdf_path_, chunk_size, chunk_overlap)
    return build_vectorstore(chunks_, embs, prefer_faiss=True)


# Advanced settings
with st.expander("⚙️ Advanced settings"):
    st.caption("Tune how the app splits text and retrieves context for your question.")

    chunk_size = st.slider(
        "Chunk size (Text split length in words– larger means fewer, longer sections)",
        300,
        1500,
        800,
        step=50,
    )

    chunk_overlap = st.slider(
        "Chunk overlap (Overlap between chunks – helps context flow)",
        0,
        600,
        200,
        step=25,
    )

    k_results = st.slider(
        "Top-k retrieved chunks (How many relevant pieces to send to the LLM)",
        3,
        20,
        10,
        step=1,
    )

    temperature = st.slider(
        "LLM temperature (Increase for creativity / randomness?)",
        0.0,
        1.0,
        0.0,
        step=0.1,
    )

go = st.button("▶️ Run Q&A", type="primary")


def cleanup_answer(answer: str) -> str:
    # Remove common parenthetical end-citations repeated throughout
    answer = re.sub(r"\s*\(" + re.escape(PAPER_CITATION) + r"\)\s*", " ", answer)
    # Remove the specific filler sentence pattern you mentioned (in case it appears)
    answer = re.sub(r"\(title not provided in the supplied excerpts\)\.?\s*", "", answer, flags=re.I)
    # Normalize whitespace
    answer = re.sub(r"[ \t]+", " ", answer)
    answer = re.sub(r"\n{3,}", "\n\n", answer).strip()
    return answer

# --- Main Flow --------------------------------------------------------------------
if go:
    if not openai_api_key:
        safe_warning("Please enter your OpenAI API key.")
        st.stop()

    if not query:
        safe_warning("Please enter a question to run the Q&A.")
        st.stop()

    try:
        with st.spinner("Building / loading vector store…"):
            vectorstore, backend = _cached_vectorstore(
                pdf_path, openai_api_key, chunk_size, chunk_overlap, _COMPAT_TAG
            )

        st.caption(
            f"Vector backend: **{backend}** | chunks: {_cached_chunks(pdf_path, chunk_size, chunk_overlap).__len__()}"
        )

        with st.spinner("Retrieving relevant chunks…"):
            try:
                docs_scores: List[Tuple[Document, float]] = (
                    vectorstore.similarity_search_with_score(query, k=k_results)
                )
            except AttributeError:
                docs = vectorstore.similarity_search(query, k=k_results)
                docs_scores = [(d, 0.0) for d in docs]

            try:
                docs_scores = sorted(
                    docs_scores, key=lambda x: float(x[1]), reverse=True
                )
            except Exception:
                pass

            if not docs_scores:
                safe_warning(
                    "No chunks retrieved. Try broadening your question or adjusting chunk size/overlap."
                )
                st.stop()

            grounded_context = format_sources(docs_scores)

            with st.spinner("Generating answer…"):
                llm = build_llm(
                    openai_api_key, model_choice=model_choice, temperature=temperature
                )
                system_prompt = build_system_prompt()
                user_prompt = build_user_prompt(query, grounded_context)
                prompt = f"{system_prompt}\n\n{user_prompt}"
                answer = call_llm(llm, prompt)
                answer = cleanup_answer(answer)

            st.subheader("💡 Answer")
            st.markdown(answer)

            st.subheader("🔎 Retrieved Chunks")
            for i, (doc, score) in enumerate(docs_scores, start=1):
                page = doc.metadata.get("page", "?")
                with st.expander(f"Chunk {i} — p.{page} (score={score:.4f})"):
                    st.write(doc.page_content)
                    # if doc.metadata:
                    #     st.caption(f"Metadata: {doc.metadata}")

            st.divider()
            st.caption(f"Citation format reminder: {PAPER_CITATION}")

    except FileNotFoundError as e:
        safe_warning(str(e))
        show_exception(e)
    except Exception as e:
        st.error("Something went wrong while running the Q&A. See details below.")
        show_exception(e)








