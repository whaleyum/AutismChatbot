import streamlit as st
import torch
import numpy as np
import pandas as pd
import fitz  # PyMuPDF
import faiss
import os
import tempfile
import nltk
import re
import gc
from datetime import datetime
from sentence_transformers import SentenceTransformer
from transformers import AutoModelForCausalLM, AutoTokenizer
from langchain_text_splitters import RecursiveCharacterTextSplitter
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
import warnings
import streamlit as st

# MUST be the first Streamlit command
st.set_page_config(
    page_title="Autism Research Chatbot",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Add this to prevent automatic processing
import os
import sys

# Check if this is the first run and there are PDFs in the directory
if 'pdf_chunks' not in st.session_state:
    st.session_state.pdf_chunks = []
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'embedder' not in st.session_state:
    st.session_state.embedder = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'pdf_names' not in st.session_state:
    st.session_state.pdf_names = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'models_loaded' not in st.session_state:
    st.session_state.models_loaded = False
warnings.filterwarnings('ignore')

# Set page config FIRST (must be the first Streamlit command)
st.set_page_config(
    page_title="Autism Research Chatbot",
    page_icon="🧩",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Download NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #4ECDC4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #FFD166;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .user-message {
        background-color: #2b2b2b;
        border-left: 4px solid #4ECDC4;
    }
    .bot-message {
        background-color: #1e1e1e;
        border-left: 4px solid #FFD166;
    }
    .refusal-message {
        color: #FF6B6B;
        font-style: italic;
        padding: 1rem;
        background-color: rgba(255, 107, 107, 0.1);
        border-radius: 0.5rem;
    }
    .stProgress > div > div > div > div {
        background-color: #4ECDC4;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'pdf_chunks' not in st.session_state:
    st.session_state.pdf_chunks = []
if 'faiss_index' not in st.session_state:
    st.session_state.faiss_index = None
if 'embedder' not in st.session_state:
    st.session_state.embedder = None
if 'model' not in st.session_state:
    st.session_state.model = None
if 'tokenizer' not in st.session_state:
    st.session_state.tokenizer = None
if 'pdf_names' not in st.session_state:
    st.session_state.pdf_names = []
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False

# Constants
REFUSAL_STRING = "I do not understand this question, and the topic you mentioned is not present in the provided PDF corpus."

def clean_text(text):
    """Clean extracted PDF text"""
    # Remove specific headers/footers
    text = re.sub(r'Lord et al\. Page \d+.*?\n', '', text, flags=re.DOTALL)
    text = re.sub(r'Lancet\. Author manuscript.*?\n', '', text, flags=re.DOTALL)
    text = re.sub(r'Author Manuscript', '', text)
    # Remove URLs
    text = re.sub(r'http\S+', '', text)
    # Fix broken hyphenations
    text = re.sub(r'(\w+)-\s+(\w+)', r'\1\2', text)
    # Remove multiple spaces/newlines
    text = re.sub(r'\s+', ' ', text).strip()
    return text

@st.cache_resource
def load_embedder():
    """Load the sentence transformer model with caching"""
    return SentenceTransformer('BAAI/bge-small-en-v1.5')

@st.cache_resource
def load_model():
    """Load the Qwen model with LoRA"""
    model_name = "Qwen/Qwen2.5-1.5B-Instruct"
    
    # Set environment variable to suppress tokenizer warnings
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True,
        use_fast=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    
    # Load model with appropriate settings
    if torch.cuda.is_available():
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            trust_remote_code=True,
            low_cpu_mem_usage=True
        )
    
    return model, tokenizer

def process_pdfs(uploaded_files):
    """Process uploaded PDF files"""
    pdf_documents = []
    pdf_names = []
    
    progress_bar = st.progress(0)
    for i, uploaded_file in enumerate(uploaded_files):
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        # Extract text
        try:
            doc = fitz.open(tmp_path)
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            
            # Clean text
            cleaned_text = clean_text(text)
            pdf_documents.append(cleaned_text)
            pdf_names.append(uploaded_file.name)
        except Exception as e:
            st.error(f"Error processing {uploaded_file.name}: {str(e)}")
        finally:
            # Clean up temp file
            os.unlink(tmp_path)
        
        # Update progress
        progress_bar.progress((i + 1) / len(uploaded_files))
    
    progress_bar.empty()
    return pdf_documents, pdf_names

def create_chunks(pdf_documents, pdf_names):
    """Split documents into chunks"""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=300,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len
    )
    
    pdf_chunks = []
    for i, doc_text in enumerate(pdf_documents):
        raw_chunks = text_splitter.split_text(doc_text)
        source_name = pdf_names[i]
        for chunk in raw_chunks:
            if len(chunk.strip()) > 50:  # Only keep meaningful chunks
                contextualized_chunk = f"Source: {source_name}\nContent: {chunk}"
                pdf_chunks.append(contextualized_chunk)
    
    return pdf_chunks

def create_faiss_index(chunks, embedder):
    """Create FAISS index from chunks"""
    with st.spinner("Creating embeddings..."):
        chunk_embeddings = embedder.encode(
            chunks, 
            normalize_embeddings=True, 
            convert_to_numpy=True,
            show_progress_bar=True
        )
        dimension = chunk_embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dimension)
        faiss_index.add(chunk_embeddings)
    return faiss_index

def retrieve_top_k_chunks(question, embedder, faiss_index, chunks, k=5):
    """Retrieve top k relevant chunks for a question"""
    instruction = "Represent this sentence for searching relevant passages: "
    query_text = instruction + question
    q_emb = embedder.encode([query_text], normalize_embeddings=True, convert_to_numpy=True)
    distances, indices = faiss_index.search(q_emb, min(k, len(chunks)))
    retrieved_chunks = [chunks[i] for i in indices[0]]
    return " ".join(retrieved_chunks)

def ask_question(question, context, model, tokenizer):
    """Generate answer using the model"""
    # List of forbidden topics for quick rejection
    FORBIDDEN_TOPICS = [
        'capital', 'france', 'paris', 'london', 'berlin', 'tokyo', 'beijing',
        'invented', 'inventor', 'discovered', 'created', 'founded',
        'tire', 'tyre', 'wheel', 'car', 'vehicle', 'automobile',
        'recipe', 'cook', 'bake', 'ingredient', 'kitchen',
        'stock', 'price', 'share', 'market', 'dollar', 'euro',
        'population', 'people live', 'how many',
        'president', 'prime minister', 'leader',
        'sport', 'game', 'match', 'super bowl', 'world cup',
        'weather', 'temperature', 'forecast', 'rain', 'sunny'
    ]
    
    question_lower = question.lower()
    contains_forbidden = any(topic in question_lower for topic in FORBIDDEN_TOPICS)
    
    # Quick rejection for clearly off-topic questions
    if contains_forbidden and (not context or len(context.strip()) < 100):
        return REFUSAL_STRING
    
    # Format prompt
    messages = [
        {"role": "system", "content": f"""You are a STRICT PDF assistant. Your ONLY source of knowledge is the Context below.

ABSOLUTE RULES - YOU MUST FOLLOW THESE:
1. If the Context does NOT contain information to answer the question, you MUST respond with EXACTLY:
   "{REFUSAL_STRING}"
2. This includes ALL questions about topics not covered in the PDFs
3. Do NOT use ANY outside knowledge or common sense
4. Do NOT make up ANY information
5. If you are unsure, ALWAYS choose the refusal message

Context:
{context}"""},
        {"role": "user", "content": question}
    ]
    
    try:
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                temperature=0.1,
                do_sample=True,
                top_p=0.9,
                num_beams=1,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id
            )
        
        response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()
        
        # Safety check
        if contains_forbidden and len(response.split()) < 15 and REFUSAL_STRING not in response:
            if not any(sent in context for sent in response.split('.') if len(sent) > 20):
                return REFUSAL_STRING
        
        return response
    except Exception as e:
        return f"Error generating response: {str(e)}"

def calculate_metrics(question, answer, context, embedder, faiss_index, chunks):
    """Calculate evaluation metrics"""
    metrics = {}
    
    try:
        # Context relevancy
        question_emb = embedder.encode([question])
        context_emb = embedder.encode([context])
        metrics['context_relevancy'] = float(cosine_similarity(question_emb, context_emb)[0][0])
        
        # Answer relevancy
        answer_emb = embedder.encode([answer])
        metrics['answer_relevancy'] = float(cosine_similarity(question_emb, answer_emb)[0][0])
        
        # Groundedness
        metrics['groundedness'] = float(cosine_similarity(answer_emb, context_emb)[0][0])
        
        # Faithfulness (simplified)
        if REFUSAL_STRING in answer:
            metrics['faithfulness'] = 0.0
        else:
            answer_sentences = [s.strip() for s in re.split(r'[.!?]+', answer) if len(s.strip()) > 10]
            context_sentences = [s.strip() for s in re.split(r'[.!?]+', context) if len(s.strip()) > 10]
            
            if answer_sentences and context_sentences:
                ans_embs = embedder.encode(answer_sentences)
                ctx_embs = embedder.encode(context_sentences)
                sim_matrix = cosine_similarity(ans_embs, ctx_embs)
                supported_count = sum(1 for i in range(len(answer_sentences)) if max(sim_matrix[i]) > 0.60)
                metrics['faithfulness'] = supported_count / len(answer_sentences)
            else:
                metrics['faithfulness'] = 0.0
    except Exception as e:
        # Default values if metrics calculation fails
        metrics = {
            'context_relevancy': 0.0,
            'answer_relevancy': 0.0,
            'groundedness': 0.0,
            'faithfulness': 0.0
        }
    
    return metrics

def plot_metrics_radar(metrics):
    """Create radar chart for metrics"""
    categories = list(metrics.keys())
    values = list(metrics.values())
    
    # Number of variables
    N = len(categories)
    
    # Create angles
    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    values += values[:1]
    angles += angles[:1]
    
    # Create plot
    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(projection='polar'))
    ax.plot(angles, values, 'o-', linewidth=2, color='#4ECDC4')
    ax.fill(angles, values, alpha=0.25, color='#4ECDC4')
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, size=10)
    ax.set_ylim(0, 1)
    ax.set_title('Response Quality Metrics', size=14, pad=20)
    ax.grid(True)
    
    return fig

# Main app layout
st.markdown('<h1 class="main-header">🧩 Autism Research Chatbot</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Ask questions about autism based on uploaded research PDFs</p>', unsafe_allow_html=True)

# Sidebar for configuration
with st.sidebar:
    st.header("📁 Document Upload")
    st.markdown("Upload PDF files containing autism research papers to build the knowledge base.")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        if st.button("🔄 Process PDFs", type="primary"):
            with st.spinner("Processing PDFs..."):
                # Process PDFs
                pdf_documents, pdf_names = process_pdfs(uploaded_files)
                st.session_state.pdf_names = pdf_names
                
                if pdf_documents:
                    # Create chunks
                    st.session_state.pdf_chunks = create_chunks(pdf_documents, pdf_names)
                    
                    # Load embedder
                    st.session_state.embedder = load_embedder()
                    
                    # Create FAISS index
                    st.session_state.faiss_index = create_faiss_index(
                        st.session_state.pdf_chunks, 
                        st.session_state.embedder
                    )
                    
                    # Load model
                    with st.spinner("Loading AI model... (this may take a minute)"):
                        st.session_state.model, st.session_state.tokenizer = load_model()
                    
                    st.session_state.processing_complete = True
                    st.success(f"✅ Processed {len(pdf_documents)} PDFs")
                    st.success(f"✅ Created {len(st.session_state.pdf_chunks)} chunks")
                    st.success("✅ Model loaded successfully")
                    st.rerun()
                else:
                    st.error("No valid PDFs could be processed")
    
    # Display stats
    if st.session_state.pdf_chunks:
        st.divider()
        st.header("📊 Statistics")
        st.metric("PDFs Loaded", len(st.session_state.pdf_names))
        st.metric("Knowledge Chunks", len(st.session_state.pdf_chunks))
        
        # Clear button
        if st.button("🗑️ Clear All Data", type="secondary"):
            st.session_state.chat_history = []
            st.session_state.pdf_chunks = []
            st.session_state.faiss_index = None
            st.session_state.embedder = None
            st.session_state.model = None
            st.session_state.tokenizer = None
            st.session_state.pdf_names = []
            st.session_state.processing_complete = False
            st.rerun()

# Main chat interface
if st.session_state.pdf_chunks and st.session_state.model:
    # Chat input
    col1, col2 = st.columns([6, 1])
    with col1:
        user_question = st.text_input("💬 Ask your question:", placeholder="e.g., What is autism?")
    with col2:
        show_metrics = st.checkbox("Show metrics", value=True)
    
    if user_question:
        with st.spinner("🔍 Searching knowledge base..."):
            # Retrieve context
            context = retrieve_top_k_chunks(
                user_question,
                st.session_state.embedder,
                st.session_state.faiss_index,
                st.session_state.pdf_chunks
            )
            
            # Generate answer
            answer = ask_question(
                user_question,
                context,
                st.session_state.model,
                st.session_state.tokenizer
            )
            
            # Calculate metrics if requested
            metrics = {}
            if show_metrics:
                metrics = calculate_metrics(
                    user_question,
                    answer,
                    context,
                    st.session_state.embedder,
                    st.session_state.faiss_index,
                    st.session_state.pdf_chunks
                )
            
            # Add to chat history
            st.session_state.chat_history.append({
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'question': user_question,
                'answer': answer,
                'context': context[:200] + '...' if len(context) > 200 else context,
                'metrics': metrics
            })
    
    # Display chat history
    for chat in reversed(st.session_state.chat_history):
        # User message
        st.markdown(f"""
        <div class="chat-message user-message">
            <strong>🧑 You:</strong><br>
            {chat['question']}
        </div>
        """, unsafe_allow_html=True)
        
        # Bot message
        if REFUSAL_STRING in chat['answer']:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🤖 Assistant:</strong><br>
                <div class="refusal-message">{chat['answer']}</div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="chat-message bot-message">
                <strong>🤖 Assistant:</strong><br>
                {chat['answer']}
            </div>
            """, unsafe_allow_html=True)
        
        # Display metrics if available
        if chat.get('metrics'):
            cols = st.columns(4)
            metrics_display = [
                ("Context Relevancy", chat['metrics'].get('context_relevancy', 0)),
                ("Answer Relevancy", chat['metrics'].get('answer_relevancy', 0)),
                ("Groundedness", chat['metrics'].get('groundedness', 0)),
                ("Faithfulness", chat['metrics'].get('faithfulness', 0))
            ]
            
            for col, (label, value) in zip(cols, metrics_display):
                with col:
                    st.metric(label, f"{value:.2f}")
            
            # Overall score
            avg_score = sum(chat['metrics'].values()) / len(chat['metrics'])
            st.progress(avg_score, text=f"Overall Quality: {avg_score:.2f}")
            
            # Radar chart
            fig = plot_metrics_radar(chat['metrics'])
            st.pyplot(fig)
            plt.close()
        
        st.divider()

elif not st.session_state.pdf_chunks:
    st.info("👈 Please upload and process PDF files in the sidebar to start chatting.")
    
    # Show example questions
    st.markdown("### 📝 Example questions you can ask:")
    examples = [
        "What is autism?",
        "What are the main symptoms of autism?",
        "How is autism diagnosed?",
        "What treatments are available?",
        "What causes autism?"
    ]
    for ex in examples:
        st.markdown(f"- {ex}")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666;">
    <small>⚠️ This chatbot only answers questions based on the uploaded PDFs. It will refuse to answer questions outside its knowledge base.</small>
</div>
""", unsafe_allow_html=True)
