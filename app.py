"""
Streamlit web application for ArXiv Vector Search System
Fixed Similarity Scores & Enhanced Search Features
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import json
import sys
import os
import zipfile
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_processor import ArXivDataProcessor
from search_engine import ArXivSearchEngine, SearchResult
from utils import (format_authors, format_categories, clean_text_for_display, 
                  format_file_size, extract_arxiv_id, validate_arxiv_id,
                  download_pdf, check_pdf_availability, create_download_summary)
import config

# Page configuration
st.set_page_config(
    page_title="Re-Search AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with modern search page design
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;600;700;900&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    /* Remove empty boxes */
    .element-container:has(> .stMarkdown:empty) {
        display: none !important;
    }
    
    div[data-testid="stVerticalBlock"]:empty {
        display: none !important;
    }
    
    /* Elegant gradient background */
    .main {
        background: linear-gradient(135deg, #FFF8DC 0%, #f5e6ff 50%, #e8d4f5 100%);
        background-attachment: fixed;
    }
    
    /* Remove automatic box styling */
    div[data-testid="stVerticalBlock"] > div {
        background: transparent !important;
        backdrop-filter: none !important;
        border: none !important;
        padding: 0 !important;
        margin: 0 !important;
        box-shadow: none !important;
    }
    
    /* Modern search container */
    .search-container {
        background: rgba(255, 255, 255, 0.9);
        backdrop-filter: blur(20px);
        border-radius: 25px;
        padding: 40px;
        box-shadow: 0 15px 50px rgba(201, 160, 220, 0.3);
        border: 3px solid rgba(201, 160, 220, 0.4);
        margin-bottom: 30px;
        transition: all 0.4s ease;
    }
    
    .search-container:hover {
        box-shadow: 0 20px 60px rgba(201, 160, 220, 0.5);
        transform: translateY(-3px);
    }
    
    /* Headers */
    h1, h2, h3 {
        color: #8b5fa8;
        font-weight: 900;
    }
    
    h1 {
        background: linear-gradient(135deg, #c9a0dc 0%, #8b5fa8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    
    /* Modern buttons with glow */
    .stButton>button {
        background: linear-gradient(135deg, #c9a0dc 0%, #b88dd4 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 15px !important;
        padding: 14px 35px !important;
        font-weight: 700 !important;
        letter-spacing: 1px !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 8px 25px rgba(201, 160, 220, 0.5) !important;
        text-transform: uppercase !important;
        font-size: 14px !important;
    }
    
    .stButton>button:hover {
        transform: translateY(-4px) scale(1.05) !important;
        box-shadow: 0 15px 40px rgba(201, 160, 220, 0.7) !important;
        background: linear-gradient(135deg, #d4b0e8 0%, #c9a0dc 100%) !important;
    }
    
    .stButton>button:active {
        transform: translateY(-2px) scale(1.02) !important;
    }
    
    /* Download button with special styling */
    .stDownloadButton>button {
        background: linear-gradient(135deg, #FFF8DC 0%, #c9a0dc 100%) !important;
        color: #8b5fa8 !important;
        border: 2px solid #c9a0dc !important;
        border-radius: 12px !important;
        font-weight: 700 !important;
        box-shadow: 0 6px 20px rgba(201, 160, 220, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    
    .stDownloadButton>button:hover {
        transform: scale(1.08) !important;
        box-shadow: 0 12px 35px rgba(201, 160, 220, 0.6) !important;
        background: linear-gradient(135deg, #c9a0dc 0%, #FFF8DC 100%) !important;
        color: white !important;
    }
    
    /* Modern input with glow effect */
    .stTextInput>div>div>input {
        background: white !important;
        border: 3px solid rgba(201, 160, 220, 0.4) !important;
        border-radius: 20px !important;
        color: #6b4c7a !important;
        padding: 18px 25px !important;
        font-size: 17px !important;
        transition: all 0.3s ease !important;
        font-weight: 500 !important;
        box-shadow: 0 5px 15px rgba(201, 160, 220, 0.2) !important;
    }
    
    .stTextInput>div>div>input:focus {
        border: 3px solid #c9a0dc !important;
        box-shadow: 0 0 30px rgba(201, 160, 220, 0.5), 0 8px 25px rgba(201, 160, 220, 0.3) !important;
        transform: scale(1.02) !important;
    }
    
    .stTextInput>div>div>input::placeholder {
        color: rgba(139, 95, 168, 0.5) !important;
        font-style: italic !important;
    }
    
    /* Similarity score badge */
    .similarity-high {
        background: linear-gradient(135deg, #90c695 0%, #72b377 100%);
        color: white;
        padding: 8px 18px;
        border-radius: 25px;
        font-weight: 800;
        font-size: 15px;
        box-shadow: 0 4px 15px rgba(144, 198, 149, 0.4);
        display: inline-block;
    }
    
    .similarity-medium {
        background: linear-gradient(135deg, #c9a0dc 0%, #b88dd4 100%);
        color: white;
        padding: 8px 18px;
        border-radius: 25px;
        font-weight: 800;
        font-size: 15px;
        box-shadow: 0 4px 15px rgba(201, 160, 220, 0.4);
        display: inline-block;
    }
    
    .similarity-low {
        background: linear-gradient(135deg, #89b4d4 0%, #6a9dc4 100%);
        color: white;
        padding: 8px 18px;
        border-radius: 25px;
        font-weight: 800;
        font-size: 15px;
        box-shadow: 0 4px 15px rgba(137, 180, 212, 0.4);
        display: inline-block;
    }
    
    /* Metric cards */
    div[data-testid="stMetricValue"] {
        font-size: 36px !important;
        font-weight: 900 !important;
        color: #8b5fa8 !important;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #6b4c7a !important;
        font-weight: 600 !important;
        font-size: 13px !important;
        text-transform: uppercase !important;
        letter-spacing: 1.5px !important;
    }
    
    /* Dividers with gradient */
    hr {
        border: none !important;
        height: 3px !important;
        background: linear-gradient(90deg, transparent, #c9a0dc, transparent) !important;
        margin: 35px 0 !important;
        box-shadow: 0 0 10px rgba(201, 160, 220, 0.5) !important;
    }
    
    /* Modern expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, rgba(255, 255, 255, 0.95) 0%, rgba(201, 160, 220, 0.15) 100%) !important;
        border-radius: 15px !important;
        border: 2px solid rgba(201, 160, 220, 0.5) !important;
        color: #6b4c7a !important;
        font-weight: 700 !important;
        transition: all 0.3s ease !important;
        padding: 18px !important;
        box-shadow: 0 5px 15px rgba(201, 160, 220, 0.2) !important;
    }
    
    .streamlit-expanderHeader:hover {
        background: linear-gradient(135deg, rgba(201, 160, 220, 0.25) 0%, rgba(255, 255, 255, 0.95) 100%) !important;
        box-shadow: 0 8px 25px rgba(201, 160, 220, 0.4) !important;
        transform: scale(1.03) !important;
        border-color: #c9a0dc !important;
    }
    
    /* Alert boxes with modern design */
    .stSuccess, .stError, .stWarning, .stInfo {
        background: rgba(255, 255, 255, 0.95) !important;
        backdrop-filter: blur(10px) !important;
        border-radius: 15px !important;
        border-left: 5px solid !important;
        padding: 20px !important;
        font-weight: 500 !important;
        box-shadow: 0 5px 20px rgba(0, 0, 0, 0.1) !important;
    }
    
    .stSuccess { border-left-color: #90c695 !important; color: #4a7c4e !important; }
    .stError { border-left-color: #e88b8b !important; color: #a64444 !important; }
    .stWarning { border-left-color: #f4c542 !important; color: #997a2a !important; }
    .stInfo { border-left-color: #89b4d4 !important; color: #4a6d8c !important; }
    
    /* Sidebar styling (unchanged) */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, rgba(201, 160, 220, 0.95) 0%, rgba(255, 248, 220, 0.95) 100%) !important;
    }
    
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label {
        color: #6b4c7a !important;
    }
    
    /* Modern tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 15px !important;
        background: rgba(255, 255, 255, 0.7) !important;
        padding: 15px !important;
        border-radius: 20px !important;
        box-shadow: 0 5px 15px rgba(201, 160, 220, 0.2) !important;
    }
    
    .stTabs [data-baseweb="tab"] {
        background: rgba(255, 248, 220, 0.8) !important;
        border-radius: 15px !important;
        color: #6b4c7a !important;
        font-weight: 700 !important;
        padding: 15px 35px !important;
        transition: all 0.3s ease !important;
        border: 2px solid transparent !important;
    }
    
    .stTabs [data-baseweb="tab"]:hover {
        background: rgba(201, 160, 220, 0.3) !important;
        transform: translateY(-2px) !important;
        border: 2px solid rgba(201, 160, 220, 0.5) !important;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #c9a0dc 0%, #b88dd4 100%) !important;
        color: white !important;
        box-shadow: 0 8px 25px rgba(201, 160, 220, 0.5) !important;
        border: 2px solid rgba(255, 255, 255, 0.3) !important;
    }
    
    /* Result number badge with animation */
    .result-badge {
        background: linear-gradient(135deg, #c9a0dc, #8b5fa8);
        color: white;
        border-radius: 50%;
        width: 55px;
        height: 55px;
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        font-weight: 900;
        box-shadow: 0 8px 25px rgba(201, 160, 220, 0.5);
        transition: all 0.4s ease;
    }
    
    .result-badge:hover {
        transform: rotate(360deg) scale(1.15);
        box-shadow: 0 12px 35px rgba(201, 160, 220, 0.7);
    }
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 12px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 248, 220, 0.3);
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #c9a0dc 0%, #d4b0e8 100%);
        border-radius: 10px;
        border: 2px solid rgba(255, 248, 220, 0.5);
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #b88dd4 0%, #c9a0dc 100%);
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load and cache the processed dataset"""
    try:
        processor = ArXivDataProcessor()
        df = processor.process_dataset()
        return df, None
    except Exception as e:
        return None, str(e)

@st.cache_resource
def initialize_search_engine():
    """Initialize and cache the search engine"""
    try:
        df, error = load_data()
        if error:
            return None, error
        
        # Initialize with enhanced features
        search_engine = ArXivSearchEngine(
            df,
            use_reranker=True,  # Enable re-ranking
            use_hybrid=True     # Enable hybrid embeddings
        )
        return search_engine, None
    except Exception as e:
        return None, str(e)

def download_single_pdf_streamlit(arxiv_id: str, title: str = None) -> dict:
    """Download a single PDF"""
    try:
        if not validate_arxiv_id(arxiv_id):
            return {'success': False, 'error': 'Invalid ArXiv ID'}
        
        import requests
        pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
        headers = {'User-Agent': 'Mozilla/5.0'}
        
        response = requests.get(pdf_url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        pdf_data = response.content
        
        if not pdf_data or not pdf_data.startswith(b'%PDF'):
            return {'success': False, 'error': 'Invalid PDF'}
        
        if title:
            clean_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()[:50]
            filename = f"{arxiv_id}_{clean_title}.pdf"
        else:
            filename = f"{arxiv_id}.pdf"
        
        return {'success': True, 'data': pdf_data, 'filename': filename, 'size': len(pdf_data)}
    except:
        return {'success': False, 'error': 'Download failed'}

def create_zip_download_streamlit(selected_results: list) -> dict:
    """Create ZIP file with selected PDFs"""
    if not selected_results:
        return {'success': False, 'error': 'No papers selected'}
    
    import io
    zip_buffer = io.BytesIO()
    successful = []
    failed = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, result in enumerate(selected_results):
            arxiv_id = result.get('arxiv_id') or extract_arxiv_id(result.get('document_id', ''))
            
            if not arxiv_id:
                failed.append({'title': result.get('title', 'Unknown'), 'error': 'No ID'})
                continue
            
            status_text.text(f"📥 {arxiv_id}... ({i+1}/{len(selected_results)})")
            
            download_result = download_single_pdf_streamlit(arxiv_id, result.get('title'))
            
            if download_result and download_result.get('success'):
                zip_file.writestr(download_result['filename'], download_result['data'])
                successful.append({'arxiv_id': arxiv_id, 'title': result.get('title')})
            else:
                failed.append({'title': result.get('title'), 'error': 'Failed'})
            
            progress_bar.progress((i + 1) / len(selected_results))
    
    status_text.empty()
    progress_bar.empty()
    
    if successful:
        zip_buffer.seek(0)
        return {
            'success': True,
            'data': zip_buffer.getvalue(),
            'filename': f'papers_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip',
            'successful': successful,
            'failed': failed
        }
    return {'success': False, 'error': 'No PDFs downloaded'}

def display_search_result(result: dict, index: int):
    """Display a single search result with fixed similarity score (0-1 range)"""
    
    with st.container():
        st.markdown(f"""
        <div style="background: rgba(255, 255, 255, 0.95); backdrop-filter: blur(15px);
                    border-radius: 20px; border: 2px solid rgba(201, 160, 220, 0.3);
                    padding: 30px; margin-bottom: 25px; box-shadow: 0 10px 35px rgba(201, 160, 220, 0.2);
                    transition: all 0.3s ease;">
        """, unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns([0.05, 0.75, 0.2])
        
        with col1:
            st.markdown(f'<div class="result-badge">{index + 1}</div>', unsafe_allow_html=True)
        
        with col2:
            arxiv_id = result.get('arxiv_id') or extract_arxiv_id(result.get('document_id', ''))
            
            if arxiv_id:
                st.markdown(f"### [{result['title']}](https://arxiv.org/abs/{arxiv_id})")
            else:
                st.markdown(f"### {result['title']}")
            
            # Fixed similarity score (ensure it's between 0 and 1)
            score = float(result.get('similarity_score', 0))
            # Normalize if score is outside 0-1 range
            if score > 1.0:
                score = min(score / 10.0, 1.0)  # Normalize scores that are too high
            score = max(0.0, min(1.0, score))  # Clamp between 0 and 1
            
            # Determine badge style based on score
            if score > 0.7:
                badge_class = "similarity-high"
                emoji = "🔥"
                label = "High Match"
            elif score > 0.5:
                badge_class = "similarity-medium"
                emoji = "⭐"
                label = "Good Match"
            else:
                badge_class = "similarity-low"
                emoji = "💫"
                label = "Fair Match"
            
            categories_str = format_categories(result['categories'], 2)
            
            col_meta1, col_meta2 = st.columns(2)
            
            with col_meta1:
                st.markdown(f'<div class="{badge_class}">{emoji} {score:.3f} - {label}</div>', unsafe_allow_html=True)
            
            with col_meta2:
                st.markdown(f"**🏷️ Categories:** {categories_str}")
            
            st.markdown("<br>", unsafe_allow_html=True)
            
            if result.get('authors'):
                st.markdown(f"**👥 Authors:** {format_authors(result['authors'], 3)}")
            
            # AI Summary or Abstract
            if result.get('ai_summary'):
                with st.expander("🤖 AI Summary", expanded=True):
                    st.info(result['ai_summary'])
                with st.expander("📄 Full Abstract"):
                    st.write(result.get('abstract', 'No abstract'))
            else:
                with st.expander("📄 Abstract"):
                    st.write(result.get('abstract', 'No abstract'))
        
        with col3:
            if arxiv_id:
                # Find Similar Papers button
                if st.button("🔗 Similar", key=f"sim_{index}", use_container_width=True, help="Find papers similar to this one"):
                    st.session_state.similar_paper_id = arxiv_id
                    st.session_state.show_similar = True
                    st.rerun()
                
                if st.button("📥 PDF", key=f"dl_{index}", use_container_width=True):
                    with st.spinner("⏳"):
                        dl_result = download_single_pdf_streamlit(arxiv_id, result.get('title'))
                        
                        if dl_result and dl_result.get('success'):
                            st.download_button(
                                "💾 Save",
                                dl_result['data'],
                                dl_result['filename'],
                                "application/pdf",
                                key=f"sv_{index}",
                                use_container_width=True
                            )
                            st.success("✅")
                        else:
                            st.error("❌")
                
                st.link_button("🌐 ArXiv", f"https://arxiv.org/abs/{arxiv_id}", use_container_width=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

def display_sidebar_stats(df: pd.DataFrame):
    """Display sidebar statistics (unchanged)"""
    
    st.sidebar.markdown("""
    <div style="text-align: center; padding: 25px 10px; background: rgba(255, 248, 220, 0.7);
                border-radius: 20px; margin-bottom: 20px; border: 2px solid rgba(201, 160, 220, 0.5);">
        <div style="font-size: 48px;">🔍</div>
        <h2 style="font-size: 24px; margin: 10px 0 5px 0;">PaperPilot</h2>
        <p style="font-size: 12px; margin: 0;">Semantic Search</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.sidebar.markdown("### 📊 Dataset")
    col1, col2 = st.sidebar.columns(2)
    
    with col1:
        st.metric("Papers", f"{len(df):,}")
    with col2:
        if 'word_count' in df.columns:
            st.metric("Avg Words", f"{int(df['word_count'].mean()):,}")
    
    all_cats = []
    for cats in df['categories']:
        if isinstance(cats, list):
            all_cats.extend(cats)
    
    if all_cats:
        from collections import Counter
        top_cats = Counter(all_cats).most_common(5)
        
        st.sidebar.markdown("### 🏷️ Top Categories")
        for cat, count in top_cats:
            pct = (count / len(df)) * 100
            st.sidebar.write(f"**{cat}:** {count} ({pct:.1f}%)")
    
    if config.ENABLE_INDIVIDUAL_SUMMARIES:
        st.sidebar.markdown("### 🤖 AI Engine")
        st.sidebar.success("✅ Active")
        st.sidebar.write(f"**Model:** {config.OPENAI_MODEL}")
        cost = config.estimate_cost_per_search()
        st.sidebar.write(f"**Cost:** ~${cost:.4f}/search")

def search_page():
    """Enhanced search page with modern design"""
    
    # Hero section with modern design
    st.markdown("""
    <div style="text-align: center; padding: 50px 0 40px;">
        <h1 style="font-size: 64px; margin-bottom: 15px; font-weight: 900;">
            PaperPilot
        </h1>
        <p style="font-size: 20px; color: #8b5fa8; font-weight: 500; letter-spacing: 0.5px;">
            Charting Academia's Universe with Zero Turbulence
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    search_engine, error = initialize_search_engine()
    
    if error:
        st.error(f"❌ {error}")
        st.info("💡 Run: `python main.py process && python main.py build-index`")
        return
    
    # Modern search container
    st.markdown('<div class="search-container">', unsafe_allow_html=True)
    
    col1, col2 = st.columns([0.8, 0.2])
    
    with col1:
        query = st.text_input(
            "Search", 
            placeholder="e.g., neural networks, quantum computing, deep learning...", 
            label_visibility="collapsed",
            key="main_search"
        )
    
    with col2:
        search_button = st.button("SEARCH", type="primary", use_container_width=True)
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Advanced options
    with st.expander("⚙️ Advanced Search Options"):
        col_opt1, col_opt2 = st.columns(2)
        
        with col_opt1:
            st.markdown("**🎛️ Search Settings**")
            top_k = st.slider("Number of results", 1, 50, config.TOP_K_RESULTS)
            min_similarity = st.slider("Minimum similarity (0-1)", 0.0, 1.0, max(0.0, min(1.0, config.SIMILARITY_THRESHOLD)), 0.05)
            categories = st.multiselect("Filter by categories", ["cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.RO", "stat.ML", "math.ST", "physics.data-an"])
        
        with col_opt2:
            st.markdown("**🤖 AI Features**")
            generate_summaries = st.checkbox("Generate AI Summaries", value=config.ENABLE_INDIVIDUAL_SUMMARIES)
            if generate_summaries:
                st.info(f"💰 Estimated cost: ~${config.estimate_cost_per_search():.4f} per search")
            export_format = st.selectbox("Export format", ["None", "JSON", "CSV", "BibTeX"])
    
    # Find Similar Papers section
    if st.session_state.get('show_similar') and st.session_state.get('similar_paper_id'):
        st.markdown("---")
        st.markdown("### 🔗 Find Similar Papers")
        
        col_sim1, col_sim2, col_sim3 = st.columns([0.6, 0.2, 0.2])
        
        with col_sim1:
            st.info(f"Finding papers similar to: **{st.session_state.similar_paper_id}**")
        
        with col_sim2:
            num_similar = st.number_input("Number of similar papers", 1, 20, 5, key="num_sim")
        
        with col_sim3:
            if st.button("🔍 Find Similar", use_container_width=True):
                try:
                    df, _ = load_data()
                    similar_papers = search_engine.get_similar_papers(
                        st.session_state.similar_paper_id, 
                        top_k=num_similar
                    )
                    
                    if similar_papers:
                        results = [paper.to_dict() for paper in similar_papers]
                        st.session_state.results = results
                        st.session_state.query = f"Similar to {st.session_state.similar_paper_id}"
                        st.session_state.show_similar = False
                        st.rerun()
                    else:
                        st.warning("No similar papers found")
                except Exception as e:
                    st.error(f"Error: {str(e)}")
            
            if st.button("❌ Cancel", use_container_width=True):
                st.session_state.show_similar = False
                st.rerun()
    
    # Search execution
    if search_button and query.strip():
        with st.spinner("🔍 Searching through research papers..."):
            try:
                search_kwargs = {
                    'top_k': top_k,
                    'min_similarity': min_similarity,
                    'generate_summaries': generate_summaries
                }
                
                if categories:
                    search_kwargs['category_filter'] = categories
                
                search_results = search_engine.search_with_ai_summary(query, **search_kwargs)
                results = search_results['results']
                ai_summary = search_results.get('ai_summary')
                
                st.session_state.results = results
                st.session_state.query = query
                st.session_state.ai_summary = ai_summary
                st.session_state.show_similar = False
            except Exception as e:
                st.error(f"❌ Search failed: {str(e)}")
                return
    
    # Display results
    if hasattr(st.session_state, 'results'):
        results = st.session_state.results
        query_text = st.session_state.query
        
        if not results:
            st.warning("🤷 No results found. Try different keywords or lower the similarity threshold.")
            return
        
        # Results header with modern design
        st.markdown(f"""
        <div style="background: linear-gradient(135deg, #90c695 0%, #c9a0dc 100%);
                    color: white; padding: 30px; border-radius: 25px; margin: 30px 0;
                    text-align: center; box-shadow: 0 15px 45px rgba(201, 160, 220, 0.4);
                    border: 3px solid rgba(255, 255, 255, 0.3);">
            <h2 style="margin: 0; font-weight: 900; color: white; font-size: 32px;">
                ✅ Found {len(results)} Papers
            </h2>
            <p style="margin: 15px 0 0 0; opacity: 0.95; font-size: 18px; font-weight: 500;">
                for query: <em>"{query_text}"</em>
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        # Overall AI Summary
        if st.session_state.get('ai_summary'):
            st.markdown("### 🧠 AI Research Overview")
            st.info(st.session_state.ai_summary)
        
        # Tabs for organization
        tab1, tab2, tab3 = st.tabs(["📚 Research Papers", "📊 Analytics", "📦 Bulk Download"])
        
        with tab1:
            st.markdown(f"<br>", unsafe_allow_html=True)
            for i, result in enumerate(results):
                display_search_result(result, i)
        
        with tab2:
            st.markdown("### 📊 Search Results Analytics")
            
            if len(results) >= 2:
                # Create visualizations
                scores = [float(r.get('similarity_score', 0)) for r in results]
                # Normalize scores if needed
                scores = [min(s / 10.0 if s > 1.0 else s, 1.0) for s in scores]
                
                # Similarity distribution
                col_viz1, col_viz2 = st.columns(2)
                
                with col_viz1:
                    fig_scores = go.Figure()
                    fig_scores.add_trace(go.Histogram(
                        x=scores,
                        nbinsx=15,
                        marker=dict(color='#c9a0dc', line=dict(color='#8b5fa8', width=2)),
                        hovertemplate='Score: %{x:.3f}<br>Count: %{y}<extra></extra>'
                    ))
                    
                    fig_scores.update_layout(
                        title="Similarity Score Distribution",
                        plot_bgcolor='rgba(255, 248, 220, 0.3)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font=dict(color='#6b4c7a', family='Poppins', size=12),
                        xaxis=dict(title="Similarity Score (0-1)", gridcolor='rgba(201, 160, 220, 0.2)', range=[0, 1]),
                        yaxis=dict(title="Count", gridcolor='rgba(201, 160, 220, 0.2)'),
                        height=350
                    )
                    
                    st.plotly_chart(fig_scores, use_container_width=True)
                
                with col_viz2:
                    # Category distribution
                    all_categories = []
                    for result in results:
                        all_categories.extend(result.get('categories', []))
                    
                    if all_categories:
                        from collections import Counter
                        cat_counts = Counter(all_categories)
                        top_cats = cat_counts.most_common(8)
                        
                        fig_cats = go.Figure()
                        fig_cats.add_trace(go.Bar(
                            y=[cat for cat, _ in top_cats],
                            x=[count for _, count in top_cats],
                            orientation='h',
                            marker=dict(
                                color=['#c9a0dc', '#d4b0e8', '#e8d4f5', '#b88dd4', '#a77cc4', '#9b6fb4', '#8b5fa8', '#7b4f98'][:len(top_cats)]
                            ),
                            hovertemplate='%{y}: %{x} papers<extra></extra>'
                        ))
                        
                        fig_cats.update_layout(
                            title="Top Categories",
                            plot_bgcolor='rgba(255, 248, 220, 0.3)',
                            paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(color='#6b4c7a', family='Poppins', size=12),
                            xaxis=dict(title="Count", gridcolor='rgba(201, 160, 220, 0.2)'),
                            yaxis=dict(title="", gridcolor='rgba(201, 160, 220, 0.2)'),
                            height=350
                        )
                        
                        st.plotly_chart(fig_cats, use_container_width=True)
                
                # Summary statistics
                st.markdown("#### 📈 Statistics Summary")
                col_stat1, col_stat2, col_stat3, col_stat4 = st.columns(4)
                
                with col_stat1:
                    avg_score = np.mean(scores)
                    st.metric("Average Similarity", f"{avg_score:.3f}")
                
                with col_stat2:
                    max_score = np.max(scores)
                    st.metric("Highest Score", f"{max_score:.3f}")
                
                with col_stat3:
                    unique_cats = len(set(all_categories))
                    st.metric("Unique Categories", unique_cats)
                
                with col_stat4:
                    unique_authors = len(set([a for r in results for a in r.get('authors', [])]))
                    st.metric("Unique Authors", unique_authors)
            else:
                st.info("📊 Need at least 2 results to show analytics")
        
        with tab3:
            st.markdown("### 📦 Bulk PDF Download")
            st.write("Select multiple papers to download as a single ZIP file.")
            
            selected = st.multiselect(
                "Select papers:",
                range(len(results)),
                format_func=lambda x: f"[{x+1}] {results[x]['title'][:65]}..."
            )
            
            col_bulk1, col_bulk2 = st.columns([1, 1])
            
            with col_bulk1:
                if selected and st.button("📦 Create ZIP File", use_container_width=True):
                    selected_results = [results[i] for i in selected]
                    zip_result = create_zip_download_streamlit(selected_results)
                    
                    if zip_result and zip_result.get('success'):
                        st.session_state.zip_data = zip_result
                        st.success(f"✅ ZIP created with {len(zip_result['successful'])} papers!")
                        
                        if zip_result['failed']:
                            with st.expander(f"⚠️ {len(zip_result['failed'])} papers failed"):
                                for failed in zip_result['failed']:
                                    st.text(f"❌ {failed.get('title', 'Unknown')[:50]}: {failed['error']}")
                    else:
                        st.error(f"❌ {zip_result.get('error', 'Failed to create ZIP')}")
            
            with col_bulk2:
                if hasattr(st.session_state, 'zip_data'):
                    zip_data = st.session_state.zip_data
                    st.download_button(
                        f"⬇️ Download ZIP ({format_file_size(len(zip_data['data']))})",
                        zip_data['data'],
                        zip_data['filename'],
                        "application/zip",
                        use_container_width=True
                    )
        
        # Export functionality
        if export_format != "None":
            try:
                search_results_obj = [SearchResult(**{
                    'document_id': r.get('document_id', ''),
                    'title': r.get('title', ''),
                    'abstract': r.get('abstract', ''),
                    'authors': r.get('authors', []),
                    'categories': r.get('categories', []),
                    'similarity_score': float(r.get('similarity_score', 0)),
                    'update_date': pd.to_datetime(r['update_date']) if r.get('update_date') else None,
                    'ai_summary': r.get('ai_summary', '')
                }) for r in results]
                
                exported_data = search_engine.export_search_results(search_results_obj, export_format.lower())
                filename = f"results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format.lower()}"
                
                st.download_button(
                    f"📁 Export as {export_format}",
                    exported_data,
                    filename,
                    "text/plain",
                    use_container_width=False
                )
            except Exception as e:
                st.error(f"Export failed: {str(e)}")

def analytics_page():
    """Analytics dashboard"""
    st.markdown("""
    <div style="text-align: center; padding: 40px 0;">
        <h1 style="font-size: 52px; font-weight: 900;">📊 Analytics Dashboard</h1>
        <p style="font-size: 18px; color: #8b5fa8;">Insights into your research database</p>
    </div>
    """, unsafe_allow_html=True)
    
    df, error = load_data()
    
    if error:
        st.error(f"❌ {error}")
        return
    
    # Elegant metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(201, 160, 220, 0.2), rgba(255, 248, 220, 0.3));
                    border: 2px solid #c9a0dc; border-radius: 15px; padding: 25px; text-align: center;">
            <div style="font-size: 42px;">📚</div>
            <div style="font-size: 32px; font-weight: 900; color: #8b5fa8; margin: 10px 0;">{:,}</div>
            <div style="font-size: 12px; color: #6b4c7a; font-weight: 600;">TOTAL PAPERS</div>
        </div>
        """.format(len(df)), unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(201, 160, 220, 0.2), rgba(255, 248, 220, 0.3));
                    border: 2px solid #d4b0e8; border-radius: 15px; padding: 25px; text-align: center;">
            <div style="font-size: 42px;">📝</div>
            <div style="font-size: 32px; font-weight: 900; color: #8b5fa8; margin: 10px 0;">{:,}</div>
            <div style="font-size: 12px; color: #6b4c7a; font-weight: 600;">AVG WORDS</div>
        </div>
        """.format(int(df['word_count'].mean())), unsafe_allow_html=True)
    
    with col3:
        authors = set([a for auths in df['authors'] for a in auths])
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(201, 160, 220, 0.2), rgba(255, 248, 220, 0.3));
                    border: 2px solid #e8d4f5; border-radius: 15px; padding: 25px; text-align: center;">
            <div style="font-size: 42px;">👥</div>
            <div style="font-size: 32px; font-weight: 900; color: #8b5fa8; margin: 10px 0;">{:,}</div>
            <div style="font-size: 12px; color: #6b4c7a; font-weight: 600;">AUTHORS</div>
        </div>
        """.format(len(authors)), unsafe_allow_html=True)
    
    with col4:
        recent = sum(df['update_date'] > (datetime.now() - timedelta(days=365)))
        st.markdown("""
        <div style="background: linear-gradient(135deg, rgba(144, 198, 149, 0.2), rgba(255, 248, 220, 0.3));
                    border: 2px solid #90c695; border-radius: 15px; padding: 25px; text-align: center;">
            <div style="font-size: 42px;">🆕</div>
            <div style="font-size: 32px; font-weight: 900; color: #8b5fa8; margin: 10px 0;">{:,}</div>
            <div style="font-size: 12px; color: #6b4c7a; font-weight: 600;">THIS YEAR</div>
        </div>
        """.format(recent), unsafe_allow_html=True)

def main():
    """Main application"""
    
    # Initialize session state
    if 'show_similar' not in st.session_state:
        st.session_state.show_similar = False
    
    df, error = load_data()
    if df is not None:
        display_sidebar_stats(df)
    
    st.sidebar.markdown("---")
    
    pages = {
        "🔍 Search Papers": search_page,
        "📊 Analytics": analytics_page
    }
    
    selected = st.sidebar.radio("Navigation", list(pages.keys()), label_visibility="collapsed")
    
    pages[selected]()
    
    st.markdown("""
    <div style="text-align: center; padding: 35px; margin-top: 60px; 
                border-top: 3px solid rgba(201, 160, 220, 0.3);">
        <p style="color: #8b5fa8; font-size: 15px; font-weight: 600; margin: 0;">
            🚀 Powered by <strong>FAISS</strong> • <strong>OpenAI</strong> • <strong>Streamlit</strong>
        </p>
        <p style="color: rgba(107, 76, 122, 0.6); font-size: 13px; margin: 12px 0 0 0;">
            Made with ❤️ for researchers worldwide
        </p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()