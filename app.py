"""
Streamlit web application for ArXiv Vector Search System
Enhanced with PDF download functionality and Individual AI Summaries
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
import tempfile
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
    page_title=config.APP_TITLE,
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        
        search_engine = ArXivSearchEngine(df)
        return search_engine, None
    except Exception as e:
        return None, str(e)

def download_single_pdf_streamlit(arxiv_id: str, title: str = None) -> dict:
    """Download a single PDF for Streamlit download button"""
    if not validate_arxiv_id(arxiv_id):
        return {'success': False, 'error': 'Invalid ArXiv ID'}
    
    pdf_url = f"https://arxiv.org/pdf/{arxiv_id}.pdf"
    
    try:
        import requests
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        
        response = requests.get(pdf_url, headers=headers, stream=True, timeout=30)
        response.raise_for_status()
        
        pdf_data = response.content
        
        if len(pdf_data) > 0:
            # Create a clean filename
            if title:
                clean_title = "".join(c for c in title if c.isalnum() or c in (' ', '-', '_')).strip()
                clean_title = clean_title[:50]  # Limit length
                filename = f"{arxiv_id}_{clean_title}.pdf"
            else:
                filename = f"{arxiv_id}.pdf"
            
            return {
                'success': True,
                'data': pdf_data,
                'filename': filename,
                'size': len(pdf_data)
            }
        else:
            return {'success': False, 'error': 'Downloaded file is empty'}
            
    except Exception as e:
        return {'success': False, 'error': f'Download failed: {str(e)}'}

def create_zip_download_streamlit(selected_results: list) -> dict:
    """Create ZIP file with selected PDFs for Streamlit"""
    if not selected_results:
        return {'success': False, 'error': 'No papers selected'}
    
    # Create a temporary ZIP file in memory
    import io
    zip_buffer = io.BytesIO()
    
    successful_downloads = []
    failed_downloads = []
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for i, result in enumerate(selected_results):
            arxiv_id = result.get('arxiv_id') or extract_arxiv_id(result.get('document_id', ''))
            
            if not arxiv_id:
                failed_downloads.append({
                    'title': result.get('title', 'Unknown'),
                    'error': 'No ArXiv ID available'
                })
                continue
            
            status_text.text(f"Downloading {arxiv_id}... ({i+1}/{len(selected_results)})")
            
            download_result = download_single_pdf_streamlit(arxiv_id, result.get('title'))
            
            if download_result['success']:
                # Add to ZIP
                zip_file.writestr(download_result['filename'], download_result['data'])
                successful_downloads.append({
                    'arxiv_id': arxiv_id,
                    'title': result.get('title', 'Unknown'),
                    'size_mb': download_result['size'] / (1024 * 1024)
                })
            else:
                failed_downloads.append({
                    'title': result.get('title', 'Unknown'),
                    'arxiv_id': arxiv_id,
                    'error': download_result['error']
                })
            
            progress_bar.progress((i + 1) / len(selected_results))
    
    status_text.empty()
    progress_bar.empty()
    
    if successful_downloads:
        zip_buffer.seek(0)
        total_size = sum(d['size_mb'] for d in successful_downloads)
        
        return {
            'success': True,
            'data': zip_buffer.getvalue(),
            'filename': f'arxiv_papers_{datetime.now().strftime("%Y%m%d_%H%M%S")}.zip',
            'successful': successful_downloads,
            'failed': failed_downloads,
            'total_size_mb': total_size
        }
    else:
        return {
            'success': False,
            'error': 'No PDFs could be downloaded',
            'failed': failed_downloads
        }

def display_search_result(result: dict, index: int, show_download: bool = True):
    """Display a single search result with AI summary, abstract, and download options"""
    with st.container():
        # Main result container
        col1, col2, col3 = st.columns([0.1, 0.7, 0.2])
        
        with col1:
            st.markdown(f"**{index + 1}**")
        
        with col2:
            # Extract ArXiv ID
            arxiv_id = result.get('arxiv_id') or extract_arxiv_id(result.get('document_id', ''))
            
            # Title with link
            if arxiv_id:
                st.markdown(f"**[{result['title']}](https://arxiv.org/abs/{arxiv_id})**")
            else:
                st.markdown(f"**{result['title']}**")
            
            # Metadata row
            col_meta1, col_meta2, col_meta3 = st.columns([0.3, 0.4, 0.3])
            
            with col_meta1:
                st.markdown(f"**Similarity:** {result['similarity_score']:.3f}")
            
            with col_meta2:
                st.markdown(f"**Categories:** {format_categories(result['categories'], 2)}")
            
            with col_meta3:
                if result.get('update_date'):
                    try:
                        date_str = pd.to_datetime(result['update_date']).strftime('%Y-%m-%d')
                        st.markdown(f"**Date:** {date_str}")
                    except:
                        st.markdown("**Date:** Unknown")
            
            # Authors
            if result.get('authors'):
                st.markdown(f"**Authors:** {format_authors(result['authors'], 3)}")
            
            # AI Summary Section (New Feature)
            if result.get('ai_summary'):
                st.markdown("### 🤖 AI Summary")
                st.info(result['ai_summary'])
                
                # Show abstract in expandable section when AI summary is available
                if result.get('abstract'):
                    with st.expander("📄 View Full Abstract", expanded=False):
                        st.markdown(result['abstract'])
            
            else:
                # Show abstract directly if no AI summary
                abstract = result.get('abstract', 'No abstract available')
                if len(abstract) > 300:
                    with st.expander("📄 Abstract", expanded=False):
                        st.markdown(abstract)
                else:
                    st.markdown(f"**Abstract:** {abstract}")
        
        with col3:
            # Download and link section
            if show_download and arxiv_id:
                # PDF availability indicator
                if result.get('pdf_available'):
                    st.success("✅ PDF Available")
                else:
                    st.warning("⚠️ PDF Status Unknown")
                
                # PDF download button
                if st.button(f"📥 Download PDF", key=f"download_{index}", help=f"Download {arxiv_id}.pdf"):
                    with st.spinner(f"Downloading {arxiv_id}.pdf..."):
                        download_result = download_single_pdf_streamlit(arxiv_id, result.get('title'))
                        
                        if download_result['success']:
                            st.download_button(
                                label=f"💾 Save PDF ({format_file_size(download_result['size'])})",
                                data=download_result['data'],
                                file_name=download_result['filename'],
                                mime="application/pdf",
                                key=f"save_{index}",
                                help=f"Click to save {download_result['filename']}"
                            )
                            st.success("Ready to download!")
                        else:
                            st.error(f"❌ {download_result['error']}")
                
                # ArXiv link
                st.link_button("🔗 View on ArXiv", f"https://arxiv.org/abs/{arxiv_id}", help="Open paper on ArXiv", use_container_width=True)
            
            elif not arxiv_id:
                st.warning("No ArXiv ID available")
        
        st.divider()

def display_ai_summary_stats():
    """Display AI summary configuration and cost information"""
    if config.ENABLE_INDIVIDUAL_SUMMARIES:
        st.sidebar.markdown("### 🤖 AI Summary Settings")
        st.sidebar.success("✅ Individual summaries enabled")
        
        # Show configuration
        st.sidebar.markdown(f"**Model:** {config.OPENAI_MODEL}")
        st.sidebar.markdown(f"**Max summaries:** {config.MAX_SUMMARIES_PER_SEARCH}")
        st.sidebar.markdown(f"**Max tokens:** {config.INDIVIDUAL_SUMMARY_MAX_TOKENS}")
        
        # Estimate cost
        estimated_cost = config.estimate_cost_per_search()
        st.sidebar.markdown(f"**Est. cost per search:** ~${estimated_cost:.4f}")
        
        if config.ENABLE_SUMMARY_CACHING:
            st.sidebar.markdown("💾 Caching enabled")
    else:
        st.sidebar.markdown("### 🤖 AI Summary Settings")
        st.sidebar.warning("❌ Individual summaries disabled")

def display_statistics_sidebar(df: pd.DataFrame):
    """Display dataset statistics in sidebar"""
    st.sidebar.markdown("### 📊 Dataset Statistics")
    
    total_papers = len(df)
    st.sidebar.metric("Total Papers", f"{total_papers:,}")
    
    # Date range
    if 'update_date' in df.columns and df['update_date'].notna().any():
        min_date = df['update_date'].min()
        max_date = df['update_date'].max()
        st.sidebar.markdown(f"**Date Range:** {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    
    # Top categories
    all_categories = []
    for cats in df['categories']:
        if isinstance(cats, list):
            all_categories.extend(cats)
    
    if all_categories:
        from collections import Counter
        top_cats = Counter(all_categories).most_common(10)
        
        st.sidebar.markdown("**Top Categories:**")
        for cat, count in top_cats[:5]:
            percentage = (count / total_papers) * 100
            st.sidebar.markdown(f"• {cat}: {count:,} ({percentage:.1f}%)")

def create_results_visualization(results: list):
    """Create visualizations for search results"""
    if not results:
        return None, None
    
    # Similarity scores distribution
    scores = [r['similarity_score'] for r in results]
    
    fig_scores = px.histogram(
        x=scores,
        nbins=20,
        title="Distribution of Similarity Scores",
        labels={'x': 'Similarity Score', 'y': 'Count'}
    )
    fig_scores.update_layout(showlegend=False)
    
    # Category distribution
    all_categories = []
    for result in results:
        all_categories.extend(result.get('categories', []))
    
    if all_categories:
        from collections import Counter
        cat_counts = Counter(all_categories)
        
        fig_categories = px.bar(
            x=list(cat_counts.values())[:10],
            y=list(cat_counts.keys())[:10],
            orientation='h',
            title="Top Categories in Results",
            labels={'x': 'Count', 'y': 'Category'}
        )
        fig_categories.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
    else:
        fig_categories = None
    
    return fig_scores, fig_categories

def search_page():
    """Main search page with AI summaries"""
    st.title("🔍 Re-Search with AI Summaries")
    st.markdown(config.APP_DESCRIPTION)
    
    # Load search engine
    search_engine, error = initialize_search_engine()
    
    if error:
        st.error(f"❌ Failed to initialize search engine: {error}")
        st.info("Please make sure you have processed the dataset and built the index.")
        return
    
    # Search interface
    col1, col2 = st.columns([0.7, 0.3])
    
    with col1:
        query = st.text_input(
            "🔎 Enter your search query:",
            placeholder="e.g., machine learning transformers, quantum computing algorithms",
            help="Use natural language to describe the research topics you're interested in"
        )
    
    with col2:
        search_button = st.button("🚀 Search Papers", type="primary", use_container_width=True)
    
    # Advanced options
    with st.expander("⚙️ Advanced Search Options"):
        col_adv1, col_adv2, col_adv3 = st.columns(3)
        
        with col_adv1:
            top_k = st.slider("Number of results", 5, 50, config.TOP_K_RESULTS)
            min_similarity = st.slider("Minimum similarity", 0.0, 1.0, config.SIMILARITY_THRESHOLD, 0.05)
        
        with col_adv2:
            # Category filter
            available_categories = [
                "cs.AI", "cs.LG", "cs.CV", "cs.CL", "cs.RO", "cs.NE",
                "stat.ML", "math.ST", "math.PR", "physics.data-an", "q-bio.QM"
            ]
            selected_categories = st.multiselect("Filter by categories:", available_categories)
        
        with col_adv3:
            # AI Summary options
            generate_summaries = st.checkbox("Generate AI Summaries", 
                                            value=config.ENABLE_INDIVIDUAL_SUMMARIES,
                                            help="Generate concise AI summaries for each paper (uses OpenAI API)")
            ai_overview = st.checkbox("Generate Overall Summary", value=True)
            export_format = st.selectbox("Export format:", ["None", "JSON", "CSV", "BibTeX"])
    
    # Perform search
    if search_button and query.strip():
        with st.spinner("🔍 Searching through research papers..."):
            try:
                search_kwargs = {
                    'top_k': top_k,
                    'min_similarity': min_similarity,
                    'generate_summaries': generate_summaries
                }
                
                if selected_categories:
                    search_kwargs['category_filter'] = selected_categories
                
                if ai_overview:
                    search_results = search_engine.search_with_ai_summary(query, **search_kwargs)
                    results = search_results['results']
                    ai_summary_text = search_results['ai_summary']
                    downloadable_count = search_results.get('downloadable_pdfs', 0)
                    ai_summaries_count = search_results.get('ai_summaries_count', 0)
                    estimated_cost = search_results.get('estimated_cost', 0.0)
                else:
                    search_results_obj = search_engine.search(query, **search_kwargs)
                    results = [r.to_dict() for r in search_results_obj]
                    ai_summary_text = None
                    downloadable_count = sum(1 for r in results if r.get('pdf_available', False))
                    ai_summaries_count = sum(1 for r in results if r.get('ai_summary'))
                    estimated_cost = 0.0
                
                # Store results in session state
                st.session_state.search_results = results
                st.session_state.search_query = query
                st.session_state.ai_summary = ai_summary_text
                st.session_state.downloadable_count = downloadable_count
                st.session_state.ai_summaries_count = ai_summaries_count
                st.session_state.estimated_cost = estimated_cost
                
            except Exception as e:
                st.error(f"❌ Search failed: {str(e)}")
                return
    
    # Display results
    if hasattr(st.session_state, 'search_results'):
        results = st.session_state.search_results
        query = st.session_state.search_query
        ai_summary_text = st.session_state.ai_summary
        downloadable_count = st.session_state.get('downloadable_count', 0)
        ai_summaries_count = st.session_state.get('ai_summaries_count', 0)
        estimated_cost = st.session_state.get('estimated_cost', 0.0)
        
        if not results:
            st.warning("🤷 No results found for your query. Try adjusting your search terms or filters.")
            return
        
        # Results header with enhanced info
        col_header1, col_header2 = st.columns([2, 1])
        
        with col_header1:
            st.success(f"✅ Found **{len(results)}** results for: **{query}**")
            
            # Enhanced metrics
            metrics_text = []
            if downloadable_count > 0:
                metrics_text.append(f"📄 {downloadable_count} PDFs available")
            if ai_summaries_count > 0:
                metrics_text.append(f"🤖 {ai_summaries_count} AI summaries")
            if estimated_cost > 0:
                metrics_text.append(f"💰 ~${estimated_cost:.4f} API cost")
            
            if metrics_text:
                st.markdown(" • ".join(metrics_text))
        
        with col_header2:
            # Quick stats
            if ai_summaries_count > 0:
                st.metric("AI Summaries", ai_summaries_count, f"{(ai_summaries_count/len(results)*100):.0f}% coverage")
        
        # Overall AI Summary
        if ai_summary_text:
            st.markdown("### 🧠 AI Research Overview")
            st.info(ai_summary_text)
        
        # Bulk download section
        st.markdown("### 📦 Bulk Operations")
        col_bulk1, col_bulk2, col_bulk3 = st.columns([2, 1, 1])
        
        with col_bulk1:
            # Multi-select for bulk download
            selected_indices = st.multiselect(
                "Select papers for bulk download:",
                range(len(results)),
                format_func=lambda x: f"[{x+1}] {results[x]['title'][:50]}..." if len(results[x]['title']) > 50 else f"[{x+1}] {results[x]['title']}",
                help="Select multiple papers to download as a single ZIP file"
            )
        
        with col_bulk2:
            if selected_indices and st.button("📦 Create ZIP", use_container_width=True):
                selected_results = [results[i] for i in selected_indices]
                zip_result = create_zip_download_streamlit(selected_results)
                
                if zip_result['success']:
                    st.session_state.zip_download = zip_result
                    st.success(f"✅ ZIP created ({len(zip_result['successful'])} papers)")
                    
                    if zip_result['failed']:
                        st.warning(f"⚠️ {len(zip_result['failed'])} papers failed")
                        with st.expander("Show failed downloads"):
                            for failed in zip_result['failed']:
                                st.text(f"❌ {failed.get('arxiv_id', 'Unknown')}: {failed['error']}")
                else:
                    st.error(f"❌ ZIP creation failed: {zip_result['error']}")
        
        with col_bulk3:
            # Show ZIP download button if available
            if hasattr(st.session_state, 'zip_download'):
                zip_data = st.session_state.zip_download
                st.download_button(
                    label=f"⬇️ Download ZIP ({format_file_size(len(zip_data['data']))})",
                    data=zip_data['data'],
                    file_name=zip_data['filename'],
                    mime="application/zip",
                    help=f"Download ZIP with {len(zip_data['successful'])} papers",
                    use_container_width=True
                )
        
        # Visualizations
        if len(results) > 3:  # Only show charts for meaningful datasets
            st.markdown("### 📊 Results Analysis")
            col_viz1, col_viz2 = st.columns(2)
            
            fig_scores, fig_categories = create_results_visualization(results)
            
            with col_viz1:
                if fig_scores:
                    st.plotly_chart(fig_scores, use_container_width=True)
            
            with col_viz2:
                if fig_categories:
                    st.plotly_chart(fig_categories, use_container_width=True)
        
        # Export functionality
        if export_format != "None":
            search_results_obj = [SearchResult(**{
                'document_id': r.get('document_id', ''),
                'title': r.get('title', ''),
                'abstract': r.get('abstract', ''),
                'authors': r.get('authors', []),
                'categories': r.get('categories', []),
                'similarity_score': r.get('similarity_score', 0.0),
                'update_date': pd.to_datetime(r['update_date']) if r.get('update_date') else None,
                'ai_summary': r.get('ai_summary', '')
            }) for r in results]
            
            exported_data = search_engine.export_search_results(search_results_obj, export_format.lower())
            
            filename = f"search_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format.lower()}"
            st.download_button(
                label=f"📁 Download {export_format}",
                data=exported_data,
                file_name=filename,
                mime="text/plain"
            )
        
        # Pagination
        results_per_page = config.PAGE_SIZE if hasattr(config, 'PAGE_SIZE') else 10
        total_pages = (len(results) + results_per_page - 1) // results_per_page
        
        if total_pages > 1:
            page = st.selectbox("📄 Page:", list(range(1, total_pages + 1)), key="results_page") - 1
            start_idx = page * results_per_page
            end_idx = min(start_idx + results_per_page, len(results))
        else:
            start_idx = 0
            end_idx = len(results)
        
        # Display individual results
        st.markdown("### 📚 Search Results")
        
        for i in range(start_idx, end_idx):
            display_search_result(results[i], i, show_download=True)

# Similar papers and analytics pages remain the same...
def similar_papers_page():
    """Similar papers finder page"""
    st.title("🔗 Find Similar Papers")
    st.markdown("Enter an ArXiv paper ID to find similar research papers.")
    
    # Load search engine
    search_engine, error = initialize_search_engine()
    
    if error:
        st.error(f"❌ Failed to initialize search engine: {error}")
        return
    
    # Input
    col1, col2 = st.columns([0.7, 0.3])
    
    with col1:
        paper_id = st.text_input(
            "ArXiv Paper ID:",
            placeholder="e.g., 2201.12345, cs.AI/0701123",
            help="Enter the ArXiv ID of a paper to find similar research"
        )
    
    with col2:
        col_sub1, col_sub2 = st.columns(2)
        with col_sub1:
            top_k = st.number_input("Similar papers:", 1, 20, 5)
        with col_sub2:
            with_summaries = st.checkbox("AI Summaries", value=True)
        find_button = st.button("🔍 Find Similar", type="primary", use_container_width=True)
    
    if find_button and paper_id.strip():
        with st.spinner("🔍 Finding similar papers..."):
            try:
                # Load data to show original paper
                df, _ = load_data()
                original_paper = df[df['id'] == paper_id.strip()]
                
                if original_paper.empty:
                    st.error(f"❌ Paper {paper_id} not found in the dataset.")
                    return
                
                # Show original paper
                original = original_paper.iloc[0]
                st.markdown("### 📄 Original Paper")
                
                with st.container():
                    st.markdown(f"**[{original['title']}](https://arxiv.org/abs/{original['id']})**")
                    st.markdown(f"**Categories:** {format_categories(original['categories'], 3)}")
                    st.markdown(f"**Authors:** {format_authors(original['authors'], 3)}")
                    
                    with st.expander("📄 Abstract", expanded=False):
                        st.markdown(original['abstract'])
                
                # Find similar papers
                similar_papers = search_engine.get_similar_papers(
                    paper_id.strip(), 
                    top_k=top_k,
                    generate_summaries=with_summaries
                )
                
                if not similar_papers:
                    st.warning("🤷 No similar papers found.")
                    return
                
                st.markdown(f"### 🔗 {len(similar_papers)} Similar Papers")
                
                # Convert to dict format for display
                results = [paper.to_dict() for paper in similar_papers]
                
                for i, result in enumerate(results):
                    display_search_result(result, i, show_download=True)
                
            except Exception as e:
                st.error(f"❌ Failed to find similar papers: {str(e)}")

def analytics_page():
    """Analytics and insights page (unchanged)"""
    st.title("📊 Analytics & Insights")
    
    # Load data
    df, error = load_data()
    search_engine, search_error = initialize_search_engine()
    
    if error:
        st.error(f"❌ Failed to load data: {error}")
        return
    
    # Dataset overview
    st.markdown("### 📈 Dataset Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Papers", f"{len(df):,}")
    
    with col2:
        avg_words = df['word_count'].mean() if 'word_count' in df.columns else 0
        st.metric("Avg Words", f"{avg_words:.0f}")
    
    with col3:
        unique_authors = len(set([author for authors in df['authors'] for author in authors]))
        st.metric("Unique Authors", f"{unique_authors:,}")
    
    with col4:
        if 'update_date' in df.columns:
            recent_papers = sum(df['update_date'] > (datetime.now() - timedelta(days=365)))
            st.metric("Papers (Last Year)", f"{recent_papers:,}")

def main():
    """Main application with AI summary features"""
    # Sidebar navigation
    st.sidebar.title("Data Dashboard")
    
    # Load data for sidebar stats
    df, error = load_data()
    if df is not None:
        display_statistics_sidebar(df)
        display_ai_summary_stats()  # New AI summary stats
    
    # Navigation
    pages = {
        "🔍 Search Papers": search_page,
        "🔗 Similar Papers": similar_papers_page,
        "📊 Analytics": analytics_page
    }
    
    selected_page = st.sidebar.selectbox("Navigate to:", list(pages.keys()))
    
    # System info in sidebar
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ System Info")
    st.sidebar.markdown(f"**Model:** {config.EMBEDDING_MODEL}")
    st.sidebar.markdown(f"**Index Type:** {config.FAISS_INDEX_TYPE}")
    
    # Configuration validation
    if config.USE_OPENAI and not config.OPENAI_API_KEY:
        st.sidebar.error("⚠️ OpenAI API key not configured")
        st.sidebar.markdown("Add your key to `.env` file")
    
    # Run selected page
    pages[selected_page]()
    
    # Footer
    st.markdown("---")
    st.markdown(
        "<div style='text-align: center; color: #666;'>"
        "Enhanced with AI Summaries • Built with Streamlit • Data from ArXiv • Powered by FAISS & OpenAI"
        "</div>", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()