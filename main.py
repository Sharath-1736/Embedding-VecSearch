"""
Command Line Interface for ArXiv Vector Search System
"""
import argparse
import sys
from pathlib import Path
import logging
from datetime import datetime

import pandas as pd

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from data_processor import ArXivDataProcessor
from embedding_generator import EmbeddingGenerator
from vector_store import FAISSVectorStore
from search_engine import ArXivSearchEngine
from utils import setup_logging, ensure_directories, Timer, validate_config
import config

def setup_cli():
    """Setup command line argument parser"""
    parser = argparse.ArgumentParser(
        description="ArXiv Vector Search System - Semantic search through research papers",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py process --sample-size 10000
  python main.py build-index
  python main.py search "machine learning neural networks"
  python main.py search "quantum computing" --top-k 20 --export json
  python main.py similar-papers 2201.12345
        """
    )
    
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose logging')
    parser.add_argument('--log-file', type=str, 
                       help='Log file path')
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Process command
    process_parser = subparsers.add_parser('process', help='Process ArXiv dataset')
    process_parser.add_argument('--sample-size', type=int, default=config.SAMPLE_SIZE,
                               help=f'Number of papers to process (default: {config.SAMPLE_SIZE})')
    process_parser.add_argument('--force', action='store_true',
                               help='Force reprocessing even if processed data exists')
    
    # Build index command
    build_parser = subparsers.add_parser('build-index', help='Build vector search index')
    build_parser.add_argument('--index-type', choices=['Flat', 'IVF', 'HNSW'],
                             default=config.FAISS_INDEX_TYPE,
                             help=f'FAISS index type (default: {config.FAISS_INDEX_TYPE})')
    build_parser.add_argument('--force', action='store_true',
                             help='Force rebuilding even if index exists')
    
    # Search command
    search_parser = subparsers.add_parser('search', help='Search research papers')
    search_parser.add_argument('query', type=str, help='Search query')
    search_parser.add_argument('--top-k', type=int, default=config.TOP_K_RESULTS,
                              help=f'Number of results to return (default: {config.TOP_K_RESULTS})')
    search_parser.add_argument('--min-similarity', type=float, default=config.SIMILARITY_THRESHOLD,
                              help=f'Minimum similarity score (default: {config.SIMILARITY_THRESHOLD})')
    search_parser.add_argument('--categories', nargs='+',
                              help='Filter by categories (e.g., cs.AI cs.LG)')
    search_parser.add_argument('--export', choices=['json', 'csv', 'bibtex'],
                              help='Export results in specified format')
    search_parser.add_argument('--ai-summary', action='store_true',
                              help='Generate AI summary of results')
    search_parser.add_argument('--output', type=str,
                              help='Output file path for exported results')
    
    # Similar papers command
    similar_parser = subparsers.add_parser('similar-papers', help='Find similar papers')
    similar_parser.add_argument('paper_id', type=str, help='ArXiv paper ID')
    similar_parser.add_argument('--top-k', type=int, default=5,
                               help='Number of similar papers to find (default: 5)')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Show system and dataset information')
    info_parser.add_argument('--dataset-stats', action='store_true',
                            help='Show dataset statistics')
    info_parser.add_argument('--index-stats', action='store_true',
                            help='Show index statistics')
    info_parser.add_argument('--trending', type=int, default=30,
                            help='Show trending categories (days back, default: 30)')
    
    # Validate command
    validate_parser = subparsers.add_parser('validate', help='Validate configuration')
    
    return parser

def cmd_process(args):
    """Process ArXiv dataset"""
    print("🔄 Processing dataset...")
    
    # Update config if needed
    if args.sample_size != config.SAMPLE_SIZE:
        config.SAMPLE_SIZE = args.sample_size
        print(f"📊 Sample size set to: {args.sample_size:,}")
    
    # Delete existing processed data if force flag is set
    if args.force and config.PROCESSED_DATA_PATH.exists():
        config.PROCESSED_DATA_PATH.unlink()
        print("🗑️  Removed existing processed data")
    
    # Process dataset
    processor = ArXivDataProcessor()
    
    with Timer("Dataset processing"):
        df = processor.process_dataset()
    
    # Show statistics
    stats = processor.get_statistics(df)
    
    print(f"\n📈 Processing Results:")
    print(f"   Total papers: {stats['total_papers']:,}")
    print(f"   Average text length: {stats['avg_text_length']:.0f} characters")
    print(f"   Average word count: {stats['avg_word_count']:.0f} words")
    print(f"   Unique authors: {stats['unique_authors']:,}")
    
    if stats['top_categories']:
        print(f"\n🏷️  Top categories:")
        for cat, count in stats['top_categories'][:5]:
            print(f"     {cat}: {count:,}")
    
    print("\n✅ Dataset processing completed!")

def cmd_build_index(args):
    """Build vector search index"""
    print("🏗️  Building vector search index...")
    
    # Load processed data
    processor = ArXivDataProcessor()
    
    if not config.PROCESSED_DATA_PATH.exists():
        print("❌ No processed data found. Run 'process' command first.")
        return
    
    df = processor.process_dataset()
    print(f"📊 Loaded {len(df):,} processed papers")
    
    # Delete existing index if force flag is set
    if args.force:
        index_files = [
            config.INDEX_DIR / f"faiss_index_{args.index_type}.index",
            config.INDEX_DIR / f"document_map_{args.index_type}.pkl",
            config.INDEX_DIR / f"embeddings_{config.EMBEDDING_MODEL.replace('/', '_')}.pkl",
            config.INDEX_DIR / f"metadata_{config.EMBEDDING_MODEL.replace('/', '_')}.pkl"
        ]
        
        for file_path in index_files:
            if file_path.exists():
                file_path.unlink()
                print(f"🗑️  Removed existing {file_path.name}")
    
    # Generate embeddings
    print("🧠 Generating embeddings...")
    generator = EmbeddingGenerator()
    
    with Timer("Embedding generation"):
        embeddings = generator.generate_embeddings_from_dataframe(df)
    
    print(f"✨ Generated embeddings shape: {embeddings.shape}")
    
    # Build vector index
    print(f"🔍 Building {args.index_type} index...")
    vector_store = FAISSVectorStore(embeddings.shape[1], args.index_type)
    
    document_ids = df['id'].tolist()
    
    with Timer("Index building"):
        vector_store.build_index(embeddings, document_ids)
        vector_store.save_index()
    
    # Show index stats
    stats = vector_store.get_stats()
    print(f"\n📊 Index Statistics:")
    for key, value in stats.items():
        print(f"     {key}: {value}")
    
    print("\n✅ Index building completed!")

def cmd_search(args):
    """Search research papers"""
    print(f"🔍 Searching for: '{args.query}'")
    
    # Load processed data
    processor = ArXivDataProcessor()
    
    if not config.PROCESSED_DATA_PATH.exists():
        print("❌ No processed data found. Run 'process' command first.")
        return
    
    df = processor.process_dataset()
    
    # Initialize search engine
    with Timer("Search engine initialization"):
        search_engine = ArXivSearchEngine(df)
    
    # Prepare search parameters
    search_kwargs = {
        'top_k': args.top_k,
        'min_similarity': args.min_similarity
    }
    
    if args.categories:
        search_kwargs['category_filter'] = args.categories
        print(f"🏷️  Filtering by categories: {', '.join(args.categories)}")
    
    # Perform search
    with Timer("Search execution"):
        if args.ai_summary:
            search_results = search_engine.search_with_ai_summary(args.query, **search_kwargs)
            results = [SearchResult(**r) for r in search_results['results']]
            ai_summary = search_results['ai_summary']
        else:
            results = search_engine.search(args.query, **search_kwargs)
            ai_summary = None
    
    # Display results
    if not results:
        print("❌ No results found.")
        return
    
    print(f"\n📊 Found {len(results)} results:")
    
    if ai_summary:
        print(f"\n🤖 AI Summary:")
        print(f"   {ai_summary}")
    
    print(f"\n📝 Search Results:")
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result.title}")
        print(f"   🎯 Similarity: {result.similarity_score:.4f}")
        print(f"   🏷️  Categories: {', '.join(result.categories[:3])}")
        print(f"   👤 Authors: {', '.join(result.authors[:3])}")
        if result.update_date:
            print(f"   📅 Date: {result.update_date.strftime('%Y-%m-%d')}")
        print(f"   📄 Abstract: {result.get_summary(150)}")
        print(f"   🔗 URL: https://arxiv.org/abs/{result.document_id}")
    
    # Export results if requested
    if args.export:
        output_path = args.output
        if not output_path:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = f"search_results_{timestamp}.{args.export}"
        
        exported_data = search_engine.export_search_results(results, args.export)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(exported_data)
        
        print(f"\n💾 Results exported to: {output_path}")

def cmd_similar_papers(args):
    """Find papers similar to a given paper"""
    print(f"🔍 Finding papers similar to: {args.paper_id}")
    
    # Load processed data
    processor = ArXivDataProcessor()
    
    if not config.PROCESSED_DATA_PATH.exists():
        print("❌ No processed data found. Run 'process' command first.")
        return
    
    df = processor.process_dataset()
    
    # Initialize search engine
    search_engine = ArXivSearchEngine(df)
    
    # Find original paper
    original_paper = df[df['id'] == args.paper_id]
    if original_paper.empty:
        print(f"❌ Paper {args.paper_id} not found in dataset.")
        return
    
    original = original_paper.iloc[0]
    print(f"\n📄 Original Paper:")
    print(f"   Title: {original['title']}")
    print(f"   Categories: {', '.join(original['categories'][:3])}")
    print(f"   Authors: {', '.join(original['authors'][:3])}")
    
    # Find similar papers
    with Timer("Similar papers search"):
        similar_papers = search_engine.get_similar_papers(args.paper_id, args.top_k)
    
    if not similar_papers:
        print("❌ No similar papers found.")
        return
    
    print(f"\n🔗 {len(similar_papers)} Similar Papers:")
    
    for i, paper in enumerate(similar_papers, 1):
        print(f"\n{i}. {paper.title}")
        print(f"   🎯 Similarity: {paper.similarity_score:.4f}")
        print(f"   🏷️  Categories: {', '.join(paper.categories[:3])}")
        print(f"   👤 Authors: {', '.join(paper.authors[:3])}")
        print(f"   📄 Abstract: {paper.get_summary(150)}")
        print(f"   🔗 URL: https://arxiv.org/abs/{paper.document_id}")

def cmd_info(args):
    """Show system and dataset information"""
    from utils import get_system_info, export_config_summary
    
    print("ℹ️  System Information")
    
    # System info
    system_info = get_system_info()
    print(f"\n🖥️  System:")
    for key, value in system_info.items():
        print(f"     {key}: {value}")
    
    # Configuration
    config_summary = export_config_summary()
    print(f"\n⚙️  Configuration:")
    for key, value in config_summary.items():
        if key != 'system_info':
            print(f"     {key}: {value}")
    
    # Dataset statistics
    if args.dataset_stats:
        if config.PROCESSED_DATA_PATH.exists():
            processor = ArXivDataProcessor()
            df = processor.process_dataset()
            stats = processor.get_statistics(df)
            
            print(f"\n📊 Dataset Statistics:")
            print(f"     Total papers: {stats['total_papers']:,}")
            print(f"     Average text length: {stats['avg_text_length']:.0f}")
            print(f"     Average word count: {stats['avg_word_count']:.0f}")
            print(f"     Unique authors: {stats['unique_authors']:,}")
            print(f"     Date range: {stats['date_range']['earliest']} to {stats['date_range']['latest']}")
            
            print(f"\n🏷️  Top Categories:")
            for cat, count in stats['top_categories'][:10]:
                print(f"     {cat}: {count:,}")
        else:
            print("\n❌ No processed dataset found. Run 'process' command first.")
    
    # Index statistics
    if args.index_stats:
        try:
            generator = EmbeddingGenerator()
            vector_store = FAISSVectorStore(generator.embedding_dim)
            vector_store.load_index()
            
            stats = vector_store.get_stats()
            print(f"\n🔍 Index Statistics:")
            for key, value in stats.items():
                print(f"     {key}: {value}")
        except Exception as e:
            print(f"\n❌ Could not load index: {e}")
    
    # Trending categories
    if args.trending:
        if config.PROCESSED_DATA_PATH.exists():
            processor = ArXivDataProcessor()
            df = processor.process_dataset()
            search_engine = ArXivSearchEngine(df)
            
            trending = search_engine.get_trending_categories(args.trending)
            if trending:
                print(f"\n📈 Trending Categories (last {trending['period_days']} days):")
                print(f"     Total papers: {trending['total_papers']:,}")
                
                for cat, count in list(trending['trending_categories'].items())[:10]:
                    print(f"     {cat}: {count:,}")
            else:
                print(f"\n❌ No trending data available for last {args.trending} days")
        else:
            print("\n❌ No processed dataset found. Run 'process' command first.")

def cmd_validate(args):
    """Validate system configuration"""
    print("🔧 Validating system configuration...")
    
    issues = validate_config()
    
    if not issues:
        print("✅ All configuration checks passed!")
        return
    
    print(f"❌ Found {len(issues)} configuration issues:")
    for i, issue in enumerate(issues, 1):
        print(f"  {i}. {issue}")
    
    print("\n💡 Please fix these issues before using the system.")

def main():
    """Main CLI entry point"""
    parser = setup_cli()
    args = parser.parse_args()
    
    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(log_level, args.log_file)
    
    # Ensure directories exist
    ensure_directories()
    
    # Handle commands
    if args.command == 'process':
        cmd_process(args)
    elif args.command == 'build-index':
        cmd_build_index(args)
    elif args.command == 'search':
        cmd_search(args)
    elif args.command == 'similar-papers':
        cmd_similar_papers(args)
    elif args.command == 'info':
        cmd_info(args)
    elif args.command == 'validate':
        cmd_validate(args)
    else:
        parser.print_help()

if __name__ == "__main__":
    main()