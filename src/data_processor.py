"""
Data processing module for ArXiv dataset
Handles loading, cleaning, and preprocessing of research papers
Updated to handle JSON array format (not JSONL)
"""
import json
import pandas as pd
import numpy as np
import re
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Iterator
from datetime import datetime
import logging
from tqdm import tqdm

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from textblob import TextBlob
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException

# Try ijson for streaming large JSON files
try:
    import ijson
    HAS_IJSON = True
except ImportError:
    HAS_IJSON = False
    print("Warning: ijson not installed. Install with: pip install ijson")

import config

# Setup logging
logging.basicConfig(level=getattr(logging, config.LOG_LEVEL), format=config.LOG_FORMAT)
logger = logging.getLogger(__name__)

class ArXivDataProcessor:
    """
    Processes ArXiv dataset for vector search
    Updated to handle JSON array format from Kaggle
    """
    
    def __init__(self):
        self.data_path = config.ARXIV_DATASET_PATH
        self.processed_path = config.PROCESSED_DATA_PATH
        self.sample_size = config.SAMPLE_SIZE
        self.included_categories = config.INCLUDED_CATEGORIES
        
        # Download NLTK data if not present
        self._download_nltk_data()
        
        # Initialize stopwords
        self.stop_words = set(stopwords.words('english'))
        
    def _download_nltk_data(self):
        """Download required NLTK data"""
        try:
            nltk.data.find('tokenizers/punkt')
        except LookupError:
            nltk.download('punkt')
        
        try:
            nltk.data.find('corpora/stopwords')
        except LookupError:
            nltk.download('stopwords')
    
    def load_raw_data(self) -> Iterator[Dict]:
        """
        Load ArXiv dataset - handles both JSON array and JSONL formats
        Uses streaming for memory efficiency with large files
        """
        if not self.data_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.data_path}")
        
        logger.info(f"Loading dataset from {self.data_path}")
        file_size_gb = self.data_path.stat().st_size / (1024**3)
        logger.info(f"File size: {file_size_gb:.2f} GB")
        
        # Try streaming with ijson first (most memory efficient)
        if HAS_IJSON:
            try:
                yield from self._load_with_ijson()
                return
            except Exception as e:
                logger.warning(f"ijson streaming failed: {e}. Trying direct JSON load...")
        
        # Fallback to direct JSON loading
        yield from self._load_direct_json()
    
    def _load_with_ijson(self) -> Iterator[Dict]:
        """Load using ijson streaming parser (memory efficient)"""
        logger.info("Using ijson streaming parser...")
        
        with open(self.data_path, 'rb') as file:
            parser = ijson.items(file, 'item')  # Assumes JSON array format
            
            for i, paper in enumerate(parser):
                if self.sample_size and i >= self.sample_size:
                    logger.info(f"Reached sample limit: {self.sample_size}")
                    break
                
                yield paper
                
                if (i + 1) % 10000 == 0:
                    logger.info(f"Loaded {i + 1} papers...")
    
    def _load_direct_json(self) -> Iterator[Dict]:
        """Fallback: Load entire JSON file into memory"""
        logger.warning("Loading entire JSON file into memory. This may use significant RAM.")
        
        try:
            with open(self.data_path, 'r', encoding='utf-8') as file:
                # Check if it's JSON array or JSONL
                first_char = file.read(1)
                file.seek(0)
                
                if first_char == '[':
                    # JSON array format
                    logger.info("Detected JSON array format")
                    data = json.load(file)
                    
                    if self.sample_size:
                        data = data[:self.sample_size]
                        logger.info(f"Using sample of {len(data)} papers")
                    
                    for paper in data:
                        yield paper
                
                else:
                    # JSONL format
                    logger.info("Detected JSONL format")
                    for line_num, line in enumerate(file):
                        if self.sample_size and line_num >= self.sample_size:
                            break
                        
                        try:
                            paper = json.loads(line.strip())
                            yield paper
                        except json.JSONDecodeError as e:
                            logger.warning(f"Skipping malformed JSON at line {line_num}: {e}")
                            continue
        
        except MemoryError:
            logger.error("Out of memory! Try installing ijson: pip install ijson")
            raise
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            raise
    
    def clean_text(self, text: str) -> str:
        """Clean and preprocess text"""
        if not isinstance(text, str):
            return ""
        
        # Remove LaTeX commands and math expressions
        text = re.sub(r'\\[a-zA-Z]+\{[^}]*\}', '', text)
        text = re.sub(r'\$[^$]*\$', '', text)
        text = re.sub(r'\$\$[^$]*\$\$', '', text)
        
        # Remove URLs, emails, and special characters
        text = re.sub(r'http\S+|www\S+|https\S+', '', text)
        text = re.sub(r'\S+@\S+', '', text)
        
        # Clean up whitespace and newlines
        text = re.sub(r'\s+', ' ', text)
        text = text.replace('\n', ' ')
        
        # Remove extra spaces and trim
        text = ' '.join(text.split())
        
        return text.strip()
    
    def extract_categories(self, categories_str: str) -> List[str]:
        """Extract and clean category information"""
        if not categories_str:
            return []
        
        # Categories are space-separated in ArXiv format
        categories = categories_str.strip().split()
        
        # Clean each category
        cleaned_categories = []
        for cat in categories:
            cat = cat.strip()
            if cat and cat not in cleaned_categories:
                cleaned_categories.append(cat)  # Keep original case
        
        return cleaned_categories
    
    def extract_authors(self, paper: Dict) -> List[str]:
        """Extract author names from paper data"""
        authors = []
        
        # Try authors_parsed field first (structured format)
        if 'authors_parsed' in paper and paper['authors_parsed']:
            for author in paper['authors_parsed']:
                if isinstance(author, list) and len(author) >= 2:
                    # Format: [lastname, firstname, middle]
                    lastname = author[0].strip()
                    firstname = author[1].strip() if len(author) > 1 else ""
                    
                    if lastname and firstname:
                        full_name = f"{firstname} {lastname}".strip()
                        authors.append(full_name)
                    elif lastname:
                        authors.append(lastname)
        
        # Fallback to authors field if available
        elif 'authors' in paper and paper['authors']:
            raw_authors = paper['authors']
            # Simple split by comma and clean
            for author in raw_authors.split(','):
                author = author.strip()
                if author:
                    authors.append(author)
        
        return authors[:10]  # Limit to 10 authors
    
    def is_english_text(self, text: str) -> bool:
        """Detect if text is in English"""
        if len(text.strip()) < 20:
            return True  # Assume short text is English
        
        try:
            # Sample first 200 characters for detection
            sample = text[:200]
            return detect(sample) == 'en'
        except (LangDetectError, Exception):
            return True  # Assume English if detection fails
    
    def should_include_paper(self, paper: Dict) -> bool:
        """Determine if paper should be included based on filters"""
        
        # Check required fields
        if not paper.get('id', '').strip():
            return False
        
        title = paper.get('title', '').strip()
        if not title:
            return False
        
        abstract = paper.get('abstract', '').strip()
        if not abstract:
            return False
        
        # Check abstract length
        if len(abstract) < config.MIN_ABSTRACT_LENGTH:
            return False
        
        if len(abstract) > config.MAX_ABSTRACT_LENGTH:
            return False
        
        # Check language
        if config.LANGUAGE_FILTER == 'en' and not self.is_english_text(abstract):
            return False
        
        # Check categories
        if self.included_categories:
            paper_categories = self.extract_categories(paper.get('categories', ''))
            if not any(cat in self.included_categories for cat in paper_categories):
                return False
        
        return True
    
    def process_paper(self, paper: Dict) -> Optional[Dict]:
        """Process a single paper"""
        if not self.should_include_paper(paper):
            return None
        
        # Extract and clean fields
        paper_id = paper.get('id', '').strip()
        title = self.clean_text(paper.get('title', ''))
        abstract = self.clean_text(paper.get('abstract', ''))
        
        # Extract authors
        authors = self.extract_authors(paper)
        
        # Extract categories
        categories = self.extract_categories(paper.get('categories', ''))
        
        # Parse date
        update_date = paper.get('update_date', '')
        parsed_date = None
        if update_date:
            try:
                parsed_date = datetime.strptime(update_date, '%Y-%m-%d')
            except (ValueError, TypeError):
                # Try alternative date formats
                try:
                    parsed_date = datetime.strptime(update_date[:10], '%Y-%m-%d')
                except:
                    pass
        
        # Create combined text for embedding
        combined_text = f"{title}. {abstract}"
        
        # Additional fields
        processed_paper = {
            'id': paper_id,
            'title': title,
            'abstract': abstract,
            'combined_text': combined_text,
            'authors': authors,
            'categories': categories,
            'update_date': parsed_date,
            'submitter': paper.get('submitter', ''),
            'doi': paper.get('doi', ''),
            'journal_ref': paper.get('journal-ref', ''),
            'text_length': len(combined_text),
            'word_count': len(combined_text.split())
        }
        
        return processed_paper
    
    def process_dataset(self, save_processed: bool = True) -> pd.DataFrame:
        """Process entire dataset"""
        logger.info("Starting dataset processing...")
        
        # Check if processed data already exists
        if self.processed_path.exists():
            logger.info(f"Loading existing processed data from {self.processed_path}")
            try:
                with open(self.processed_path, 'rb') as f:
                    df = pickle.load(f)
                logger.info(f"Loaded {len(df)} processed papers")
                return df
            except Exception as e:
                logger.warning(f"Failed to load existing processed data: {e}")
                logger.info("Reprocessing dataset...")
        
        processed_papers = []
        total_papers = 0
        skipped_papers = 0
        
        # Process papers with progress tracking
        try:
            for paper in tqdm(self.load_raw_data(), desc="Processing papers"):
                total_papers += 1
                processed_paper = self.process_paper(paper)
                
                if processed_paper:
                    processed_papers.append(processed_paper)
                else:
                    skipped_papers += 1
                
                # Progress update every 10000 papers
                if total_papers % 10000 == 0:
                    logger.info(f"Processed {total_papers} papers, kept {len(processed_papers)}")
                    
                    # Optional: Save intermediate results for very large datasets
                    if total_papers % 50000 == 0 and save_processed:
                        temp_path = self.processed_path.with_suffix('.temp.pkl')
                        with open(temp_path, 'wb') as f:
                            temp_df = pd.DataFrame(processed_papers)
                            pickle.dump(temp_df, f)
                        logger.info(f"Saved intermediate results to {temp_path}")
        
        except KeyboardInterrupt:
            logger.info("Processing interrupted by user")
        except Exception as e:
            logger.error(f"Error during processing: {e}")
            raise
        
        logger.info(f"Dataset processing complete:")
        logger.info(f"  Total papers processed: {total_papers}")
        logger.info(f"  Papers kept: {len(processed_papers)}")
        logger.info(f"  Papers skipped: {skipped_papers}")
        logger.info(f"  Success rate: {len(processed_papers)/total_papers*100:.1f}%")
        
        # Convert to DataFrame
        df = pd.DataFrame(processed_papers)
        
        # Save processed data
        if save_processed and len(df) > 0:
            logger.info(f"Saving processed data to {self.processed_path}")
            try:
                with open(self.processed_path, 'wb') as f:
                    pickle.dump(df, f)
                logger.info("Successfully saved processed data")
                
                # Remove temporary file if it exists
                temp_path = self.processed_path.with_suffix('.temp.pkl')
                if temp_path.exists():
                    temp_path.unlink()
            except Exception as e:
                logger.error(f"Failed to save processed data: {e}")
        
        return df
    
    def get_statistics(self, df: pd.DataFrame) -> Dict:
        """Generate dataset statistics"""
        if len(df) == 0:
            return {"error": "No data to analyze"}
        
        stats = {
            'total_papers': len(df),
            'avg_text_length': df['text_length'].mean(),
            'avg_word_count': df['word_count'].mean(),
            'unique_authors': len(set([author for authors in df['authors'] for author in authors])),
            'date_range': {
                'earliest': df[df['update_date'].notna()]['update_date'].min() if df['update_date'].notna().any() else None,
                'latest': df[df['update_date'].notna()]['update_date'].max() if df['update_date'].notna().any() else None
            },
            'categories_distribution': {},
            'top_categories': []
        }
        
        # Category statistics
        all_categories = []
        for cats in df['categories']:
            if isinstance(cats, list):
                all_categories.extend(cats)
        
        if all_categories:
            from collections import Counter
            category_counts = Counter(all_categories)
            stats['categories_distribution'] = dict(category_counts)
            stats['top_categories'] = category_counts.most_common(20)
        
        return stats

def main():
    """Main function for testing data processing"""
    processor = ArXivDataProcessor()
    
    # Process dataset
    df = processor.process_dataset()
    
    if len(df) == 0:
        print("No papers were processed. Check your dataset and filters.")
        return
    
    # Print statistics
    stats = processor.get_statistics(df)
    
    print(f"\n=== Dataset Statistics ===")
    print(f"Total papers: {stats['total_papers']:,}")
    print(f"Average text length: {stats['avg_text_length']:.1f} characters")
    print(f"Average word count: {stats['avg_word_count']:.1f} words")
    print(f"Unique authors: {stats['unique_authors']:,}")
    
    if stats['date_range']['earliest']:
        print(f"Date range: {stats['date_range']['earliest'].strftime('%Y-%m-%d')} to {stats['date_range']['latest'].strftime('%Y-%m-%d')}")
    
    if stats['top_categories']:
        print(f"\nTop 10 categories:")
        for cat, count in stats['top_categories'][:10]:
            print(f"  {cat}: {count:,}")
    
    # Show sample papers
    print(f"\n=== Sample Papers ===")
    for idx, (_, paper) in enumerate(df.head(3).iterrows()):
        print(f"\nPaper {idx + 1}:")
        print(f"  ID: {paper['id']}")
        print(f"  Title: {paper['title'][:80]}...")
        print(f"  Categories: {', '.join(paper['categories'][:5])}")
        print(f"  Authors: {', '.join(paper['authors'][:3])}")
        print(f"  Text length: {paper['text_length']} characters")
        print(f"  Word count: {paper['word_count']}")

if __name__ == "__main__":
    main()