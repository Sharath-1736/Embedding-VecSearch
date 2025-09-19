"""
Utility functions for the ArXiv Vector Search System
Enhanced with PDF download functionality
"""
import os
import json
import pickle
import logging
import requests
import re
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime, timedelta
import hashlib

import pandas as pd
import numpy as np
from tqdm import tqdm

import config

def setup_logging(level: str = "INFO", log_file: Optional[str] = None):
    """
    Setup logging configuration
    """
    log_level = getattr(logging, level.upper())
    
    # Create formatter
    formatter = logging.Formatter(config.LOG_FORMAT)
    
    # Setup root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)
    
    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)
    
    # File handler (optional)
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

def ensure_directories():
    """
    Ensure all required directories exist
    """
    directories = [
        config.DATA_DIR,
        config.MODELS_DIR,
        config.INDEX_DIR,
        Path(config.BASE_DIR) / "logs",
        Path(config.BASE_DIR) / "exports",
        Path(config.BASE_DIR) / "downloads"  # Add downloads directory
    ]
    
    for directory in directories:
        directory.mkdir(exist_ok=True, parents=True)
    
    logging.info(f"Ensured directories exist: {[str(d) for d in directories]}")

def load_json_lines(file_path: Path, max_lines: Optional[int] = None) -> List[Dict]:
    """
    Load JSON lines file efficiently
    """
    data = []
    
    with open(file_path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(tqdm(f, desc=f"Loading {file_path.name}")):
            if max_lines and i >= max_lines:
                break
            
            try:
                data.append(json.loads(line.strip()))
            except json.JSONDecodeError:
                logging.warning(f"Skipping malformed JSON at line {i+1}")
                continue
    
    return data

def save_json_lines(data: List[Dict], file_path: Path):
    """
    Save data as JSON lines file
    """
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in tqdm(data, desc=f"Saving {file_path.name}"):
            f.write(json.dumps(item) + '\n')

def load_pickle(file_path: Path) -> Any:
    """
    Load pickle file with error handling
    """
    try:
        with open(file_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        logging.error(f"Failed to load pickle file {file_path}: {e}")
        raise

def save_pickle(data: Any, file_path: Path):
    """
    Save data as pickle file with error handling
    """
    try:
        file_path.parent.mkdir(exist_ok=True, parents=True)
        with open(file_path, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"Saved pickle file: {file_path}")
    except Exception as e:
        logging.error(f"Failed to save pickle file {file_path}: {e}")
        raise

def get_file_hash(file_path: Path, algorithm: str = 'md5') -> str:
    """
    Get hash of a file
    """
    hash_func = hashlib.new(algorithm)
    
    with open(file_path, 'rb') as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_func.update(chunk)
    
    return hash_func.hexdigest()

def format_file_size(size_bytes: int) -> str:
    """
    Format file size in human readable format
    """
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    import math
    i = int(math.floor(math.log(size_bytes, 1024)))
    p = math.pow(1024, i)
    s = round(size_bytes / p, 2)
    
    return f"{s} {size_names[i]}"

def get_system_info() -> Dict:
    """
    Get system information for debugging
    """
    import platform
    import psutil
    import torch
    
    info = {
        'platform': platform.platform(),
        'python_version': platform.python_version(),
        'cpu_count': os.cpu_count(),
        'memory_total': format_file_size(psutil.virtual_memory().total),
        'memory_available': format_file_size(psutil.virtual_memory().available),
        'torch_version': torch.__version__,
        'cuda_available': torch.cuda.is_available(),
        'cuda_device_count': torch.cuda.device_count() if torch.cuda.is_available() else 0
    }
    
    if torch.cuda.is_available():
        info['cuda_version'] = torch.version.cuda
        info['gpu_names'] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
    
    return info

def validate_config() -> List[str]:
    """
    Validate configuration settings and return list of issues
    """
    issues = []
    
    # Check required files
    if not config.ARXIV_DATASET_PATH.exists():
        issues.append(f"Dataset file not found: {config.ARXIV_DATASET_PATH}")
    
    # Check embedding model
    try:
        from sentence_transformers import SentenceTransformer
        SentenceTransformer(config.EMBEDDING_MODEL)
    except Exception as e:
        issues.append(f"Failed to load embedding model '{config.EMBEDDING_MODEL}': {e}")
    
    # Check API keys if needed
    if config.USE_OPENAI and not config.OPENAI_API_KEY:
        issues.append("OpenAI API key not configured but USE_OPENAI is True")
    
    # Check numeric configurations
    if config.EMBEDDING_DIMENSION <= 0:
        issues.append(f"Invalid embedding dimension: {config.EMBEDDING_DIMENSION}")
    
    if config.BATCH_SIZE <= 0:
        issues.append(f"Invalid batch size: {config.BATCH_SIZE}")
    
    if config.TOP_K_RESULTS <= 0:
        issues.append(f"Invalid top-k results: {config.TOP_K_RESULTS}")
    
    return issues

# PDF Download Functions
def extract_arxiv_id(paper_id: str) -> Optional[str]:
    """
    Extract ArXiv ID from various formats (enhanced version)
    """
    if not paper_id:
        return None
    
    # Remove common prefixes and URLs
    paper_id = paper_id.replace('http://arxiv.org/abs/', '')
    paper_id = paper_id.replace('https://arxiv.org/abs/', '')
    paper_id = paper_id.replace('http://arxiv.org/pdf/', '')
    paper_id = paper_id.replace('https://arxiv.org/pdf/', '')
    paper_id = paper_id.replace('arXiv:', '')
    paper_id = paper_id.replace('.pdf', '')
    
    # ArXiv ID patterns - enhanced
    patterns = [
        r'^(\d{4}\.\d{4,5})(v\d+)?$',  # New format: 1234.5678v1
        r'^([a-z-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?$',  # Old format: cs.AI/0123456v1
    ]
    
    for pattern in patterns:
        match = re.match(pattern, paper_id, re.IGNORECASE)
        if match:
            return match.group(1)  # Return without version
    
    # If no pattern matches but it looks like an ArXiv ID, return as-is
    if re.match(r'^\d{4}\.\d{4,5}$', paper_id) or re.match(r'^[a-z-]+/\d{7}$', paper_id, re.IGNORECASE):
        return paper_id
    
    return None

def validate_arxiv_id(arxiv_id: str) -> bool:
    """
    Validate ArXiv ID format
    """
    if not arxiv_id:
        return False
    
    patterns = [
        r'^\d{4}\.\d{4,5}$',        # New format: 1234.5678
        r'^[a-z-]+(?:\.[A-Z]{2})?/\d{7}$',  # Old format: cs.AI/0123456
    ]
    
    return any(re.match(pattern, arxiv_id, re.IGNORECASE) for pattern in patterns)

def download_pdf(url: str, file_path: str, timeout: int = 30, chunk_size: int = 8192) -> bool:
    """
    Download PDF from URL with progress tracking
    
    Args:
        url: PDF URL
        file_path: Local path to save file
        timeout: Request timeout in seconds
        chunk_size: Download chunk size
        
    Returns:
        True if successful, False otherwise
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        }
        
        response = requests.get(url, headers=headers, stream=True, timeout=timeout)
        response.raise_for_status()
        
        # Check if it's actually a PDF
        content_type = response.headers.get('Content-Type', '')
        if 'pdf' not in content_type.lower() and 'application/octet-stream' not in content_type.lower():
            logging.warning(f"Content type is {content_type}, may not be PDF")
        
        total_size = int(response.headers.get('Content-Length', 0))
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'wb') as file:
            if total_size > 0:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Downloading {Path(file_path).name}") as pbar:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        if chunk:
                            file.write(chunk)
                            pbar.update(len(chunk))
            else:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        file.write(chunk)
        
        # Verify file was downloaded and has content
        if os.path.exists(file_path) and os.path.getsize(file_path) > 0:
            logging.info(f"Successfully downloaded: {file_path} ({format_file_size(os.path.getsize(file_path))})")
            return True
        else:
            logging.error(f"Downloaded file is empty or doesn't exist: {file_path}")
            return False
        
    except requests.exceptions.RequestException as e:
        logging.error(f"Download failed for {url}: {e}")
        return False
    except Exception as e:
        logging.error(f"Unexpected error during download: {e}")
        return False

def check_pdf_availability(pdf_url: str, timeout: int = 5) -> bool:
    """
    Check if PDF is available for download without downloading it
    """
    try:
        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
        }
        response = requests.head(pdf_url, headers=headers, timeout=timeout)
        return response.status_code == 200
    except:
        return False

def sanitize_filename(filename: str) -> str:
    """
    Sanitize filename for safe saving
    """
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    sanitized = re.sub(r'\s+', '_', sanitized)
    
    # Remove leading/trailing periods and spaces
    sanitized = sanitized.strip('. ')
    
    # Limit length
    if len(sanitized) > 100:
        sanitized = sanitized[:100]
    
    return sanitized

def create_download_summary(download_results: Dict[str, Any]) -> str:
    """
    Create a formatted summary of download results
    """
    successful = len(download_results.get('successful', []))
    failed = len(download_results.get('failed', []))
    total = successful + failed
    size_mb = download_results.get('total_size_mb', 0)
    
    summary = f"Download Summary:\n"
    summary += f"✅ Successful: {successful}/{total} papers\n"
    summary += f"❌ Failed: {failed}/{total} papers\n"
    summary += f"📁 Total size: {size_mb:.2f} MB\n"
    
    if failed > 0:
        summary += f"\nFailed downloads:\n"
        for failure in download_results.get('failed', []):
            arxiv_id = failure.get('arxiv_id', failure.get('id', 'Unknown'))
            error = failure.get('error', 'Unknown error')
            summary += f"  • {arxiv_id}: {error}\n"
    
    return summary

# Original utility functions
class Timer:
    """
    Context manager for timing code execution
    """
    
    def __init__(self, description: str = "Operation"):
        self.description = description
        self.start_time = None
        self.end_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        logging.info(f"Starting: {self.description}")
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.end_time = datetime.now()
        duration = self.end_time - self.start_time
        logging.info(f"Completed: {self.description} in {duration.total_seconds():.2f} seconds")
    
    @property
    def duration(self) -> Optional[timedelta]:
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None

class ProgressTracker:
    """
    Track progress of long-running operations
    """
    
    def __init__(self, total: int, description: str = "Processing"):
        self.total = total
        self.description = description
        self.current = 0
        self.start_time = datetime.now()
        self.pbar = tqdm(total=total, desc=description)
    
    def update(self, increment: int = 1):
        """Update progress"""
        self.current += increment
        self.pbar.update(increment)
    
    def set_description(self, description: str):
        """Update description"""
        self.pbar.set_description(description)
    
    def close(self):
        """Close progress bar"""
        self.pbar.close()
        duration = datetime.now() - self.start_time
        logging.info(f"Completed {self.description}: {self.current}/{self.total} in {duration.total_seconds():.2f}s")
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

def clean_text_for_display(text: str, max_length: int = 100) -> str:
    """
    Clean and truncate text for display purposes
    """
    if not isinstance(text, str):
        return str(text)
    
    # Remove excessive whitespace
    text = ' '.join(text.split())
    
    # Truncate if needed
    if len(text) > max_length:
        text = text[:max_length-3] + "..."
    
    return text

def format_authors(authors: List[str], max_authors: int = 3) -> str:
    """
    Format author list for display
    """
    if not authors:
        return "Unknown authors"
    
    if len(authors) <= max_authors:
        return ", ".join(authors)
    
    return f"{', '.join(authors[:max_authors])}, et al."

def format_categories(categories: List[str], max_categories: int = 3) -> str:
    """
    Format categories for display
    """
    if not categories:
        return "No categories"
    
    if len(categories) <= max_categories:
        return ", ".join(categories)
    
    return f"{', '.join(categories[:max_categories])}, +{len(categories) - max_categories} more"

def compute_text_statistics(texts: List[str]) -> Dict:
    """
    Compute basic statistics for a list of texts
    """
    if not texts:
        return {}
    
    lengths = [len(text) for text in texts]
    word_counts = [len(text.split()) for text in texts]
    
    return {
        'total_texts': len(texts),
        'total_characters': sum(lengths),
        'total_words': sum(word_counts),
        'avg_length': np.mean(lengths),
        'median_length': np.median(lengths),
        'avg_words': np.mean(word_counts),
        'median_words': np.median(word_counts),
        'min_length': min(lengths),
        'max_length': max(lengths),
    }

def create_sample_dataset(df: pd.DataFrame, sample_size: int, 
                         strategy: str = 'random') -> pd.DataFrame:
    """
    Create a sample dataset for testing
    """
    if len(df) <= sample_size:
        return df.copy()
    
    if strategy == 'random':
        return df.sample(n=sample_size, random_state=42)
    
    elif strategy == 'recent':
        # Sample most recent papers
        if 'update_date' in df.columns:
            return df.nlargest(sample_size, 'update_date')
        else:
            return df.tail(sample_size)
    
    elif strategy == 'diverse_categories':
        # Sample papers from diverse categories
        samples_per_category = max(1, sample_size // 20)  # Assume ~20 categories
        sampled_dfs = []
        
        for category in df['categories'].explode().value_counts().head(20).index:
            category_df = df[df['categories'].apply(lambda x: category in x)]
            if len(category_df) > 0:
                n_sample = min(samples_per_category, len(category_df))
                sampled_dfs.append(category_df.sample(n=n_sample, random_state=42))
        
        result_df = pd.concat(sampled_dfs, ignore_index=True).drop_duplicates(subset=['id'])
        
        if len(result_df) < sample_size:
            # Fill remaining with random samples
            remaining = sample_size - len(result_df)
            excluded_ids = set(result_df['id'])
            remaining_df = df[~df['id'].isin(excluded_ids)]
            if len(remaining_df) > 0:
                additional = remaining_df.sample(n=min(remaining, len(remaining_df)), random_state=42)
                result_df = pd.concat([result_df, additional], ignore_index=True)
        
        return result_df.head(sample_size)
    
    else:
        raise ValueError(f"Unknown sampling strategy: {strategy}")

def benchmark_function(func, *args, **kwargs):
    """
    Benchmark a function's execution time and memory usage
    """
    import psutil
    import gc
    
    # Force garbage collection
    gc.collect()
    
    # Get initial memory
    process = psutil.Process()
    initial_memory = process.memory_info().rss
    
    # Run function
    with Timer(f"Benchmarking {func.__name__}") as timer:
        result = func(*args, **kwargs)
    
    # Get final memory
    final_memory = process.memory_info().rss
    memory_used = final_memory - initial_memory
    
    benchmark_result = {
        'function': func.__name__,
        'execution_time': timer.duration.total_seconds(),
        'memory_used': format_file_size(memory_used),
        'memory_used_bytes': memory_used
    }
    
    logging.info(f"Benchmark result: {benchmark_result}")
    
    return result, benchmark_result

def export_config_summary() -> Dict:
    """
    Export current configuration as summary
    """
    return {
        'dataset_path': str(config.ARXIV_DATASET_PATH),
        'sample_size': config.SAMPLE_SIZE,
        'embedding_model': config.EMBEDDING_MODEL,
        'embedding_dimension': config.EMBEDDING_DIMENSION,
        'batch_size': config.BATCH_SIZE,
        'faiss_index_type': config.FAISS_INDEX_TYPE,
        'top_k_results': config.TOP_K_RESULTS,
        'use_openai': config.USE_OPENAI,
        'use_huggingface': config.USE_HUGGINGFACE,
        'included_categories': config.INCLUDED_CATEGORIES,
        'system_info': get_system_info()
    }

def main():
    """
    Main function for testing utilities including PDF download
    """
    # Setup logging
    setup_logging()
    ensure_directories()
    
    # Test configuration validation
    issues = validate_config()
    if issues:
        print("Configuration issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("Configuration validation passed!")
    
    # Test system info
    system_info = get_system_info()
    print(f"\nSystem Information:")
    for key, value in system_info.items():
        print(f"  {key}: {value}")
    
    # Test ArXiv ID extraction
    test_ids = [
        "2201.12345",
        "arXiv:2201.12345v1",
        "https://arxiv.org/abs/2201.12345",
        "https://arxiv.org/pdf/2201.12345.pdf",
        "cs.AI/0701123v1",
        "math-ph/0123456"
    ]
    
    print(f"\nTesting ArXiv ID extraction:")
    for test_id in test_ids:
        extracted = extract_arxiv_id(test_id)
        valid = validate_arxiv_id(extracted) if extracted else False
        print(f"  {test_id} -> {extracted} (valid: {valid})")
    
    # Test PDF download (with a known ArXiv paper)
    test_arxiv_id = "2201.12345"  # Replace with a known ID
    test_url = f"https://arxiv.org/pdf/{test_arxiv_id}.pdf"
    test_file = f"test_downloads/{test_arxiv_id}.pdf"
    
    print(f"\nTesting PDF availability check:")
    is_available = check_pdf_availability(test_url)
    print(f"  PDF available for {test_arxiv_id}: {is_available}")
    
    # Test timer
    with Timer("Test operation"):
        import time
        time.sleep(1)
    
    # Test config export
    config_summary = export_config_summary()
    print(f"\nConfiguration Summary:")
    for key, value in config_summary.items():
        if isinstance(value, dict):
            print(f"  {key}: {len(value)} items")
        else:
            print(f"  {key}: {value}")

if __name__ == "__main__":
    main()