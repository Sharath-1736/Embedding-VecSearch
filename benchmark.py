"""
Benchmark script to compare search performance
Run this to generate metrics for your presentation
"""
import time
import numpy as np
from collections import defaultdict
from typing import List, Dict
import json
from datetime import datetime

from data_processor import ArXivDataProcessor
from search_engine import ArXivSearchEngine
import config

# Test queries for evaluation
TEST_QUERIES = [
    "transformer attention mechanisms deep learning",
    "graph neural networks molecular property prediction",
    "reinforcement learning robotic manipulation",
    "computer vision object detection convolutional networks",
    "natural language processing BERT language models",
    "generative adversarial networks image synthesis",
    "quantum computing machine learning",
    "federated learning privacy preserving",
    "neural architecture search automl",
    "transfer learning few shot learning",
    "self supervised learning representation",
    "meta learning model agnostic",
    "explainable artificial intelligence interpretability",
    "adversarial examples robustness",
    "continuous learning catastrophic forgetting"
]


class SearchBenchmark:
    """Benchmark search engine performance"""
    
    def __init__(self, search_engine: ArXivSearchEngine):
        self.search_engine = search_engine
        self.results = defaultdict(list)
    
    def benchmark_latency(self, queries: List[str], top_k: int = 10, 
                         use_reranking: bool = True) -> Dict:
        """Measure search latency"""
        print(f"\n📊 Benchmarking Latency (reranking={use_reranking})...")
        
        latencies = []
        
        for i, query in enumerate(queries, 1):
            start_time = time.time()
            results = self.search_engine.search(
                query, 
                top_k=top_k,
                use_reranking=use_reranking,
                use_query_expansion=True
            )
            elapsed = (time.time() - start_time) * 1000  # Convert to ms
            
            latencies.append(elapsed)
            print(f"  Query {i}/{len(queries)}: {elapsed:.1f}ms")
        
        return {
            'mean_latency_ms': np.mean(latencies),
            'median_latency_ms': np.median(latencies),
            'std_latency_ms': np.std(latencies),
            'min_latency_ms': np.min(latencies),
            'max_latency_ms': np.max(latencies),
            'total_queries': len(latencies)
        }
    
    def benchmark_relevance_scores(self, queries: List[str], top_k: int = 10) -> Dict:
        """Measure similarity score distribution"""
        print(f"\n📊 Benchmarking Relevance Scores...")
        
        all_scores = []
        top1_scores = []
        top5_avg_scores = []
        
        for i, query in enumerate(queries, 1):
            results = self.search_engine.search(query, top_k=top_k)
            
            if results:
                scores = [r.similarity_score for r in results]
                all_scores.extend(scores)
                top1_scores.append(scores[0])
                top5_avg_scores.append(np.mean(scores[:5]))
            
            print(f"  Query {i}/{len(queries)}: Top score = {scores[0]:.4f}")
        
        return {
            'mean_all_scores': np.mean(all_scores),
            'mean_top1_score': np.mean(top1_scores),
            'mean_top5_score': np.mean(top5_avg_scores),
            'min_score': np.min(all_scores),
            'max_score': np.max(all_scores),
            'scores_above_0.7': sum(1 for s in all_scores if s > 0.7) / len(all_scores),
            'scores_above_0.5': sum(1 for s in all_scores if s > 0.5) / len(all_scores)
        }
    
    def benchmark_result_diversity(self, queries: List[str], top_k: int = 10) -> Dict:
        """Measure diversity of results"""
        print(f"\n📊 Benchmarking Result Diversity...")
        
        category_diversity = []
        author_diversity = []
        
        for i, query in enumerate(queries, 1):
            results = self.search_engine.search(query, top_k=top_k)
            
            if results:
                # Category diversity (unique categories in top-k)
                all_categories = set()
                for r in results:
                    all_categories.update(r.categories)
                category_diversity.append(len(all_categories))
                
                # Author diversity (unique authors in top-k)
                all_authors = set()
                for r in results:
                    all_authors.update(r.authors)
                author_diversity.append(len(all_authors))
            
            print(f"  Query {i}/{len(queries)}: {len(all_categories)} categories, {len(all_authors)} authors")
        
        return {
            'mean_category_diversity': np.mean(category_diversity),
            'mean_author_diversity': np.mean(author_diversity),
            'min_categories': np.min(category_diversity),
            'max_categories': np.max(category_diversity)
        }
    
    def compare_with_without_reranking(self, queries: List[str], top_k: int = 10) -> Dict:
        """Compare results with and without re-ranking"""
        print(f"\n📊 Comparing With/Without Re-ranking...")
        
        agreement_scores = []
        score_improvements = []
        
        for i, query in enumerate(queries, 1):
            # Without re-ranking
            results_no_rerank = self.search_engine.search(
                query, 
                top_k=top_k,
                use_reranking=False
            )
            ids_no_rerank = [r.document_id for r in results_no_rerank]
            scores_no_rerank = [r.similarity_score for r in results_no_rerank]
            
            # With re-ranking
            results_rerank = self.search_engine.search(
                query, 
                top_k=top_k,
                use_reranking=True
            )
            ids_rerank = [r.document_id for r in results_rerank]
            scores_rerank = [r.similarity_score for r in results_rerank]
            
            # Calculate agreement (how many papers in common in top-k)
            agreement = len(set(ids_no_rerank) & set(ids_rerank)) / top_k
            agreement_scores.append(agreement)
            
            # Calculate score improvement
            if scores_rerank and scores_no_rerank:
                improvement = (scores_rerank[0] - scores_no_rerank[0]) / scores_no_rerank[0]
                score_improvements.append(improvement)
            
            print(f"  Query {i}/{len(queries)}: {agreement*100:.1f}% agreement, "
                  f"{improvement*100:+.1f}% score change")
        
        return {
            'mean_agreement': np.mean(agreement_scores),
            'mean_score_improvement': np.mean(score_improvements),
            'reranking_changes_results': 1.0 - np.mean(agreement_scores)
        }
    
    def run_full_benchmark(self, queries: List[str] = None, top_k: int = 10) -> Dict:
        """Run complete benchmark suite"""
        queries = queries or TEST_QUERIES[:10]  # Use first 10 for quick test
        
        print(f"\n{'='*80}")
        print(f"SEARCH ENGINE PERFORMANCE BENCHMARK")
        print(f"{'='*80}")
        print(f"Test queries: {len(queries)}")
        print(f"Top-K results: {top_k}")
        print(f"Model: {config.EMBEDDING_MODEL}")
        print(f"Re-ranking: {config.USE_RERANKING}")
        print(f"{'='*80}\n")
        
        results = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'num_queries': len(queries),
                'top_k': top_k,
                'model': config.EMBEDDING_MODEL,
                'reranking_enabled': config.USE_RERANKING,
                'hybrid_embeddings': config.USE_HYBRID_EMBEDDINGS
            }
        }
        
        # Benchmark latency (without re-ranking)
        results['latency_no_rerank'] = self.benchmark_latency(
            queries, top_k, use_reranking=False
        )
        
        # Benchmark latency (with re-ranking)
        if config.USE_RERANKING and self.search_engine.embedding_generator.reranker:
            results['latency_with_rerank'] = self.benchmark_latency(
                queries, top_k, use_reranking=True
            )
        
        # Benchmark relevance scores
        results['relevance'] = self.benchmark_relevance_scores(queries, top_k)
        
        # Benchmark diversity
        results['diversity'] = self.benchmark_result_diversity(queries, top_k)
        
        # Compare with/without re-ranking
        if config.USE_RERANKING and self.search_engine.embedding_generator.reranker:
            results['reranking_comparison'] = self.compare_with_without_reranking(queries, top_k)
        
        return results
    
    def print_summary(self, results: Dict):
        """Print benchmark summary"""
        print(f"\n{'='*80}")
        print(f"BENCHMARK SUMMARY")
        print(f"{'='*80}\n")
        
        # Latency
        print("⚡ LATENCY:")
        if 'latency_no_rerank' in results:
            lat = results['latency_no_rerank']
            print(f"  Without re-ranking: {lat['mean_latency_ms']:.1f}ms "
                  f"(±{lat['std_latency_ms']:.1f}ms)")
        
        if 'latency_with_rerank' in results:
            lat = results['latency_with_rerank']
            print(f"  With re-ranking:    {lat['mean_latency_ms']:.1f}ms "
                  f"(±{lat['std_latency_ms']:.1f}ms)")
            
            if 'latency_no_rerank' in results:
                overhead = (results['latency_with_rerank']['mean_latency_ms'] - 
                           results['latency_no_rerank']['mean_latency_ms'])
                print(f"  Re-ranking overhead: +{overhead:.1f}ms")
        
        # Relevance
        print("\n🎯 RELEVANCE:")
        rel = results['relevance']
        print(f"  Mean top-1 score: {rel['mean_top1_score']:.4f}")
        print(f"  Mean top-5 score: {rel['mean_top5_score']:.4f}")
        print(f"  Scores > 0.7: {rel['scores_above_0.7']*100:.1f}%")
        print(f"  Scores > 0.5: {rel['scores_above_0.5']*100:.1f}%")
        
        # Diversity
        print("\n🎨 DIVERSITY:")
        div = results['diversity']
        print(f"  Avg categories per query: {div['mean_category_diversity']:.1f}")
        print(f"  Avg unique authors: {div['mean_author_diversity']:.1f}")
        
        # Re-ranking comparison
        if 'reranking_comparison' in results:
            print("\n⭐ RE-RANKING IMPACT:")
            rerank = results['reranking_comparison']
            print(f"  Result agreement: {rerank['mean_agreement']*100:.1f}%")
            print(f"  Score improvement: {rerank['mean_score_improvement']*100:+.1f}%")
            print(f"  Changes results: {rerank['reranking_changes_results']*100:.1f}% of queries")
        
        print(f"\n{'='*80}\n")
    
    def save_results(self, results: Dict, filename: str = None):
        """Save benchmark results to JSON"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"benchmark_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"💾 Results saved to: {filename}")


def generate_presentation_data(results: Dict) -> str:
    """Generate formatted data for presentation slides"""
    
    output = []
    output.append("="*60)
    output.append("PRESENTATION DATA")
    output.append("="*60)
    output.append("")
    
    # Slide 1: Performance Metrics
    output.append("SLIDE: Performance Metrics")
    output.append("-"*60)
    
    if 'latency_with_rerank' in results:
        lat = results['latency_with_rerank']
        output.append(f"Search Speed: {lat['mean_latency_ms']:.0f}ms average")
    
    rel = results['relevance']
    output.append(f"Top Result Relevance: {rel['mean_top1_score']:.1%}")
    output.append(f"High Quality Results (>0.7): {rel['scores_above_0.7']:.1%}")
    output.append("")
    
    # Slide 2: Re-ranking Impact
    if 'reranking_comparison' in results:
        output.append("SLIDE: Re-ranking Impact")
        output.append("-"*60)
        rerank = results['reranking_comparison']
        output.append(f"Results Changed: {rerank['reranking_changes_results']:.1%}")
        output.append(f"Score Improvement: {rerank['mean_score_improvement']:+.1%}")
        output.append("")
    
    # Slide 3: Quality Metrics
    output.append("SLIDE: Search Quality")
    output.append("-"*60)
    output.append(f"Average Relevance Score: {rel['mean_all_scores']:.3f}")
    output.append(f"Best Score: {rel['max_score']:.3f}")
    div = results['diversity']
    output.append(f"Avg Categories per Search: {div['mean_category_diversity']:.1f}")
    output.append("")
    
    return "\n".join(output)


def main():
    """Run benchmark"""
    print("🚀 Loading search engine...")
    
    # Load data and initialize search engine
    processor = ArXivDataProcessor()
    df = processor.process_dataset()
    
    search_engine = ArXivSearchEngine(
        df,
        use_reranker=config.USE_RERANKING,
        use_hybrid=config.USE_HYBRID_EMBEDDINGS
    )
    
    print(f"✅ Loaded {len(df)} papers")
    print(f"✅ Model: {config.EMBEDDING_MODEL}")
    print(f"✅ Re-ranking: {config.USE_RERANKING}")
    
    # Create benchmark
    benchmark = SearchBenchmark(search_engine)
    
    # Run benchmark (use fewer queries for quick test)
    print("\n💡 Running quick benchmark with 10 queries...")
    print("   (Use TEST_QUERIES for full 15-query benchmark)")
    
    results = benchmark.run_full_benchmark(
        queries=TEST_QUERIES[:10],  # Quick test with 10 queries
        top_k=10
    )
    
    # Print summary
    benchmark.print_summary(results)
    
    # Save results
    benchmark.save_results(results)
    
    # Generate presentation data
    presentation_data = generate_presentation_data(results)
    print("\n" + presentation_data)
    
    # Save presentation data
    with open('presentation_data.txt', 'w') as f:
        f.write(presentation_data)
    
    print("\n💾 Presentation data saved to: presentation_data.txt")
    print("\n✅ Benchmark complete!")
    print("\n💡 Tip: Run this before and after model changes to compare!")


if __name__ == "__main__":
    main()