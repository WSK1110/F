#!/usr/bin/env python3
"""
Performance testing script for RAG chatbot optimizations.
Compares original vs optimized implementation performance.
"""

import time
import psutil
import os
import gc
from typing import Dict, Any, List
import pandas as pd
from datetime import datetime
import json

class PerformanceTester:
    """Performance testing framework for RAG implementations"""
    
    def __init__(self):
        self.results = []
        self.process = psutil.Process()
        
    def measure_operation(self, operation_name: str, operation_func, *args, **kwargs) -> Dict[str, Any]:
        """Measure performance of a single operation"""
        # Get initial state
        start_time = time.time()
        start_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        start_cpu = self.process.cpu_percent()
        
        # Force garbage collection for accurate measurement
        gc.collect()
        
        try:
            # Execute operation
            result = operation_func(*args, **kwargs)
            success = True
            error = None
        except Exception as e:
            result = None
            success = False
            error = str(e)
        
        # Get final state
        end_time = time.time()
        end_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        end_cpu = self.process.cpu_percent()
        
        # Calculate metrics
        metrics = {
            'operation': operation_name,
            'duration': end_time - start_time,
            'memory_start': start_memory,
            'memory_end': end_memory,
            'memory_delta': end_memory - start_memory,
            'cpu_start': start_cpu,
            'cpu_end': end_cpu,
            'success': success,
            'error': error,
            'timestamp': datetime.now().isoformat(),
            'result_size': len(str(result)) if result else 0
        }
        
        self.results.append(metrics)
        return metrics
    
    def measure_document_loading(self, data_dir: str = "10k_files") -> Dict[str, Any]:
        """Measure document loading performance"""
        def load_docs():
            from RAG_optimized import load_10k_files_cached
            return load_10k_files_cached(data_dir)
        
        return self.measure_operation("Document Loading", load_docs)
    
    def measure_vector_store_creation(self, documents, embeddings_provider: str = "gemini") -> Dict[str, Any]:
        """Measure vector store creation performance"""
        def create_vector_store():
            from RAG_optimized import RAGChatbotOptimized
            chatbot = RAGChatbotOptimized()
            return chatbot.create_vector_store(documents, embeddings_provider)
        
        return self.measure_operation("Vector Store Creation", create_vector_store)
    
    def measure_query_performance(self, chatbot, question: str) -> Dict[str, Any]:
        """Measure query processing performance"""
        def process_query():
            return chatbot.ask_question(question)
        
        return self.measure_operation(f"Query: {question[:30]}...", process_query)
    
    def run_comprehensive_test(self, test_questions: List[str] = None) -> Dict[str, Any]:
        """Run comprehensive performance test"""
        if test_questions is None:
            test_questions = [
                "How much cash does Amazon have at the end of 2024?",
                "Compare revenue growth between Microsoft and Google",
                "What are the main business risks mentioned by these companies?",
                "What is Apple's main source of revenue?",
                "How do these companies compete in cloud computing?"
            ]
        
        print("🚀 Starting comprehensive performance test...")
        
        # Test 1: Document Loading
        print("📄 Testing document loading...")
        doc_metrics = self.measure_document_loading()
        print(f"   ✅ Completed in {doc_metrics['duration']:.2f}s, Memory: {doc_metrics['memory_delta']:.1f}MB")
        
        # Test 2: System Initialization
        print("⚙️ Testing system initialization...")
        try:
            from RAG_optimized import RAGChatbotOptimized
            
            def init_system():
                chatbot = RAGChatbotOptimized()
                documents = chatbot.load_10k_files()
                vector_store = chatbot.create_vector_store(documents)
                qa_chain = chatbot.create_qa_chain()
                return chatbot
            
            init_metrics = self.measure_operation("System Initialization", init_system)
            print(f"   ✅ Completed in {init_metrics['duration']:.2f}s, Memory: {init_metrics['memory_delta']:.1f}MB")
            
            if init_metrics['success']:
                # Get the initialized chatbot for query tests
                chatbot = RAGChatbotOptimized()
                documents = chatbot.load_10k_files()
                chatbot.create_vector_store(documents)
                chatbot.create_qa_chain()
                
                # Test 3: Query Performance
                print("💬 Testing query performance...")
                query_metrics = []
                
                for i, question in enumerate(test_questions):
                    print(f"   📝 Query {i+1}/{len(test_questions)}: {question[:50]}...")
                    metrics = self.measure_query_performance(chatbot, question)
                    query_metrics.append(metrics)
                    
                    if metrics['success']:
                        print(f"      ✅ {metrics['duration']:.2f}s")
                    else:
                        print(f"      ❌ Failed: {metrics['error']}")
                
                # Calculate query statistics
                successful_queries = [m for m in query_metrics if m['success']]
                if successful_queries:
                    avg_query_time = sum(m['duration'] for m in successful_queries) / len(successful_queries)
                    max_query_time = max(m['duration'] for m in successful_queries)
                    min_query_time = min(m['duration'] for m in successful_queries)
                    
                    print(f"   📊 Query Stats: Avg={avg_query_time:.2f}s, Min={min_query_time:.2f}s, Max={max_query_time:.2f}s")
        
        except Exception as e:
            print(f"   ❌ System initialization failed: {e}")
        
        # Generate summary report
        summary = self.generate_summary()
        return summary
    
    def generate_summary(self) -> Dict[str, Any]:
        """Generate performance summary"""
        if not self.results:
            return {"error": "No performance data collected"}
        
        # Calculate overall statistics
        total_operations = len(self.results)
        successful_operations = len([r for r in self.results if r['success']])
        
        durations = [r['duration'] for r in self.results if r['success']]
        memory_deltas = [r['memory_delta'] for r in self.results if r['success'] and r['memory_delta'] > 0]
        
        summary = {
            'test_timestamp': datetime.now().isoformat(),
            'total_operations': total_operations,
            'successful_operations': successful_operations,
            'success_rate': successful_operations / total_operations * 100 if total_operations > 0 else 0,
            'performance_stats': {
                'avg_duration': sum(durations) / len(durations) if durations else 0,
                'min_duration': min(durations) if durations else 0,
                'max_duration': max(durations) if durations else 0,
                'avg_memory_delta': sum(memory_deltas) / len(memory_deltas) if memory_deltas else 0,
                'total_memory_used': sum(memory_deltas) if memory_deltas else 0
            },
            'operation_breakdown': {}
        }
        
        # Group by operation type
        operation_types = set(r['operation'].split(':')[0] for r in self.results)
        for op_type in operation_types:
            op_results = [r for r in self.results if r['operation'].startswith(op_type)]
            if op_results:
                op_durations = [r['duration'] for r in op_results if r['success']]
                summary['operation_breakdown'][op_type] = {
                    'count': len(op_results),
                    'success_rate': len([r for r in op_results if r['success']]) / len(op_results) * 100,
                    'avg_duration': sum(op_durations) / len(op_durations) if op_durations else 0,
                    'total_duration': sum(op_durations) if op_durations else 0
                }
        
        return summary
    
    def save_results(self, filename: str = None):
        """Save detailed results to file"""
        if filename is None:
            filename = f"performance_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        
        data = {
            'summary': self.generate_summary(),
            'detailed_results': self.results
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"📊 Performance results saved to {filename}")
        return filename
    
    def compare_with_baseline(self, baseline_file: str) -> Dict[str, Any]:
        """Compare current results with baseline performance"""
        try:
            with open(baseline_file, 'r') as f:
                baseline_data = json.load(f)
            
            baseline_summary = baseline_data['summary']
            current_summary = self.generate_summary()
            
            comparison = {
                'baseline_file': baseline_file,
                'comparison_timestamp': datetime.now().isoformat(),
                'improvements': {},
                'regressions': {}
            }
            
            # Compare key metrics
            metrics_to_compare = [
                'avg_duration',
                'max_duration',
                'avg_memory_delta',
                'total_memory_used'
            ]
            
            for metric in metrics_to_compare:
                baseline_val = baseline_summary['performance_stats'].get(metric, 0)
                current_val = current_summary['performance_stats'].get(metric, 0)
                
                if baseline_val > 0:
                    improvement_pct = ((baseline_val - current_val) / baseline_val) * 100
                    
                    if improvement_pct > 0:
                        comparison['improvements'][metric] = {
                            'baseline': baseline_val,
                            'current': current_val,
                            'improvement_percent': improvement_pct
                        }
                    elif improvement_pct < -5:  # Only flag regressions > 5%
                        comparison['regressions'][metric] = {
                            'baseline': baseline_val,
                            'current': current_val,
                            'regression_percent': abs(improvement_pct)
                        }
            
            return comparison
            
        except Exception as e:
            return {"error": f"Could not compare with baseline: {e}"}

def run_memory_stress_test(duration_minutes: int = 5) -> Dict[str, Any]:
    """Run memory stress test to check for memory leaks"""
    print(f"🧠 Running memory stress test for {duration_minutes} minutes...")
    
    tester = PerformanceTester()
    start_time = time.time()
    end_time = start_time + (duration_minutes * 60)
    
    memory_samples = []
    iteration = 0
    
    try:
        from RAG_optimized import RAGChatbotOptimized
        chatbot = RAGChatbotOptimized()
        documents = chatbot.load_10k_files()
        chatbot.create_vector_store(documents)
        chatbot.create_qa_chain()
        
        test_questions = [
            "What is Amazon's revenue?",
            "How much cash does Microsoft have?",
            "Compare Google and Apple",
            "What are the main risks?",
            "Describe the business models"
        ]
        
        while time.time() < end_time:
            iteration += 1
            current_memory = psutil.Process().memory_info().rss / 1024 / 1024
            memory_samples.append({
                'iteration': iteration,
                'timestamp': time.time() - start_time,
                'memory_mb': current_memory
            })
            
            # Process a random question
            question = test_questions[iteration % len(test_questions)]
            result = chatbot.ask_question(question)
            
            # Simulate some processing time
            time.sleep(1)
            
            if iteration % 10 == 0:
                print(f"   Iteration {iteration}, Memory: {current_memory:.1f}MB")
        
        # Analyze memory usage
        initial_memory = memory_samples[0]['memory_mb']
        final_memory = memory_samples[-1]['memory_mb']
        max_memory = max(s['memory_mb'] for s in memory_samples)
        
        memory_growth = final_memory - initial_memory
        memory_growth_rate = memory_growth / duration_minutes  # MB per minute
        
        return {
            'test_duration_minutes': duration_minutes,
            'total_iterations': iteration,
            'initial_memory_mb': initial_memory,
            'final_memory_mb': final_memory,
            'max_memory_mb': max_memory,
            'memory_growth_mb': memory_growth,
            'memory_growth_rate_mb_per_min': memory_growth_rate,
            'memory_samples': memory_samples,
            'potential_memory_leak': memory_growth_rate > 5  # Flag if > 5MB/min growth
        }
        
    except Exception as e:
        return {"error": f"Memory stress test failed: {e}"}

def main():
    """Main performance testing function"""
    print("🎯 RAG Chatbot Performance Testing Suite")
    print("=" * 50)
    
    # Initialize tester
    tester = PerformanceTester()
    
    # Run comprehensive test
    print("\n1️⃣ Running comprehensive performance test...")
    summary = tester.run_comprehensive_test()
    
    # Save results
    results_file = tester.save_results()
    
    # Print summary
    print("\n📊 Performance Test Summary:")
    print(f"   Total Operations: {summary['total_operations']}")
    print(f"   Success Rate: {summary['success_rate']:.1f}%")
    print(f"   Average Duration: {summary['performance_stats']['avg_duration']:.2f}s")
    print(f"   Max Duration: {summary['performance_stats']['max_duration']:.2f}s")
    print(f"   Average Memory Usage: {summary['performance_stats']['avg_memory_delta']:.1f}MB")
    
    # Run memory stress test
    print("\n2️⃣ Running memory stress test...")
    memory_results = run_memory_stress_test(duration_minutes=2)  # Short test for demo
    
    if 'error' not in memory_results:
        print(f"   Memory Growth: {memory_results['memory_growth_mb']:.1f}MB")
        print(f"   Growth Rate: {memory_results['memory_growth_rate_mb_per_min']:.2f}MB/min")
        
        if memory_results['potential_memory_leak']:
            print("   ⚠️  Potential memory leak detected!")
        else:
            print("   ✅ Memory usage appears stable")
    else:
        print(f"   ❌ Memory test failed: {memory_results['error']}")
    
    print(f"\n🎉 Performance testing completed! Results saved to {results_file}")
    return summary, memory_results

if __name__ == "__main__":
    # Check if optimized RAG module is available
    try:
        import RAG_optimized
        main()
    except ImportError:
        print("❌ RAG_optimized module not found. Please ensure it's in the same directory.")
        print("💡 Run this script from the same directory as RAG_optimized.py")