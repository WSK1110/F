import json
import polars as pl
from datetime import datetime
from typing import List, Dict, Any
import time
from RAG import RAGChatbot

class RAGEvaluator:
    """
    Evaluation framework for RAG chatbot performance across different model configurations.
    """
    
    def __init__(self):
        self.results = []
        self.test_questions = self._load_test_questions()
        
    def _load_test_questions(self) -> List[Dict[str, Any]]:
        """Load predefined test questions with expected answers"""
        return [
            {
                "id": "Q1",
                "category": "Risk Evaluation",
                "question": "Do these companies worry about the challenges or business risks in China or India in terms of cloud service?",
                "expected_keywords": ["china", "india", "risk", "cloud", "service", "challenges"],
                "expected_companies": ["alphabet", "amazon", "microsoft"],
                "complexity": "high"
            },
            {
                "id": "Q2",
                "category": "Financial Analysis",
                "question": "How much CASH does Amazon have at the end of 2024?",
                "expected_keywords": ["cash", "amazon", "2024", "dollar", "million", "billion"],
                "expected_companies": ["amazon"],
                "complexity": "medium"
            },
            {
                "id": "Q3",
                "category": "Financial Analysis",
                "question": "Compared to 2023, does Amazon's liquidity decrease or increase?",
                "expected_keywords": ["liquidity", "amazon", "2023", "decrease", "increase", "compare"],
                "expected_companies": ["amazon"],
                "complexity": "medium"
            },
            {
                "id": "Q4",
                "category": "Business Analysis",
                "question": "What is the business where main revenue comes from for Amazon / Google / Microsoft?",
                "expected_keywords": ["revenue", "business", "amazon", "google", "microsoft", "main"],
                "expected_companies": ["alphabet", "amazon", "microsoft"],
                "complexity": "medium"
            },
            {
                "id": "Q5",
                "category": "Business Analysis",
                "question": "What main businesses does Amazon do?",
                "expected_keywords": ["business", "amazon", "main", "services", "products"],
                "expected_companies": ["amazon"],
                "complexity": "low"
            },
            {
                "id": "Q6",
                "category": "Competitive Analysis",
                "question": "How do Alphabet, Amazon, and Microsoft compete in the cloud computing market?",
                "expected_keywords": ["cloud", "computing", "compete", "market", "alphabet", "amazon", "microsoft"],
                "expected_companies": ["alphabet", "amazon", "microsoft"],
                "complexity": "high"
            },
            {
                "id": "Q7",
                "category": "Financial Analysis",
                "question": "What is Microsoft's revenue growth rate in 2024 compared to 2023?",
                "expected_keywords": ["revenue", "growth", "microsoft", "2024", "2023", "rate"],
                "expected_companies": ["microsoft"],
                "complexity": "medium"
            },
            {
                "id": "Q8",
                "category": "Risk Analysis",
                "question": "What are the main cybersecurity risks mentioned in these companies' 10-K reports?",
                "expected_keywords": ["cybersecurity", "risk", "security", "threat", "vulnerability"],
                "expected_companies": ["alphabet", "amazon", "microsoft"],
                "complexity": "high"
            }
        ]
    
    def evaluate_response(self, question: Dict, response: str, sources: List[str]) -> Dict[str, Any]:
        """Evaluate a single response"""
        evaluation = {
            "question_id": question["id"],
            "category": question["category"],
            "complexity": question["complexity"],
            "response_length": len(response),
            "has_sources": len(sources) > 0,
            "source_count": len(sources),
            "metrics": {}
        }
        
        # Keyword coverage
        response_lower = response.lower()
        expected_keywords = question["expected_keywords"]
        found_keywords = [kw for kw in expected_keywords if kw in response_lower]
        evaluation["metrics"]["keyword_coverage"] = len(found_keywords) / len(expected_keywords)
        evaluation["metrics"]["keywords_found"] = found_keywords
        evaluation["metrics"]["keywords_missing"] = [kw for kw in expected_keywords if kw not in response_lower]
        
        # Company mention coverage
        expected_companies = question["expected_companies"]
        found_companies = [comp for comp in expected_companies if comp in response_lower]
        evaluation["metrics"]["company_coverage"] = len(found_companies) / len(expected_companies)
        evaluation["metrics"]["companies_found"] = found_companies
        
        # Response quality indicators
        evaluation["metrics"]["has_numbers"] = any(char.isdigit() for char in response)
        evaluation["metrics"]["has_dates"] = any(year in response for year in ["2024", "2023", "2022"])
        evaluation["metrics"]["has_comparisons"] = any(word in response_lower for word in ["compare", "versus", "vs", "than", "increase", "decrease"])
        
        # Hallucination indicators
        evaluation["metrics"]["potential_hallucination"] = self._check_hallucination_indicators(response)
        
        return evaluation
    
    def _check_hallucination_indicators(self, response: str) -> Dict[str, Any]:
        """Check for potential hallucination indicators"""
        indicators = {
            "generic_phrases": 0,
            "unsupported_claims": 0,
            "contradictory_info": 0,
            "overconfident_statements": 0
        }
        
        response_lower = response.lower()
        
        # Generic phrases that might indicate hallucination
        generic_phrases = [
            "it is important to note",
            "generally speaking",
            "typically",
            "usually",
            "in general",
            "broadly speaking"
        ]
        
        indicators["generic_phrases"] = sum(1 for phrase in generic_phrases if phrase in response_lower)
        
        # Overconfident statements
        overconfident_phrases = [
            "definitely",
            "certainly",
            "absolutely",
            "without a doubt",
            "clearly shows"
        ]
        
        indicators["overconfident_statements"] = sum(1 for phrase in overconfident_phrases if phrase in response_lower)
        
        return indicators
    
    def test_model_configuration(self, llm_provider: str, llm_model: str, 
                                embedding_provider: str, embedding_model: str,
                                chunk_size: int = 850, chunk_overlap: int = 300,
                                top_k: int = 8) -> Dict[str, Any]:
        """Test a specific model configuration"""
        
        print(f"\n🧪 Testing Configuration:")
        print(f"LLM: {llm_provider}/{llm_model}")
        print(f"Embeddings: {embedding_provider}/{embedding_model}")
        print(f"RAG Settings: chunk_size={chunk_size}, overlap={chunk_overlap}, top_k={top_k}")
        
        # Initialize chatbot with configuration
        chatbot = RAGChatbot()
        
        # Update configuration
        chatbot.config['LLM']['provider'] = llm_provider
        chatbot.config['LLM']['model'] = llm_model
        chatbot.config['EMBEDDINGS']['provider'] = embedding_provider
        chatbot.config['EMBEDDINGS']['model'] = embedding_model
        chatbot.config['RAG']['chunk_size'] = str(chunk_size)
        chatbot.config['RAG']['chunk_overlap'] = str(chunk_overlap)
        chatbot.config['RAG']['similarity_top_k'] = str(top_k)
        
        # Load documents and create vector store
        documents = chatbot.load_10k_files()
        if not documents:
            return {"error": "No documents loaded"}
        
        vector_store = chatbot.create_vector_store(documents, embedding_provider)
        if not vector_store:
            return {"error": "Failed to create vector store"}
        
        qa_chain = chatbot.create_qa_chain(llm_provider)
        if not qa_chain:
            return {"error": "Failed to create QA chain"}
        
        # Test all questions
        results = []
        total_time = 0
        
        for question in self.test_questions:
            print(f"\n📝 Testing {question['id']}: {question['question'][:50]}...")
            
            start_time = time.time()
            result = chatbot.ask_question(question['question'])
            end_time = time.time()
            
            if result:
                response_time = end_time - start_time
                total_time += response_time
                
                evaluation = self.evaluate_response(question, result['answer'], result['sources'])
                evaluation['response_time'] = response_time
                evaluation['response'] = result['answer']
                evaluation['sources'] = result['sources']
                
                results.append(evaluation)
                
                print(f"✅ Response time: {response_time:.2f}s")
                print(f"📊 Keyword coverage: {evaluation['metrics']['keyword_coverage']:.2%}")
            else:
                print(f"❌ Failed to get response")
                results.append({
                    "question_id": question["id"],
                    "error": "No response received",
                    "response_time": 0
                })
        
        # Calculate aggregate metrics
        config_results = {
            "configuration": {
                "llm_provider": llm_provider,
                "llm_model": llm_model,
                "embedding_provider": embedding_provider,
                "embedding_model": embedding_model,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "top_k": top_k
            },
            "performance": {
                "total_time": total_time,
                "avg_response_time": total_time / len(self.test_questions),
                "successful_responses": len([r for r in results if 'error' not in r]),
                "total_questions": len(self.test_questions)
            },
            "quality_metrics": self._calculate_aggregate_metrics(results),
            "detailed_results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        self.results.append(config_results)
        return config_results
    
    def _calculate_aggregate_metrics(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate aggregate metrics across all questions"""
        successful_results = [r for r in results if 'error' not in r]
        
        if not successful_results:
            return {"error": "No successful results"}
        
        metrics: Dict[str, Any] = {
            "avg_keyword_coverage": sum(r['metrics']['keyword_coverage'] for r in successful_results) / len(successful_results),
            "avg_company_coverage": sum(r['metrics']['company_coverage'] for r in successful_results) / len(successful_results),
            "avg_response_length": sum(r['response_length'] for r in successful_results) / len(successful_results),
            "source_citation_rate": sum(1 for r in successful_results if r['has_sources']) / len(successful_results),
            "number_mention_rate": sum(1 for r in successful_results if r['metrics']['has_numbers']) / len(successful_results),
            "date_mention_rate": sum(1 for r in successful_results if r['metrics']['has_dates']) / len(successful_results),
            "comparison_mention_rate": sum(1 for r in successful_results if r['metrics']['has_comparisons']) / len(successful_results)
        }
        
        # Hallucination indicators
        hallucination_indicators = {
            "avg_generic_phrases": sum(r['metrics']['potential_hallucination']['generic_phrases'] for r in successful_results) / len(successful_results),
            "avg_overconfident_statements": sum(r['metrics']['potential_hallucination']['overconfident_statements'] for r in successful_results) / len(successful_results)
        }
        
        metrics["hallucination_indicators"] = hallucination_indicators
        
        return metrics
    
    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run evaluation across multiple model configurations"""
        
        configurations = [
            # High Performance (Cloud Models)
            {"llm": "gemini", "llm_model": "gemini-pro", "emb": "gemini", "emb_model": "embedding-001"},
            {"llm": "openai", "llm_model": "gpt-4", "emb": "openai", "emb_model": "text-embedding-ada-002"},
            
            # Cost-Effective (Hybrid)
            {"llm": "ollama", "llm_model": "llama3.1", "emb": "gemini", "emb_model": "embedding-001"},
            {"llm": "ollama", "llm_model": "mistral", "emb": "openai", "emb_model": "text-embedding-ada-002"},
            
            # Local Only
            {"llm": "ollama", "llm_model": "deepseek", "emb": "ollama", "emb_model": "nomic-embed-text"},
        ]
        
        print("🚀 Starting Comprehensive RAG Evaluation")
        print("=" * 60)
        
        for i, config in enumerate(configurations, 1):
            print(f"\n📊 Configuration {i}/{len(configurations)}")
            print("-" * 40)
            
            try:
                result = self.test_model_configuration(
                    llm_provider=config["llm"],
                    llm_model=config["llm_model"],
                    embedding_provider=config["emb"],
                    embedding_model=config["emb_model"]
                )
                
                if "error" not in result:
                    print(f"✅ Configuration completed successfully")
                    print(f"⏱️  Average response time: {result['performance']['avg_response_time']:.2f}s")
                    print(f"📈 Keyword coverage: {result['quality_metrics']['avg_keyword_coverage']:.2%}")
                else:
                    print(f"❌ Configuration failed: {result['error']}")
                    
            except Exception as e:
                print(f"❌ Error testing configuration: {e}")
        
        return self.generate_evaluation_report()
    
    def generate_evaluation_report(self) -> Dict[str, Any]:
        """Generate comprehensive evaluation report"""
        
        successful_results = [r for r in self.results if 'error' not in r]
        
        if not successful_results:
            return {"error": "No successful evaluations"}
        
        # Build comparison data for tabular display
        comparison_data = []
        for result in successful_results:
            config = result['configuration']
            perf = result['performance']
            quality = result['quality_metrics']
            
            comparison_data.append({
                'Configuration': f"{config['llm_provider']}/{config['llm_model']} + {config['embedding_provider']}/{config['embedding_model']}",
                'Avg Response Time (s)': perf['avg_response_time'],
                'Success Rate (%)': (perf['successful_responses'] / perf['total_questions']) * 100,
                'Keyword Coverage (%)': quality['avg_keyword_coverage'] * 100,
                'Company Coverage (%)': quality['avg_company_coverage'] * 100,
                'Source Citation Rate (%)': quality['source_citation_rate'] * 100,
                'Avg Response Length': quality['avg_response_length'],
                'Number Mention Rate (%)': quality['number_mention_rate'] * 100,
                'Date Mention Rate (%)': quality['date_mention_rate'] * 100
            })
        
        # Use simple python min/max to avoid heavy DataFrame operations
        best_performance = min(comparison_data, key=lambda x: x['Avg Response Time (s)'])
        best_quality = max(comparison_data, key=lambda x: x['Keyword Coverage (%)'])
        best_overall = max(
            comparison_data,
            key=lambda x: (x['Keyword Coverage (%)'] * x['Success Rate (%)']) / max(x['Avg Response Time (s)'], 1e-6)
        )

        df = pl.DataFrame(comparison_data)
        
        report = {
            "summary": {
                "total_configurations_tested": len(self.results),
                "successful_configurations": len(successful_results),
                "best_performance": best_performance['Configuration'],
                "best_quality": best_quality['Configuration'],
                "best_overall": best_overall['Configuration']
            },
            "comparison_table": df.to_dicts(),
            "detailed_results": self.results,
            "recommendations": self._generate_recommendations(comparison_data),
            "timestamp": datetime.now().isoformat()
        }
        
        return report
    
    def _generate_recommendations(self, table: List[Dict]) -> List[str]:
        """Generate recommendations based on evaluation results"""
        recommendations = []
        
        # Performance recommendations
        fastest = min(table, key=lambda x: x['Avg Response Time (s)'])
        recommendations.append(f"Fastest configuration: {fastest['Configuration']} ({fastest['Avg Response Time (s)']:.2f}s)")
        
        # Quality recommendations
        best_coverage = max(table, key=lambda x: x['Keyword Coverage (%)'])
        recommendations.append(f"Best keyword coverage: {best_coverage['Configuration']} ({best_coverage['Keyword Coverage (%)']:.1f}%)")
        
        # Cost-effectiveness recommendations
        if len(table) > 1:
            most_efficient = max(
                table,
                key=lambda x: (x['Keyword Coverage (%)'] * x['Success Rate (%)']) / max(x['Avg Response Time (s)'], 1e-6)
            )
            recommendations.append(f"Most efficient: {most_efficient['Configuration']}")
        
        # Specific recommendations
        if any('ollama' in row['Configuration'] for row in table):
            recommendations.append("Ollama models provide good local performance but may be slower than cloud models")
        
        if any('gemini' in row['Configuration'] for row in table):
            recommendations.append("Gemini models show strong performance for financial document analysis")
        
        return recommendations
    
    def save_results(self, filename: str = None):
        """Save evaluation results to file"""
        if not filename:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_results_{timestamp}.json"
        
        with open(filename, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        print(f"💾 Results saved to {filename}")
        return filename

def main():
    """Main evaluation script"""
    print("🔍 RAG Chatbot Evaluation Framework")
    print("=" * 50)
    
    evaluator = RAGEvaluator()
    
    # Run comprehensive evaluation
    report = evaluator.run_comprehensive_evaluation()
    
    if "error" not in report:
        print("\n📊 EVALUATION SUMMARY")
        print("=" * 30)
        print(f"Total configurations tested: {report['summary']['total_configurations_tested']}")
        print(f"Successful configurations: {report['summary']['successful_configurations']}")
        print(f"Best performance: {report['summary']['best_performance']}")
        print(f"Best quality: {report['summary']['best_quality']}")
        print(f"Best overall: {report['summary']['best_overall']}")
        
        print("\n💡 RECOMMENDATIONS")
        print("=" * 20)
        for rec in report['recommendations']:
            print(f"• {rec}")
        
        # Save results
        evaluator.save_results()
        
        # Display comparison table
        print("\n📋 COMPARISON TABLE")
        print("=" * 30)
        df = pl.DataFrame(report['comparison_table'])
        print(df)
        
    else:
        print(f"❌ Evaluation failed: {report['error']}")

if __name__ == "__main__":
    main() 