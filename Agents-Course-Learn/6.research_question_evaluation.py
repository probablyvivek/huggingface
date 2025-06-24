"""
Step 6: Testing on Complex Research Questions
===========================================

Real-World Evaluation:
1. Complex Multi-Step Questions - Like the original dataset
2. Performance Measurement - Success rate tracking
3. Reasoning Analysis - How well does it think?
4. Failure Analysis - What needs improvement?
5. Benchmarking - Compare to human performance
"""

import os
import time
import json
from datetime import datetime
from dotenv import load_dotenv
from typing import List, Dict, Any, Optional

# Import our vector agent from the previous step
from typing import TypedDict, Annotated
import operator

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.tools import tool
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

load_dotenv()

# ==========================================
# COMPLEX RESEARCH QUESTIONS DATASET
# ==========================================
# These are inspired by the original evaluation questions

COMPLEX_RESEARCH_QUESTIONS = [
    {
        "id": "research_001",
        "question": "What is the last word before the second chorus of the King of Pop's fifth single from his sixth studio album?",
        "expected_answer": "stare",
        "difficulty": "hard",
        "requires": ["web_search", "wikipedia_search", "multi_step_reasoning"],
        "steps": [
            "1. Identify 'King of Pop' (Michael Jackson)",
            "2. Find his sixth studio album (Thriller)",
            "3. Identify the fifth single from Thriller (Human Nature)",
            "4. Find lyrics of Human Nature",
            "5. Identify chorus pattern",
            "6. Find second chorus occurrence",
            "7. Find last word before it"
        ]
    },
    {
        "id": "research_002", 
        "question": "If I invest $5000 at 3.5% annual compound interest for 8 years, how much will I have, and what country has a similar GDP per capita?",
        "expected_answer": "~$6570, similar to countries like Bulgaria or Romania",
        "difficulty": "medium",
        "requires": ["advanced_calculator", "web_search"],
        "steps": [
            "1. Calculate compound interest: 5000 * (1.035)^8",
            "2. Search for countries with GDP per capita around that amount"
        ]
    },
    {
        "id": "research_003",
        "question": "What is the population density of the country where the 2024 Olympics were held, and how does it compare to Singapore?",
        "expected_answer": "France: ~118 people/km², Singapore: ~8000 people/km², Singapore is ~68x denser",
        "difficulty": "medium", 
        "requires": ["web_search", "wikipedia_search", "advanced_calculator"],
        "steps": [
            "1. Identify 2024 Olympics location (Paris, France)",
            "2. Find France's population density",
            "3. Find Singapore's population density", 
            "4. Calculate comparison ratio"
        ]
    },
    {
        "id": "research_004",
        "question": "Who won the Nobel Prize in Physics in 2023, and what university were they affiliated with when they won?",
        "expected_answer": "Pierre Agostini, Ferenc Krausz, and Anne L'Huillier for attosecond physics; various universities",
        "difficulty": "easy",
        "requires": ["web_search", "wikipedia_search"],
        "steps": [
            "1. Search for 2023 Nobel Prize Physics winners",
            "2. Find their university affiliations"
        ]
    },
    {
        "id": "research_005",
        "question": "What is the square root of the number of days in a leap year, rounded to 2 decimal places?",
        "expected_answer": "19.13",
        "difficulty": "easy",
        "requires": ["advanced_calculator"],
        "steps": [
            "1. Days in leap year = 366",
            "2. Calculate sqrt(366) ≈ 19.1311",
            "3. Round to 2 decimal places = 19.13"
        ]
    },
    {
        "id": "research_006",
        "question": "What programming language was first released in the same year as the fall of the Berlin Wall, and what are its main characteristics?",
        "expected_answer": "Python (1989/1991 depending on definition), characteristics: interpreted, object-oriented, readable syntax",
        "difficulty": "medium",
        "requires": ["wikipedia_search", "web_search"],
        "steps": [
            "1. Find year Berlin Wall fell (1989)",
            "2. Find programming languages released in 1989-1991",
            "3. Describe characteristics of the language"
        ]
    }
]

# ==========================================
# EVALUATION FRAMEWORK
# ==========================================

class ResearchEvaluator:
    """Evaluates agent performance on complex research questions"""
    
    def __init__(self, agent, vector_db):
        self.agent = agent
        self.vector_db = vector_db
        self.results = []
    
    def evaluate_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a single research question"""
        print(f"\n🔬 EVALUATING: {question_data['id']}")
        print("=" * 80)
        print(f"📝 Question: {question_data['question']}")
        print(f"🎯 Difficulty: {question_data['difficulty']}")
        print(f"🛠️  Required Tools: {', '.join(question_data['requires'])}")
        print(f"📋 Expected Steps: {len(question_data['steps'])}")
        
        # Start timing
        start_time = time.time()
        
        # Run the agent
        try:
            result = self._run_agent(question_data['question'])
            execution_time = time.time() - start_time
            
            # Analyze the result
            evaluation = self._analyze_result(question_data, result, execution_time)
            
            # Display evaluation
            self._display_evaluation(evaluation)
            
            return evaluation
            
        except Exception as e:
            print(f"❌ EXECUTION ERROR: {e}")
            return {
                "question_id": question_data['id'],
                "success": False,
                "error": str(e),
                "execution_time": time.time() - start_time
            }
    
    def _run_agent(self, question: str) -> Dict[str, Any]:
        """Run the agent on a question"""
        initial_state = {
            "messages": [HumanMessage(content=question)],
            "research_notes": "",
            "question_type": "",
            "sources_used": [],
            "confidence": 0.0,
            "similar_questions": [],
            "learning_examples": ""
        }
        
        final_state = self.agent.invoke(initial_state)
        
        # Extract final answer
        final_answer = ""
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and not hasattr(msg, 'tool_calls'):
                final_answer = msg.content
                break
        
        return {
            "final_state": final_state,
            "final_answer": final_answer,
            "tools_used": final_state.get("sources_used", []),
            "confidence": final_state.get("confidence", 0.0),
            "similar_questions": len(final_state.get("similar_questions", []))
        }
    
    def _analyze_result(self, question_data: Dict[str, Any], result: Dict[str, Any], execution_time: float) -> Dict[str, Any]:
        """Analyze agent performance"""
        
        # Basic metrics
        evaluation = {
            "question_id": question_data['id'],
            "question": question_data['question'],
            "difficulty": question_data['difficulty'],
            "expected_answer": question_data['expected_answer'],
            "agent_answer": result['final_answer'],
            "execution_time": round(execution_time, 2),
            "tools_used": result['tools_used'],
            "confidence": result['confidence'],
            "similar_questions_found": result['similar_questions']
        }
        
        # Success evaluation (simplified - in real system this would be more sophisticated)
        success_score = self._evaluate_answer_quality(
            question_data['expected_answer'], 
            result['final_answer']
        )
        
        evaluation.update({
            "success_score": success_score,
            "success": success_score > 0.6,  # 60% threshold
            "tool_usage_appropriate": self._evaluate_tool_usage(question_data, result),
            "reasoning_quality": self._evaluate_reasoning(result['final_answer'])
        })
        
        return evaluation
    
    def _evaluate_answer_quality(self, expected: str, actual: str) -> float:
        """Simple answer quality evaluation"""
        if not actual:
            return 0.0
        
        # Convert to lowercase for comparison
        expected_lower = expected.lower()
        actual_lower = actual.lower()
        
        # Check for key terms/numbers
        expected_words = set(expected_lower.split())
        actual_words = set(actual_lower.split())
        
        # Simple overlap score
        overlap = len(expected_words.intersection(actual_words))
        total_expected = len(expected_words)
        
        if total_expected == 0:
            return 0.5  # Default if no expected words
        
        base_score = overlap / total_expected
        
        # Bonus for including numbers/specific facts
        if any(char.isdigit() for char in expected) and any(char.isdigit() for char in actual):
            base_score += 0.2
        
        return min(1.0, base_score)
    
    def _evaluate_tool_usage(self, question_data: Dict[str, Any], result: Dict[str, Any]) -> bool:
        """Check if appropriate tools were used"""
        required_tools = set(question_data['requires'])
        used_tools = set(result['tools_used'])
        
        # Check for tool type overlap (simplified)
        tool_mapping = {
            'web_search': 'web_search',
            'wikipedia_search': 'wikipedia_search', 
            'advanced_calculator': 'calculator',
            'multi_step_reasoning': 'synthesis'
        }
        
        used_mapped = set()
        for tool in used_tools:
            for req_tool, mapped_tool in tool_mapping.items():
                if mapped_tool in tool:
                    used_mapped.add(req_tool)
        
        return len(required_tools.intersection(used_mapped)) > 0
    
    def _evaluate_reasoning(self, answer: str) -> float:
        """Evaluate reasoning quality"""
        if not answer:
            return 0.0
        
        # Simple heuristics for reasoning quality
        reasoning_indicators = [
            'because', 'therefore', 'first', 'then', 'next', 'finally',
            'step', 'calculate', 'search', 'find', 'according to',
            'based on', 'source', 'reference'
        ]
        
        answer_lower = answer.lower()
        reasoning_count = sum(1 for indicator in reasoning_indicators if indicator in answer_lower)
        
        # Score based on reasoning indicators and length
        length_score = min(1.0, len(answer) / 200)  # Longer answers tend to have more reasoning
        indicator_score = min(1.0, reasoning_count / 3)  # At least 3 reasoning indicators is good
        
        return (length_score + indicator_score) / 2
    
    def _display_evaluation(self, evaluation: Dict[str, Any]):
        """Display evaluation results"""
        print(f"\n📊 EVALUATION RESULTS")
        print("-" * 50)
        print(f"✅ Success: {'YES' if evaluation['success'] else 'NO'} (Score: {evaluation['success_score']:.2f})")
        print(f"⏱️  Execution Time: {evaluation['execution_time']}s")
        print(f"🔧 Tools Used: {', '.join(evaluation['tools_used'])}")
        print(f"🎯 Confidence: {evaluation['confidence']:.2f}")
        print(f"🧠 Similar Questions: {evaluation['similar_questions_found']}")
        print(f"🛠️  Tool Usage Appropriate: {'YES' if evaluation['tool_usage_appropriate'] else 'NO'}")
        print(f"💭 Reasoning Quality: {evaluation['reasoning_quality']:.2f}")
        
        print(f"\n📝 Expected Answer:")
        print(f"   {evaluation['expected_answer']}")
        print(f"\n🤖 Agent Answer:")
        print(f"   {evaluation['agent_answer'][:200]}{'...' if len(evaluation['agent_answer']) > 200 else ''}")
    
    def run_full_evaluation(self, questions: 'Optional[List[Dict[str, Any]]]' = None) -> Dict[str, Any]:
        """Run evaluation on all questions"""
        if questions is None:
            questions = COMPLEX_RESEARCH_QUESTIONS
        print("🚀 STARTING COMPREHENSIVE EVALUATION")
        print("=" * 80)
        print(f"📊 Evaluating {len(questions)} research questions")
        print(f"🎯 Difficulty levels: {set(q['difficulty'] for q in questions)}")
        
        results = []
        for question_data in questions:
            result = self.evaluate_question(question_data)
            results.append(result)
            self.results.append(result)
        
        # Generate summary
        summary = self._generate_summary(results)
        self._display_summary(summary)
        
        return {
            "summary": summary,
            "detailed_results": results
        }
    
    def _generate_summary(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate evaluation summary"""
        total_questions = len(results)
        successful = sum(1 for r in results if r.get('success', False))
        
        avg_score = sum(r.get('success_score', 0) for r in results) / total_questions
        avg_time = sum(r.get('execution_time', 0) for r in results) / total_questions
        avg_confidence = sum(r.get('confidence', 0) for r in results) / total_questions
        avg_reasoning = sum(r.get('reasoning_quality', 0) for r in results) / total_questions
        
        # Performance by difficulty
        by_difficulty = {}
        for result in results:
            diff = result.get('difficulty', 'unknown')
            if diff not in by_difficulty:
                by_difficulty[diff] = {'total': 0, 'successful': 0}
            by_difficulty[diff]['total'] += 1
            if result.get('success', False):
                by_difficulty[diff]['successful'] += 1
        
        return {
            "total_questions": total_questions,
            "successful": successful,
            "success_rate": successful / total_questions,
            "average_score": avg_score,
            "average_time": avg_time,
            "average_confidence": avg_confidence,
            "average_reasoning_quality": avg_reasoning,
            "performance_by_difficulty": by_difficulty
        }
    
    def _display_summary(self, summary: Dict[str, Any]):
        """Display evaluation summary"""
        print("\n" + "="*80)
        print("📊 EVALUATION SUMMARY")
        print("="*80)
        
        print(f"🎯 Overall Performance:")
        print(f"   Success Rate: {summary['success_rate']:.1%} ({summary['successful']}/{summary['total_questions']})")
        print(f"   Average Score: {summary['average_score']:.2f}/1.0")
        print(f"   Average Time: {summary['average_time']:.1f}s per question")
        print(f"   Average Confidence: {summary['average_confidence']:.2f}")
        print(f"   Reasoning Quality: {summary['average_reasoning_quality']:.2f}")
        
        print(f"\n📈 Performance by Difficulty:")
        for difficulty, stats in summary['performance_by_difficulty'].items():
            success_rate = stats['successful'] / stats['total'] if stats['total'] > 0 else 0
            print(f"   {difficulty.title()}: {success_rate:.1%} ({stats['successful']}/{stats['total']})")
        
        print("\n🎉 Evaluation Complete!")

# ==========================================
# IMPORT VECTOR AGENT (Simplified)
# ==========================================

# We'll need to import or recreate the vector agent here
def create_evaluation_agent():
    """Create agent for evaluation (simplified version)"""
    # This would normally import from your previous step
    # For now, creating a basic version
    
    @tool
    def web_search(query: str) -> str:
        """Search the web for information on a given query."""
        try:
            from tavily import TavilyClient
            tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
            result = tavily.search(query, max_results=3)
            if result and 'results' in result:
                return "\n".join([f"{r.get('title', '')}: {r.get('content', '')[:200]}..." for r in result['results'][:2]])
            return "No results found"
        except:
            return "Web search unavailable"
    
    @tool
    def wikipedia_search(topic: str) -> str:
        """Search Wikipedia for information on a specific topic."""
        try:
            import wikipedia
            return wikipedia.summary(topic, sentences=2)
        except:
            return "Wikipedia search unavailable"
    
    @tool
    def advanced_calculator(expression: str) -> str:
        """Perform mathematical calculations using Python's math module."""
        try:
            import math
            allowed = {k: v for k, v in math.__dict__.items() if not k.startswith("__")}
            result = eval(expression, {"__builtins__": {}}, allowed)
            return f"Result: {result}"
        except Exception as e:
            return f"Calculation error: {e}"
    
    # Create a simple agent (you'd use your full vector agent here)
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    tools = [web_search, wikipedia_search, advanced_calculator]
    llm_with_tools = llm.bind_tools(tools)
    
    class SimpleState(TypedDict):
        messages: Annotated[List[BaseMessage], operator.add]
        sources_used: List[str]
        confidence: float
        similar_questions: List[Any]
    
    def simple_agent_call(state):
        response = llm_with_tools.invoke(state["messages"])
        return {"messages": [response], "sources_used": ["simple_agent"], "confidence": 0.7, "similar_questions": []}
    
    # Simple workflow
    workflow = StateGraph(SimpleState)
    workflow.add_node("agent", simple_agent_call)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_edge(START, "agent")
    workflow.add_conditional_edges("agent", tools_condition, {"tools": "tools", END: END})
    workflow.add_edge("tools", END)
    
    return workflow.compile(), None

if __name__ == "__main__":
    print("🔬 RESEARCH QUESTION EVALUATION SYSTEM")
    print("=" * 50)
    
    # Check requirements
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Missing GROQ_API_KEY")
        exit(1)
    
    # Create agent (you'd use your vector agent here)
    print("🏗️  Creating evaluation agent...")
    agent, vector_db = create_evaluation_agent()
    
    # Create evaluator
    evaluator = ResearchEvaluator(agent, vector_db)
    
    # Run evaluation
    print("\n🚀 Starting Research Question Evaluation...")
    
    # Test on a subset first
    test_questions = COMPLEX_RESEARCH_QUESTIONS[:3]  # First 3 questions
    
    evaluation_results = evaluator.run_full_evaluation(test_questions)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"evaluation_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(evaluation_results, f, indent=2, default=str)
    
    print(f"\n💾 Results saved to: {results_file}")
    print("🎯 Your agent has been evaluated on complex research questions!")