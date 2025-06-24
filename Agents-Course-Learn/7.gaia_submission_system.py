"""
Step 7: GAIA Submission System - Real World Evaluation
====================================================

Final Integration:
1. GAIA API Integration - Connect to real evaluation system
2. Production Agent Deployment - Your agent vs. GAIA benchmark
3. Leaderboard Submission - Official scoring and ranking
4. Performance Analysis - Compare to state-of-the-art
5. Certificate Achievement - 30%+ success rate goal
"""

import os
import json
import requests
import gradio as gr
from datetime import datetime
from typing import Dict, List, Any, Optional
from dotenv import load_dotenv

# Import our complete agent system
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
# GAIA API CONFIGURATION
# ==========================================

# GAIA API endpoints (replace with actual URLs from the course)
GAIA_API_BASE = "https://agents-course-unit4-scoring.hf.space"
QUESTIONS_ENDPOINT = f"{GAIA_API_BASE}/questions"
SUBMIT_ENDPOINT = f"{GAIA_API_BASE}/submit"
RANDOM_QUESTION_ENDPOINT = f"{GAIA_API_BASE}/random-question"

# ==========================================
# PRODUCTION RESEARCH AGENT
# ==========================================
# This is your complete agent optimized for GAIA

class GAIAAgentState(TypedDict):
    """State for GAIA evaluation agent"""
    messages: Annotated[List[BaseMessage], operator.add]
    research_notes: str
    question_type: str
    sources_used: List[str]
    confidence: float
    task_id: str
    final_answer: str

# Enhanced tools for GAIA evaluation
@tool
def advanced_web_search(query: str) -> str:
    """Enhanced web search for GAIA questions"""
    try:
        from tavily import TavilyClient
        tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        
        # Enhanced search with more results for complex questions
        result = tavily.search(query, max_results=5)
        
        if result and 'results' in result:
            formatted_results = []
            for i, r in enumerate(result['results'][:5], 1):
                title = r.get('title', 'No title')
                content = r.get('content', 'No content')
                url = r.get('url', 'No URL')
                
                formatted_results.append(f"""
=== Source {i} ===
Title: {title}
Content: {content[:300]}...
URL: {url}
""")
            
            return f"Web search results for '{query}':\n" + "\n".join(formatted_results)
        return f"No web results found for '{query}'"
        
    except Exception as e:
        return f"Web search error: {e}"

@tool
def enhanced_wikipedia_search(topic: str) -> str:
    """Enhanced Wikipedia search with better error handling"""
    try:
        import wikipedia
        wikipedia.set_lang("en")
        
        # Get summary and additional info
        try:
            page = wikipedia.page(topic)
            summary = wikipedia.summary(topic, sentences=4)
            
            return f"""
Wikipedia: {topic}
Summary: {summary}
URL: {page.url}
Categories: {', '.join(page.categories[:5])}
"""
        except wikipedia.exceptions.DisambiguationError as e:
            # Try first option if disambiguation
            first_option = e.options[0]
            page = wikipedia.page(first_option)
            summary = wikipedia.summary(first_option, sentences=3)
            
            return f"""
Wikipedia: {topic} (redirected to {first_option})
Summary: {summary}
URL: {page.url}
Note: Multiple pages found, showing best match
"""
            
    except Exception as e:
        return f"Wikipedia search error for '{topic}': {e}"

@tool
def professional_calculator(expression: str) -> str:
    """Professional calculator with extensive math functions"""
    try:
        import math
        import datetime
        
        # Enhanced math functions
        allowed_names = {
            # Basic math
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'pow': pow,
            
            # Math module
            **{k: v for k, v in math.__dict__.items() if not k.startswith("__")},
            
            # Date calculations
            'datetime': datetime,
            
            # Common constants
            'pi': math.pi, 'e': math.e
        }
        
        # Safe evaluation
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        
        return f"Calculation: {expression} = {result}"
        
    except Exception as e:
        return f"Calculation error for '{expression}': {e}"

# All production tools
gaia_tools = [advanced_web_search, enhanced_wikipedia_search, professional_calculator]

# ==========================================
# GAIA-OPTIMIZED AGENT NODES
# ==========================================

def gaia_analyze_question(state: GAIAAgentState) -> GAIAAgentState:
    """Analyze GAIA question with enhanced strategy"""
    print(f"🔍 Analyzing GAIA question: {state.get('task_id', 'unknown')}")
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    question = state["messages"][-1].content
    
    analysis_prompt = f"""
    You are analyzing a GAIA evaluation question. These are complex, multi-step research questions.
    
    Question: {question}
    
    Analyze this question and determine:
    1. Question complexity: simple, medium, complex
    2. Required information types: current_events, historical_facts, calculations, technical_info, cultural_knowledge
    3. Research strategy: what tools and approach would work best
    4. Expected steps: roughly how many research steps needed
    5. Confidence prediction: how confident can we be in finding the answer (0.0-1.0)
    
    GAIA questions often require:
    - Multi-step reasoning
    - Combining information from multiple sources
    - Precise factual accuracy
    - Sometimes mathematical calculations
    
    Respond in this format:
    Complexity: [simple/medium/complex]
    Types: [list of information types needed]
    Strategy: [detailed research approach]
    Steps: [estimated number of steps]
    Confidence: [0.0-1.0]
    """
    
    response = llm.invoke([HumanMessage(content=analysis_prompt)])
    analysis = response.content
    
    # Parse analysis
    complexity = "medium"  # default
    confidence = 0.6  # default
    
    if "complex" in str(analysis).lower():
        complexity = "complex"
        confidence = 0.5
    elif "simple" in str(analysis).lower():
        complexity = "simple"
        confidence = 0.8
    
    # Extract confidence if specified
    try:
        analysis_text = str(analysis)
        if "confidence:" in analysis_text.lower():
            conf_line = [line for line in analysis_text.split('\n') if 'confidence:' in line.lower()][0]
            confidence = float(conf_line.split(':')[1].strip().split()[0])
    except:
        pass
    
    print(f"📊 Analysis: {complexity} complexity, confidence: {confidence:.2f}")
    
    return {
        "messages": state["messages"],
        "research_notes": f"GAIA question analyzed: {complexity} complexity\nAnalysis: {analysis}",
        "question_type": complexity,
        "sources_used": [],
        "confidence": confidence,
        "task_id": state.get("task_id", ""),
        "final_answer": ""
    }

def gaia_research_orchestrator(state: GAIAAgentState) -> GAIAAgentState:
    """Orchestrate research for GAIA questions"""
    print("🔬 Conducting GAIA research...")
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    llm_with_tools = llm.bind_tools(gaia_tools)
    
    question = state["messages"][-1].content
    complexity = state.get("question_type", "medium")
    research_notes = state.get("research_notes", "")
    
    system_prompt = f"""
    You are a research agent for GAIA evaluation questions. These require precise, accurate answers.
    
    Question Complexity: {complexity}
    Research Notes: {research_notes}
    
    For GAIA questions:
    1. Break down complex questions into sub-questions
    2. Use appropriate tools systematically
    3. Gather precise, factual information
    4. Pay attention to specific details (dates, numbers, names)
    5. Cross-reference information when possible
    
    Available tools:
    - advanced_web_search: For current information, news, recent events
    - enhanced_wikipedia_search: For factual, historical, biographical information
    - professional_calculator: For mathematical calculations and date arithmetic
    
    Choose the most appropriate tool for this question. If you need multiple pieces of information,
    start with the most fundamental piece first.
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Research this GAIA question: {question}")
    ]
    
    response = llm_with_tools.invoke(messages)
    updated_messages = state["messages"] + [response]
    
    return {
        "messages": updated_messages,
        "research_notes": state["research_notes"],
        "question_type": state["question_type"],
        "sources_used": state["sources_used"],
        "confidence": state["confidence"],
        "task_id": state["task_id"],
        "final_answer": state["final_answer"]
    }

def gaia_answer_synthesizer(state: GAIAAgentState) -> GAIAAgentState:
    """Synthesize final answer for GAIA submission"""
    print("📝 Synthesizing GAIA answer...")
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    # Extract information from conversation
    original_question = state["messages"][0].content
    research_notes = state.get("research_notes", "")
    
    # Get tool results
    tool_results = []
    for msg in state["messages"]:
        if hasattr(msg, 'content') and any(keyword in str(msg.content) for keyword in ['Source', 'Wikipedia', 'Calculation', '===']):
            tool_results.append(msg.content)
    
    synthesis_prompt = f"""
    Synthesize a final answer for this GAIA evaluation question.
    
    CRITICAL: GAIA requires EXACT answers. Be precise with:
    - Numbers (exact values, proper rounding)
    - Names (correct spelling, full names when needed)
    - Dates (correct format)
    - Facts (verified accuracy)
    
    Original Question: {original_question}
    Research Notes: {research_notes}
    
    Tool Results:
    {chr(10).join(tool_results) if tool_results else 'No tool results available'}
    
    Based on the research, provide a direct, precise answer. 
    
    IMPORTANT: 
    - Give ONLY the final answer, no explanations
    - Be as specific and accurate as possible
    - Use exact numbers, dates, names
    - If you're unsure, indicate your best estimate
    - Do not include "FINAL ANSWER:" or similar prefixes
    
    Your response should be the direct answer only.
    """
    
    response = llm.invoke([HumanMessage(content=synthesis_prompt)])
    # The `content` of a message can be a list for complex outputs, but here we expect a string.
    # We assert the type to satisfy the linter and ensure runtime correctness.
    assert isinstance(response.content, str)
    final_answer = response.content.strip()
    
    print(f"✅ GAIA Answer: {final_answer}")
    
    # Update messages and set final answer
    final_messages = state["messages"] + [response]
    
    return {
        "messages": final_messages,
        "research_notes": state["research_notes"] + f"\nFinal answer synthesized: {final_answer}",
        "question_type": state["question_type"],
        "sources_used": state["sources_used"] + ["synthesis"],
        "confidence": state["confidence"],
        "task_id": state["task_id"],
        "final_answer": final_answer
    }

# ==========================================
# GAIA AGENT WORKFLOW
# ==========================================

def create_gaia_agent():
    """Create optimized agent for GAIA evaluation"""
    print("🏗️  Building GAIA Evaluation Agent...")
    
    # Create state graph
    workflow = StateGraph(GAIAAgentState)
    
    # Add nodes
    workflow.add_node("analyze", gaia_analyze_question)
    workflow.add_node("research", gaia_research_orchestrator)
    workflow.add_node("tools", ToolNode(gaia_tools))
    workflow.add_node("synthesize", gaia_answer_synthesizer)
    
    # Define workflow
    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "research")
    
    # Conditional tool usage
    workflow.add_conditional_edges(
        "research",
        tools_condition,
        {
            "tools": "tools",
            END: "synthesize"
        }
    )
    
    workflow.add_edge("tools", "synthesize")
    workflow.add_edge("synthesize", END)
    
    # Compile
    agent = workflow.compile()
    
    print("✅ GAIA Agent Ready!")
    return agent

# ==========================================
# GAIA API INTERFACE
# ==========================================

class GAIAInterface:
    """Interface for GAIA evaluation system"""
    
    def __init__(self, agent):
        self.agent = agent
        self.session = requests.Session()
    
    def get_questions(self) -> List[Dict[str, Any]]:
        """Fetch all GAIA questions"""
        try:
            response = self.session.get(QUESTIONS_ENDPOINT, timeout=30)
            response.raise_for_status()
            questions = response.json()
            print(f"📥 Fetched {len(questions)} GAIA questions")
            return questions
        except Exception as e:
            print(f"❌ Error fetching questions: {e}")
            return []
    
    def get_random_question(self) -> Optional[Dict[str, Any]]:
        """Fetch a random GAIA question"""
        try:
            response = self.session.get(RANDOM_QUESTION_ENDPOINT, timeout=15)
            response.raise_for_status()
            question = response.json()
            print(f"📥 Fetched random question: {question.get('task_id', 'unknown')}")
            return question
        except Exception as e:
            print(f"❌ Error fetching random question: {e}")
            return None
    
    def process_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single GAIA question"""
        task_id = question_data.get('task_id', 'unknown')
        question_text = question_data.get('question', '')
        
        print(f"🤖 Processing GAIA question: {task_id}")
        print(f"📝 Question: {question_text[:100]}...")
        
        # Initialize state
        initial_state = {
            "messages": [HumanMessage(content=question_text)],
            "research_notes": "",
            "question_type": "",
            "sources_used": [],
            "confidence": 0.0,
            "task_id": task_id,
            "final_answer": ""
        }
        
        try:
            # Run agent
            final_state = self.agent.invoke(initial_state)
            
            # Extract final answer
            final_answer = final_state.get("final_answer", "").strip()
            if not final_answer:
                # Fallback to last AI message
                for msg in reversed(final_state["messages"]):
                    if isinstance(msg, AIMessage) and not hasattr(msg, 'tool_calls'):
                        if isinstance(msg.content, str):
                            final_answer = msg.content.strip()
                            break
            
            result = {
                "task_id": task_id,
                "submitted_answer": final_answer,
                "confidence": final_state.get("confidence", 0.0),
                "sources_used": final_state.get("sources_used", []),
                "success": True
            }
            
            print(f"✅ Answer: {final_answer}")
            return result
            
        except Exception as e:
            print(f"❌ Error processing question {task_id}: {e}")
            return {
                "task_id": task_id,
                "submitted_answer": f"Error: {str(e)}",
                "confidence": 0.0,
                "sources_used": [],
                "success": False
            }
    
    def submit_answers(self, username: str, agent_code_url: str, answers: List[Dict[str, str]]) -> Dict[str, Any]:
        """Submit answers to GAIA leaderboard"""
        submission_data = {
            "username": username,
            "agent_code": agent_code_url,
            "answers": answers
        }
        
        try:
            print(f"📤 Submitting {len(answers)} answers for {username}")
            response = self.session.post(SUBMIT_ENDPOINT, json=submission_data, timeout=60)
            response.raise_for_status()
            
            result = response.json()
            print(f"🎉 Submission successful!")
            print(f"📊 Score: {result.get('score', 'unknown')}%")
            print(f"✅ Correct: {result.get('correct_count', '?')}/{result.get('total_attempted', '?')}")
            
            return result
            
        except Exception as e:
            print(f"❌ Submission error: {e}")
            return {"error": str(e)}

# ==========================================
# GRADIO INTERFACE FOR GAIA SUBMISSION
# ==========================================

def create_gaia_gradio_interface():
    """Create Gradio interface for GAIA submission"""
    
    # Initialize agent
    gaia_agent = create_gaia_agent()
    gaia_interface = GAIAInterface(gaia_agent)
    
    def test_single_question():
        """Test agent on a single random question"""
        question = gaia_interface.get_random_question()
        if not question:
            return "❌ Failed to fetch question", ""
        
        result = gaia_interface.process_question(question)
        
        question_display = f"""
**Question ID:** {question.get('task_id', 'unknown')}

**Question:** {question.get('question', 'No question text')}

**Agent Answer:** {result.get('submitted_answer', 'No answer')}

**Confidence:** {result.get('confidence', 0.0):.2f}

**Sources Used:** {', '.join(result.get('sources_used', []))}

**Success:** {'✅ Yes' if result.get('success', False) else '❌ No'}
        """
        
        return question_display, json.dumps(result, indent=2)
    
    def run_full_evaluation(profile: gr.OAuthProfile | None):
        """Run full GAIA evaluation and submit to leaderboard"""
        if not profile:
            return "❌ Please log in with Hugging Face to submit", None
        
        username = profile.username
        
        # Get your space URL (you'll need to update this)
        space_id = os.getenv("SPACE_ID", "your-username/your-space-name")
        agent_code_url = f"https://huggingface.co/spaces/{space_id}/tree/main"
        
        print(f"🚀 Running full GAIA evaluation for {username}")
        
        # Fetch all questions
        questions = gaia_interface.get_questions()
        if not questions:
            return "❌ Failed to fetch GAIA questions", None
        
        # Process all questions
        results = []
        for i, question in enumerate(questions, 1):
            print(f"📝 Processing question {i}/{len(questions)}")
            result = gaia_interface.process_question(question)
            results.append({
                "task_id": result["task_id"],
                "submitted_answer": result["submitted_answer"]
            })
        
        # Submit to leaderboard
        submission_result = gaia_interface.submit_answers(username, agent_code_url, results)
        
        if "error" in submission_result:
            return f"❌ Submission failed: {submission_result['error']}", None
        
        # Format results
        status_message = f"""
🎉 GAIA Evaluation Complete!

**Username:** {username}
**Questions Processed:** {len(results)}
**Overall Score:** {submission_result.get('score', 'unknown')}%
**Correct Answers:** {submission_result.get('correct_count', '?')}/{submission_result.get('total_attempted', '?')}
**Leaderboard Position:** Check the official leaderboard!

**Agent Code:** {agent_code_url}

{'🏆 Congratulations! You achieved the 30%+ target!' if submission_result.get('score', 0) >= 30 else '📈 Keep improving! Target is 30%+'}
        """
        
        # Create results table
        results_data = []
        for result in results:
            results_data.append({
                "Task ID": result["task_id"],
                "Submitted Answer": result["submitted_answer"][:100] + "..." if len(result["submitted_answer"]) > 100 else result["submitted_answer"]
            })
        
        return status_message, results_data
    
    # Create Gradio interface
    with gr.Blocks(title="GAIA Research Agent Evaluation") as demo:
        gr.Markdown("# 🚀 GAIA Research Agent Evaluation")
        gr.Markdown("""
        Test your research agent on the GAIA benchmark - complex, multi-step research questions.
        
        **Goal:** Achieve 30%+ success rate to earn your certificate!
        """)
        
        with gr.Tab("🧪 Test Single Question"):
            gr.Markdown("Test your agent on a single random GAIA question")
            
            test_btn = gr.Button("🎲 Get Random Question & Test Agent", variant="primary")
            
            question_output = gr.Markdown()
            result_output = gr.Code(language="json")
            
            test_btn.click(
                fn=test_single_question,
                outputs=[question_output, result_output]
            )
        
        with gr.Tab("🏆 Full Evaluation & Submission"):
            gr.Markdown("Run your agent on all GAIA questions and submit to the leaderboard")
            
            gr.LoginButton()
            
            submit_btn = gr.Button("🚀 Run Full Evaluation & Submit", variant="primary")
            
            status_output = gr.Textbox(label="Evaluation Status", lines=10, interactive=False)
            results_table = gr.DataFrame(label="Question Results")
            
            submit_btn.click(
                fn=run_full_evaluation,
                outputs=[status_output, results_table]
            )
        
        gr.Markdown("""
        ### 📚 About GAIA
        
        GAIA (General AI Assistants) is a benchmark for evaluating AI agents on complex, multi-step research tasks.
        Your agent will be tested on questions requiring:
        
        - 🔍 **Multi-step reasoning**
        - 🌐 **Web search and information gathering**  
        - 🧮 **Mathematical calculations**
        - 📚 **Knowledge synthesis**
        - 🎯 **Precise, accurate answers**
        
        **Success Target:** 30%+ correct answers to earn your certificate!
        """)
    
    return demo

if __name__ == "__main__":
    print("🚀 GAIA Research Agent Evaluation System")
    print("=" * 50)
    
    # Check requirements
    required_keys = ["GROQ_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print(f"❌ Missing required API keys: {missing_keys}")
        print("Please add them to your .env file")
        exit(1)
    
    if not os.getenv("TAVILY_API_KEY"):
        print("⚠️  Warning: TAVILY_API_KEY missing - web search may be limited")
    
    print("✅ Creating GAIA evaluation interface...")
    
    # Create and launch interface
    demo = create_gaia_gradio_interface()
    
    print("🎯 GAIA Agent Ready for Evaluation!")
    print("🏆 Target: 30%+ success rate for certification!")
    
    demo.launch(debug=True, share=False)