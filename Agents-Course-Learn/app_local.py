"""
GAIA Research Agent - Local Development Version
==============================================
Run locally for development and testing
"""

import os
import json
import requests
import gradio as gr
from datetime import datetime
from typing import Dict, List, Any, Optional, TypedDict, Annotated
import operator
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# LangGraph and LangChain imports
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.tools import tool

# Load environment variables
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY") 
HF_TOKEN = os.getenv("HF_TOKEN")
SPACE_ID = os.getenv("SPACE_ID", "local/gaia-research-agent")

# GAIA API Configuration
GAIA_API_BASE = "https://agents-course-unit4-scoring.hf.space"
QUESTIONS_ENDPOINT = f"{GAIA_API_BASE}/questions"
SUBMIT_ENDPOINT = f"{GAIA_API_BASE}/submit"
RANDOM_QUESTION_ENDPOINT = f"{GAIA_API_BASE}/random-question"

print("🔧 Local Environment Check:")
print(f"✅ GROQ_API_KEY: {'Set' if GROQ_API_KEY else '❌ Missing'}")
print(f"✅ TAVILY_API_KEY: {'Set' if TAVILY_API_KEY else '⚠️ Missing (optional)'}")
print(f"✅ HF_TOKEN: {'Set' if HF_TOKEN else '❌ Missing'}")

# ==========================================
# AGENT STATE DEFINITION
# ==========================================

class GAIAAgentState(TypedDict):
    """State for GAIA evaluation agent"""
    messages: Annotated[List[BaseMessage], operator.add]
    research_notes: str
    question_type: str
    sources_used: List[str]
    confidence: float
    task_id: str
    final_answer: str

# ==========================================
# RESEARCH TOOLS
# ==========================================

@tool
def advanced_web_search(query: str) -> str:
    """Enhanced web search for GAIA questions"""
    try:
        from tavily import TavilyClient
        
        if not TAVILY_API_KEY:
            return "Web search unavailable - TAVILY_API_KEY not configured in .env file"
        
        tavily = TavilyClient(api_key=TAVILY_API_KEY)
        result = tavily.search(query, max_results=5)
        
        if result and 'results' in result:
            formatted_results = []
            for i, r in enumerate(result['results'][:3], 1):
                title = r.get('title', 'No title')
                content = r.get('content', 'No content')
                url = r.get('url', 'No URL')
                
                formatted_results.append(f"""
=== Source {i} ===
Title: {title}
Content: {content[:250]}...
URL: {url}
""")
            
            return f"Web search results for '{query}':\n" + "\n".join(formatted_results)
        return f"No web results found for '{query}'"
        
    except ImportError:
        return "Tavily package not available - install with: pip install tavily-python"
    except Exception as e:
        return f"Web search error: {e}"

@tool
def enhanced_wikipedia_search(topic: str) -> str:
    """Enhanced Wikipedia search with better error handling"""
    try:
        import wikipedia
        wikipedia.set_lang("en")
        
        try:
            page = wikipedia.page(topic)
            summary = wikipedia.summary(topic, sentences=4)
            
            return f"""
Wikipedia: {topic}
Summary: {summary}
URL: {page.url}
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
            
    except ImportError:
        return "Wikipedia package not available - install with: pip install wikipedia"
    except Exception as e:
        return f"Wikipedia search error for '{topic}': {e}"

@tool
def professional_calculator(expression: str) -> str:
    """Professional calculator with extensive math functions"""
    try:
        import math
        
        # Enhanced math functions
        allowed_names = {
            # Basic math
            'abs': abs, 'round': round, 'min': min, 'max': max,
            'sum': sum, 'pow': pow,
            
            # Math module
            **{k: v for k, v in math.__dict__.items() if not k.startswith("__")},
            
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
# AGENT NODES (Simplified for local testing)
# ==========================================

def gaia_analyze_question(state: GAIAAgentState) -> GAIAAgentState:
    """Analyze GAIA question with enhanced strategy"""
    
    if not GROQ_API_KEY:
        return {**state, "research_notes": "Error: GROQ_API_KEY not configured in .env file"}
    
    try:
        from pydantic import SecretStr
    
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=SecretStr(GROQ_API_KEY))
        
        question = state["messages"][-1].content
        
        analysis_prompt = f"""
        Analyze this GAIA evaluation question for research strategy.
        
        Question: {question}
        
        Determine:
        1. Complexity: simple, medium, complex  
        2. Required tools: web_search, wikipedia, calculator
        3. Confidence prediction (0.0-1.0)
        
        Format:
        Complexity: [simple/medium/complex]
        Tools: [list tools needed]
        Confidence: [0.0-1.0]
        """
        
        response = llm.invoke([HumanMessage(content=analysis_prompt)])
        analysis = response.content
        
        # Parse analysis
        complexity = "medium"
        confidence = 0.6
        
        if "complex" in str(analysis).lower():
            complexity = "complex"
            confidence = 0.5
        elif "simple" in str(analysis).lower():
            complexity = "simple"
            confidence = 0.8
        
        print(f"📊 Analysis: {complexity} complexity, confidence: {confidence:.2f}")
        
        return {
            **state,
            "research_notes": f"Analysis: {complexity} complexity\n{analysis}",
            "question_type": complexity,
            "confidence": confidence
        }
        
    except Exception as e:
        print(f"❌ Analysis error: {e}")
        return {
            **state,
            "research_notes": f"Analysis error: {e}",
            "question_type": "unknown",
            "confidence": 0.3
        }

def gaia_research_orchestrator(state: GAIAAgentState) -> GAIAAgentState:
    """Orchestrate research for GAIA questions"""
    
    if not GROQ_API_KEY:
        return {**state, "research_notes": state["research_notes"] + "\nError: GROQ_API_KEY not configured"}
    
    try:
        from pydantic import SecretStr
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=SecretStr(GROQ_API_KEY))
        llm_with_tools = llm.bind_tools(gaia_tools)
        
        question = state["messages"][-1].content
        complexity = state.get("question_type", "medium")
        
        system_prompt = f"""
        Research this GAIA question systematically.
        
        Question Complexity: {complexity}
        
        Available tools:
        - advanced_web_search: For current information, news, recent events
        - enhanced_wikipedia_search: For factual, historical, biographical information  
        - professional_calculator: For mathematical calculations
        
        Choose the most appropriate tool for this question.
        """
        
        messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Research this GAIA question: {question}")
        ]
        
        response = llm_with_tools.invoke(messages)
        updated_messages = state["messages"] + [response]
        
        print(f"🔧 Research orchestrated")
        
        return {
            **state,
            "messages": updated_messages
        }
        
    except Exception as e:
        print(f"❌ Research error: {e}")
        error_msg = AIMessage(content=f"Research error: {e}")
        return {
            **state,
            "messages": state["messages"] + [error_msg]
        }

def gaia_answer_synthesizer(state: GAIAAgentState) -> GAIAAgentState:
    """Synthesize final answer for GAIA submission"""
    
    if not GROQ_API_KEY:
        return {
            **state,
            "final_answer": "Error: GROQ_API_KEY not configured",
            "sources_used": state["sources_used"] + ["error"]
        }
    
    try:
        from pydantic import SecretStr
        llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0, api_key=SecretStr(GROQ_API_KEY))
        
        # Extract information
        original_question = state["messages"][0].content
        research_notes = state.get("research_notes", "")
        
        # Get tool results
        tool_results = []
        for msg in state["messages"]:
            if hasattr(msg, 'content') and any(keyword in str(msg.content) for keyword in ['Source', 'Wikipedia', 'Calculation', '===']):
                tool_results.append(msg.content)
        
        synthesis_prompt = f"""
        Provide a direct, precise answer for this GAIA question.
        
        Question: {original_question}
        Research Notes: {research_notes}
        
        Tool Results:
        {chr(10).join(tool_results) if tool_results else 'No tool results available'}
        
        IMPORTANT: 
        - Give ONLY the final answer, no explanations
        - Be precise with numbers, dates, names
        - Do not include "FINAL ANSWER:" or similar prefixes
        """
        
        response = llm.invoke([HumanMessage(content=synthesis_prompt)])
        final_answer = str(response.content).strip()
        
        print(f"✅ Final Answer: {final_answer}")
        
        final_messages = state["messages"] + [response]
        
        return {
            **state,
            "messages": final_messages,
            "final_answer": final_answer,
            "sources_used": state["sources_used"] + ["synthesis"]
        }
        
    except Exception as e:
        print(f"❌ Synthesis error: {e}")
        return {
            **state,
            "final_answer": f"Synthesis error: {e}",
            "sources_used": state["sources_used"] + ["error"]
        }

# ==========================================
# AGENT CREATION
# ==========================================

def create_gaia_agent():
    """Create optimized agent for GAIA evaluation"""
    
    print("🏗️  Building GAIA Agent...")
    
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
    
    def get_random_question(self) -> Optional[Dict[str, Any]]:
        """Fetch a random GAIA question"""
        try:
            print("📥 Fetching random question...")
            response = self.session.get(RANDOM_QUESTION_ENDPOINT, timeout=15)
            response.raise_for_status()
            question = response.json()
            print(f"✅ Got question: {question.get('task_id', 'unknown')}")
            return question
        except Exception as e:
            print(f"❌ Error fetching random question: {e}")
            return None
    
    def process_question(self, question_data: Dict[str, Any]) -> Dict[str, Any]:
        """Process a single GAIA question"""
        task_id = question_data.get('task_id', 'unknown')
        question_text = question_data.get('question', '')
        
        print(f"🤖 Processing: {task_id}")
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
                        final_answer = str(msg.content).strip()
                        if final_answer:
                            break
            
            result = {
                "task_id": task_id,
                "submitted_answer": final_answer,
                "confidence": final_state.get("confidence", 0.0),
                "sources_used": final_state.get("sources_used", []),
                "success": True,
                "research_notes": final_state.get("research_notes", "")
            }
            
            print(f"✅ Success: {final_answer}")
            return result
            
        except Exception as e:
            print(f"❌ Processing error: {e}")
            return {
                "task_id": task_id,
                "submitted_answer": f"Error: {str(e)}",
                "confidence": 0.0,
                "sources_used": [],
                "success": False,
                "research_notes": f"Processing error: {e}"
            }

# ==========================================
# LOCAL TESTING FUNCTIONS
# ==========================================

def test_local_agent():
    """Test the agent locally with a simple question"""
    print("\n🧪 Testing Local Agent...")
    
    try:
        # Create agent
        agent = create_gaia_agent()
        interface = GAIAInterface(agent)
        
        # Test with a simple question
        test_question = {
            "task_id": "local_test",
            "question": "What is the capital of France?"
        }
        
        result = interface.process_question(test_question)
        
        print(f"\n📊 Test Results:")
        print(f"Answer: {result['submitted_answer']}")
        print(f"Confidence: {result['confidence']}")
        print(f"Sources: {', '.join(result['sources_used'])}")
        print(f"Success: {'✅' if result['success'] else '❌'}")
        
        return result['success']
        
    except Exception as e:
        print(f"❌ Local test failed: {e}")
        return False

def create_local_interface():
    """Create simplified Gradio interface for local testing"""
    
    # Initialize agent
    try:
        gaia_agent = create_gaia_agent()
        gaia_interface = GAIAInterface(gaia_agent)
        agent_ready = True
        print("✅ Agent initialized successfully")
    except Exception as e:
        agent_ready = False
        error_msg = str(e)
        print(f"❌ Agent initialization failed: {e}")
    
    def test_single_question():
        """Test agent on a single random question"""
        if not agent_ready:
            return f"❌ Agent not ready: {error_msg}", ""
        
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
    
    # Create simple interface
    with gr.Blocks(title="GAIA Research Agent - Local Development") as demo:
        gr.Markdown("# 🏠 GAIA Research Agent - Local Development")
        
        if not agent_ready:
            gr.Markdown(f"❌ **Agent Status:** {error_msg}")
        else:
            gr.Markdown("✅ **Agent Status:** Ready for local testing!")
        
        gr.Markdown("### 🧪 Test Your Agent")
        
        test_btn = gr.Button("🎲 Get Random Question & Test Agent", variant="primary")
        
        question_output = gr.Markdown()
        result_output = gr.Code(language="json")
        
        test_btn.click(
            fn=test_single_question,
            outputs=[question_output, result_output]
        )
        
        gr.Markdown("""
        ### 🔧 Local Development Notes
        
        - **Environment**: Check your `.env` file has all required keys
        - **Dependencies**: Ensure all packages are installed
        - **Testing**: Use this interface to test before deploying
        - **Debugging**: Check console output for detailed logs
        """)
    
    return demo

# ==========================================
# MAIN APPLICATION
# ==========================================

if __name__ == "__main__":
    print("🚀 GAIA Research Agent - Local Development")
    print("=" * 50)
    
    # Quick local test
    if test_local_agent():
        print("\n🎉 Agent working! Starting Gradio interface...")
        
        # Create and launch interface
        demo = create_local_interface()
        demo.launch(
            debug=True,
            share=False,
            server_name="127.0.0.1",  # Local only
            server_port=7860
        )
    else:
        print("\n❌ Agent test failed. Please check your .env file and API keys.")
        print("\nRequired in .env file:")
        print("- GROQ_API_KEY=your_groq_key")
        print("- HF_TOKEN=your_hf_token") 
        print("- TAVILY_API_KEY=your_tavily_key (optional)")