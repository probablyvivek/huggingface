"""
Step 4: Complete LangGraph Research Agent
=========================================

Advanced Concepts:
1. LangGraph - State-based agent workflows
2. Agent State - Memory and context management
3. Conditional Routing - Smart decision making
4. Vector Database - Question similarity matching
5. Multi-step Reasoning - Complex research workflows
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List
import operator

# LangGraph imports
from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode, tools_condition

# LangChain imports  
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage, BaseMessage
from langchain_core.tools import tool

# Vector database imports
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv()

# ==========================================
# CONCEPT 1: Agent State Management
# ==========================================
# LangGraph uses state to track the conversation and research progress

class AgentState(TypedDict):
    """
    State that persists across all agent steps.
    This is like the agent's "memory" and "workspace"
    """
    messages: Annotated[List[BaseMessage], operator.add]  # Conversation history
    research_notes: str  # What we've learned so far
    question_type: str  # Type of question (factual, current, calculation)
    sources_used: List[str]  # Track information sources
    confidence: float  # How confident we are in our answer

# ==========================================
# TOOLS (Enhanced versions from previous steps)
# ==========================================

@tool
def web_search(query: str) -> str:
    """Search the web for current information."""
    try:
        from tavily import TavilyClient
        tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        result = tavily.search(query, max_results=3)
        
        if result and 'results' in result:
            formatted = []
            for i, r in enumerate(result['results'][:3], 1):
                formatted.append(f"Source {i}: {r.get('title', 'No title')}\n{r.get('content', 'No content')[:200]}...\nURL: {r.get('url', 'No URL')}")
            return "\n\n".join(formatted)
        return f"No results found for '{query}'"
    except Exception as e:
        return f"Search error: {e}"

@tool
def wikipedia_search(topic: str) -> str:
    """Search Wikipedia for factual information."""
    try:
        import wikipedia
        wikipedia.set_lang("en")
        summary = wikipedia.summary(topic, sentences=3)
        page = wikipedia.page(topic)
        return f"Wikipedia: {topic}\n{summary}\nSource: {page.url}"
    except Exception as e:
        return f"Wikipedia error: {e}"

@tool
def advanced_calculator(expression: str) -> str:
    """Advanced calculator with mathematical functions."""
    try:
        import math
        # Safe evaluation with math functions
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("__")
        }
        allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})
        
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Calculation: {expression} = {result}"
    except Exception as e:
        return f"Calculation error: {e}"

# All available tools
tools = [web_search, wikipedia_search, advanced_calculator]

# ==========================================
# CONCEPT 2: LangGraph Agent Nodes
# ==========================================
# Each node is a function that processes the state

def analyze_question(state: AgentState) -> AgentState:
    """
    ANALYZER NODE: Understand the question and plan research strategy
    """
    print("🤔 Analyzing question...")
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    # Get the latest human message
    last_message = state["messages"][-1]
    question = last_message.content if hasattr(last_message, 'content') else str(last_message)
    
    analysis_prompt = f"""
    Analyze this question and determine:
    1. Question type: "current" (news, events, real-time), "factual" (historical, definitions), or "calculation" (math, logic)
    2. Research strategy: what information sources would be most helpful
    3. Confidence level: how confident can we be in finding a good answer (0.0-1.0)
    
    Question: {question}
    
    Respond in this format:
    Type: [current/factual/calculation]
    Strategy: [brief description]
    Confidence: [0.0-1.0]
    """
    
    response = llm.invoke([HumanMessage(content=analysis_prompt)])
    analysis = response.content
    
    # Parse analysis (simple parsing for demo)
    question_type = "factual"  # default
    confidence = 0.7  # default
    
    if "current" in str(analysis).lower():
        question_type = "current"
    elif "calculation" in str(analysis).lower():
        question_type = "calculation"
    if "confidence: 0." in str(analysis).lower():
        try:
            confidence = float(str(analysis).split("Confidence: ")[1].split()[0])
        except:
            confidence = 0.7
    
    print(f"📊 Analysis: {question_type} question, confidence: {confidence}")
    
    return {
        "messages": state["messages"],
        "research_notes": f"Question analyzed as: {question_type}",
        "question_type": question_type,
        "sources_used": [],
        "confidence": confidence
    }

def research_assistant(state: AgentState) -> AgentState:
    """
    RESEARCH NODE: Use appropriate tools to gather information
    """
    print("🔍 Conducting research...")
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    
    # Build context-aware prompt
    question = state["messages"][-1].content
    question_type = state.get("question_type", "factual")
    
    system_prompt = f"""
    You are a research assistant. The question has been analyzed as: {question_type}
    
    Guidelines:
    - For "current" questions: Use web_search for latest information
    - For "factual" questions: Use wikipedia_search for reliable facts  
    - For "calculation" questions: Use advanced_calculator
    
    Choose the most appropriate tool and provide a focused query.
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]
    
    response = llm_with_tools.invoke(messages)
    
    # Add the tool call to our conversation
    updated_messages = state["messages"] + [response]
    
    return {
        "messages": updated_messages,
        "research_notes": state["research_notes"],
        "question_type": state["question_type"],
        "sources_used": state["sources_used"],
        "confidence": state["confidence"]
    }

def synthesize_answer(state: AgentState) -> AgentState:
    """
    SYNTHESIS NODE: Create final answer from research results
    """
    print("📝 Synthesizing final answer...")
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    # Get tool results from the conversation
    tool_results = []
    for msg in state["messages"]:
        if hasattr(msg, 'content') and 'Source' in str(msg.content):
            tool_results.append(msg.content)
    
    # Create synthesis prompt
    original_question = None
    for msg in state["messages"]:
        if isinstance(msg, HumanMessage):
            original_question = msg.content
            break
    
    synthesis_prompt = f"""
    Based on the research conducted, provide a comprehensive answer to the original question.
    
    Original Question: {original_question}
    Research Type: {state.get('question_type', 'unknown')}
    Confidence Level: {state.get('confidence', 0.7)}
    
    Research Results:
    {chr(10).join(tool_results) if tool_results else 'No tool results available'}
    
    Provide a clear, well-structured answer that:
    1. Directly answers the question
    2. Cites sources when available
    3. Indicates confidence level
    4. Mentions any limitations
    """
    
    response = llm.invoke([HumanMessage(content=synthesis_prompt)])
    
    # Update state with final answer
    final_messages = state["messages"] + [response]
    updated_sources = state["sources_used"] + ["synthesis"]
    
    return {
        "messages": final_messages,
        "research_notes": state["research_notes"] + f"\nFinal answer synthesized with confidence: {state['confidence']}",
        "question_type": state["question_type"],
        "sources_used": updated_sources,
        "confidence": state["confidence"]
    }

# ==========================================
# CONCEPT 3: Building the LangGraph Workflow
# ==========================================

def create_research_agent():
    """
    Create a complete research agent using LangGraph
    """
    print("🏗️  Building LangGraph research agent...")
    
    # Create the state graph
    workflow = StateGraph(AgentState)
    
    # Add nodes
    workflow.add_node("analyze", analyze_question)
    workflow.add_node("research", research_assistant)
    workflow.add_node("tools", ToolNode(tools))  # LangGraph's built-in tool executor
    workflow.add_node("synthesize", synthesize_answer)
    
    # Define the workflow
    workflow.add_edge(START, "analyze")
    workflow.add_edge("analyze", "research")
    
    # Conditional routing: if research node makes tool calls, go to tools
    workflow.add_conditional_edges(
        "research",
        tools_condition,  # Built-in function that checks for tool calls
        {
            "tools": "tools",  # If tools needed, go to tools node
            END: "synthesize"  # If no tools, go straight to synthesis
        }
    )
    
    # After tools, synthesize the answer
    workflow.add_edge("tools", "synthesize")
    workflow.add_edge("synthesize", END)
    
    # Compile the workflow
    agent = workflow.compile()
    
    print("✅ LangGraph agent created successfully!")
    return agent

# ==========================================
# CONCEPT 4: Enhanced Agent Interface
# ==========================================

def ask_research_agent(agent, question: str):
    """
    Ask the research agent a question and get a comprehensive answer
    """
    print(f"🚀 Research Agent Processing: '{question}'")
    print("=" * 60)
    
    # Initialize state
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "research_notes": "",
        "question_type": "",
        "sources_used": [],
        "confidence": 0.0
    }
    
    # Run the agent
    try:
        final_state = agent.invoke(initial_state)
        
        # Display results
        print("\n📋 Research Summary:")
        print("-" * 30)
        print(f"Question Type: {final_state.get('question_type', 'Unknown')}")
        print(f"Confidence: {final_state.get('confidence', 0.0):.1f}")
        print(f"Sources: {', '.join(final_state.get('sources_used', []))}")
        
        print("\n✅ Final Answer:")
        print("-" * 30)
        
        # Get the final answer (last AI message)
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and not hasattr(msg, 'tool_calls'):
                print(msg.content)
                break
        
        print("\n" + "=" * 60)
        return final_state
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    print("🚀 Complete LangGraph Research Agent")
    print("=" * 50)
    
    # Check API keys
    required_keys = ["GROQ_API_KEY"]
    missing_keys = [key for key in required_keys if not os.getenv(key)]
    
    if missing_keys:
        print(f"❌ Missing API keys: {missing_keys}")
        exit(1)
    
    if not os.getenv("TAVILY_API_KEY"):
        print("⚠️  Warning: TAVILY_API_KEY missing - web search limited")
    
    # Create the agent
    research_agent = create_research_agent()
    
    # Test with different types of questions
    test_questions = [
        "What is quantum computing and how does it work?",
        "What's the latest news about space exploration?", 
        "Calculate the compound interest on $10,000 at 5% annually for 10 years",
        "Who won the 2024 Nobel Prize in Physics?"
    ]
    
    print("\n🧪 Testing Complete Research Agent:")
    print("-" * 40)
    
    for question in test_questions:
        ask_research_agent(research_agent, question)
        print("\n" + "⏳ Next question..." + "\n")
    
    print("🎉 Your complete research agent is ready!")
    print("💡 Try asking it complex questions that require multiple steps!")