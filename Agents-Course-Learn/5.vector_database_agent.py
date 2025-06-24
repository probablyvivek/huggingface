"""
Step 5: Vector Database + Question Similarity Agent
=================================================

Revolutionary Concepts:
1. Vector Embeddings - Converting questions to numbers
2. Similarity Search - Finding similar questions
3. Few-shot Learning - Learning from examples
4. Knowledge Base - Building agent memory
5. Example-driven Reasoning - Using past solutions
"""

import os
import json
from dotenv import load_dotenv
from typing import TypedDict, Annotated, List, Dict, Any
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
from langchain_core.documents import Document

load_dotenv()

# ==========================================
# CONCEPT 1: Question-Answer Knowledge Base
# ==========================================
# Sample questions and answers for our vector database

SAMPLE_QA_DATABASE = [
    {
        "question": "What is the capital of France?",
        "answer": "Paris",
        "reasoning": "Direct factual question requiring Wikipedia lookup for geographical information.",
        "tools_used": ["wikipedia_search"],
        "confidence": 0.95
    },
    {
        "question": "What's 15 times 27 plus 8?",
        "answer": "413",
        "reasoning": "Mathematical calculation requiring step-by-step arithmetic: (15 * 27) + 8 = 405 + 8 = 413",
        "tools_used": ["advanced_calculator"],
        "confidence": 1.0
    },
    {
        "question": "What are the latest developments in artificial intelligence?",
        "answer": "Recent AI developments include large language models, improved computer vision, and advances in robotics automation.",
        "reasoning": "Current events question requiring web search for latest information.",
        "tools_used": ["web_search"],
        "confidence": 0.8
    },
    {
        "question": "Who invented the telephone?",
        "answer": "Alexander Graham Bell",
        "reasoning": "Historical factual question best answered through Wikipedia for reliable information.",
        "tools_used": ["wikipedia_search"],
        "confidence": 0.9
    },
    {
        "question": "Calculate the area of a circle with radius 7",
        "answer": "153.94 square units",
        "reasoning": "Mathematical calculation using formula A = π * r². Need calculator for precision: π * 7² = π * 49 ≈ 153.94",
        "tools_used": ["advanced_calculator"],
        "confidence": 1.0
    },
    {
        "question": "What's happening in space exploration recently?",
        "answer": "Recent space developments include SpaceX missions, Mars rover discoveries, and international space station activities.",
        "reasoning": "Current events about space require web search for latest news and developments.",
        "tools_used": ["web_search"],
        "confidence": 0.75
    }
]

# ==========================================
# CONCEPT 2: Enhanced Agent State with Vector Memory
# ==========================================

class VectorAgentState(TypedDict):
    """Enhanced state with vector database capabilities"""
    messages: Annotated[List[BaseMessage], operator.add]
    research_notes: str
    question_type: str
    sources_used: List[str]
    confidence: float
    similar_questions: List[Dict[str, Any]]  # NEW: Similar questions found
    learning_examples: str  # NEW: Formatted examples for few-shot learning

# ==========================================
# CONCEPT 3: Vector Database Setup
# ==========================================

class QuestionVectorDB:
    """Manages the question vector database"""
    
    def __init__(self, persist_directory: str = "./chroma_db"):
        print("🗄️  Initializing Vector Database...")
        
        # Initialize embeddings model
        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-mpnet-base-v2"
        )
        
        # Initialize ChromaDB
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
            collection_name="research_questions"
        )
        
        # Check if database is empty and populate it
        if self._is_empty():
            print("📚 Database empty, populating with sample questions...")
            self._populate_database()
        else:
            print("✅ Database loaded with existing questions")
    
    def _is_empty(self) -> bool:
        """Check if the vector database is empty"""
        try:
            # Try to get one document
            results = self.vectorstore.similarity_search("test", k=1)
            return len(results) == 0
        except:
            return True
    
    def _populate_database(self):
        """Add sample questions to the database"""
        documents = []
        
        for i, qa in enumerate(SAMPLE_QA_DATABASE):
            # Create document content
            content = f"""Question: {qa['question']}
Answer: {qa['answer']}
Reasoning: {qa['reasoning']}
Tools Used: {', '.join(qa['tools_used'])}
Confidence: {qa['confidence']}"""
            
            # Create document with metadata
            doc = Document(
                page_content=content,
                metadata={
                    "question_id": f"q_{i}",
                    "question": qa['question'],
                    "answer": qa['answer'],
                    "tools_used": qa['tools_used'],
                    "confidence": qa['confidence']
                }
            )
            documents.append(doc)
        
        # Add to vector database
        self.vectorstore.add_documents(documents)
        print(f"✅ Added {len(documents)} questions to database")
    
    def find_similar_questions(self, question: str, k: int = 2) -> List[Dict[str, Any]]:
        """Find similar questions in the database"""
        try:
            # Search for similar questions
            results = self.vectorstore.similarity_search_with_score(question, k=k)
            
            similar_questions = []
            for doc, score in results:
                similar_questions.append({
                    "question": doc.metadata.get("question", "Unknown"),
                    "answer": doc.metadata.get("answer", "Unknown"),
                    "tools_used": doc.metadata.get("tools_used", []),
                    "confidence": doc.metadata.get("confidence", 0.0),
                    "similarity_score": 1 - score,  # Convert distance to similarity
                    "full_content": doc.page_content
                })
            
            return similar_questions
        except Exception as e:
            print(f"❌ Error finding similar questions: {e}")
            return []
    
    def add_question(self, question: str, answer: str, reasoning: str, tools_used: List[str], confidence: float):
        """Add a new question-answer pair to the database"""
        content = f"""Question: {question}
Answer: {answer}
Reasoning: {reasoning}
Tools Used: {', '.join(tools_used)}
Confidence: {confidence}"""
        
        doc = Document(
            page_content=content,
            metadata={
                "question": question,
                "answer": answer,
                "tools_used": tools_used,
                "confidence": confidence
            }
        )
        
        self.vectorstore.add_documents([doc])
        print(f"📝 Added new question to database: {question[:50]}...")

# ==========================================
# CONCEPT 4: Enhanced Tools (same as before)
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
        allowed_names = {
            k: v for k, v in math.__dict__.items() if not k.startswith("__")
        }
        allowed_names.update({"abs": abs, "round": round, "min": min, "max": max})
        result = eval(expression, {"__builtins__": {}}, allowed_names)
        return f"Calculation: {expression} = {result}"
    except Exception as e:
        return f"Calculation error: {e}"

tools = [web_search, wikipedia_search, advanced_calculator]

# ==========================================
# CONCEPT 5: Vector-Enhanced Agent Nodes
# ==========================================

def find_similar_examples(state: VectorAgentState, vector_db: QuestionVectorDB) -> VectorAgentState:
    """
    NEW NODE: Find similar questions to learn from
    """
    print("🔍 Finding similar questions for learning...")
    
    # Get the current question
    current_question = state["messages"][-1].content
    
    # Search for similar questions
    question_text = current_question
    if not isinstance(question_text, str):
        # Message content can be a list of parts (e.g., for multimodal input).
        # We need to extract the text from it to perform a search.
        text_parts = []
        for part in question_text:
            if isinstance(part, str):
                text_parts.append(part)
            elif isinstance(part, dict) and "text" in part:
                text_parts.append(part["text"])
        question_text = " ".join(text_parts)

    similar_questions = vector_db.find_similar_questions(question_text, k=2)
    
    # Format learning examples
    learning_examples = ""
    if similar_questions:
        learning_examples = "Here are similar questions I've solved before:\n\n"
        for i, sim_q in enumerate(similar_questions, 1):
            learning_examples += f"Example {i} (similarity: {sim_q['similarity_score']:.2f}):\n"
            learning_examples += f"Q: {sim_q['question']}\n"
            learning_examples += f"A: {sim_q['answer']}\n"
            learning_examples += f"Method: {', '.join(sim_q['tools_used'])}\n"
            learning_examples += f"Confidence: {sim_q['confidence']}\n\n"
        
        print(f"📚 Found {len(similar_questions)} similar questions")
        for sim_q in similar_questions:
            print(f"  - {sim_q['question'][:60]}... (similarity: {sim_q['similarity_score']:.2f})")
    else:
        learning_examples = "No similar questions found in my memory."
        print("📚 No similar questions found")
    
    return {
        "messages": state["messages"],
        "research_notes": state["research_notes"],
        "question_type": state["question_type"],
        "sources_used": state["sources_used"],
        "confidence": state["confidence"],
        "similar_questions": similar_questions,
        "learning_examples": learning_examples
    }

def analyze_with_examples(state: VectorAgentState) -> VectorAgentState:
    """
    ENHANCED ANALYZER: Use similar examples to improve analysis
    """
    print("🤔 Analyzing question with learning examples...")
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    current_question = state["messages"][-1].content
    learning_examples = state.get("learning_examples", "")
    
    analysis_prompt = f"""
    Analyze this question using the similar examples I've solved before.
    
    Current Question: {current_question}
    
    {learning_examples}
    
    Based on these examples and the current question, determine:
    1. Question type: "current" (news, events), "factual" (historical, definitions), or "calculation" (math)
    2. Best tool strategy based on similar examples
    3. Confidence level (0.0-1.0) based on similarity to known examples
    
    Format:
    Type: [current/factual/calculation]
    Strategy: [which tool to use and why]
    Confidence: [0.0-1.0]
    Reasoning: [how similar examples inform this choice]
    """
    
    response = llm.invoke([HumanMessage(content=analysis_prompt)])
    analysis = response.content
    
    # Parse analysis
    question_type = "factual"
    confidence = 0.7
    
    assert isinstance(analysis, str)
    if "current" in analysis.lower():
        question_type = "current"
    elif "calculation" in analysis.lower():
        question_type = "calculation"
    try:
        if "confidence:" in analysis.lower():
            confidence_line = [line for line in analysis.split('\n') if 'confidence:' in line.lower()][0]
            confidence = float(confidence_line.split(':')[1].strip().split()[0])
    except:
        pass
    
    print(f"📊 Enhanced Analysis: {question_type}, confidence: {confidence:.2f}")
    
    return {
        "messages": state["messages"],
        "research_notes": f"Enhanced analysis with examples: {question_type}",
        "question_type": question_type,
        "sources_used": state["sources_used"],
        "confidence": confidence,
        "similar_questions": state["similar_questions"],
        "learning_examples": state["learning_examples"]
    }

def research_with_examples(state: VectorAgentState) -> VectorAgentState:
    """
    ENHANCED RESEARCH: Use examples to guide tool selection
    """
    print("🔍 Conducting research with example guidance...")
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    llm_with_tools = llm.bind_tools(tools)
    
    question = state["messages"][-1].content
    question_type = state.get("question_type", "factual")
    learning_examples = state.get("learning_examples", "")
    
    system_prompt = f"""
    You are a research assistant with access to previous examples.
    
    Question Type: {question_type}
    
    Previous Similar Examples:
    {learning_examples}
    
    Guidelines:
    - Learn from the examples: use similar tools and approaches
    - For "current" questions: use web_search
    - For "factual" questions: use wikipedia_search
    - For "calculation" questions: use advanced_calculator
    
    Choose the most appropriate tool based on the question and examples.
    """
    
    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=question)
    ]
    
    response = llm_with_tools.invoke(messages)
    updated_messages = state["messages"] + [response]
    
    return {
        "messages": updated_messages,
        "research_notes": state["research_notes"],
        "question_type": state["question_type"],
        "sources_used": state["sources_used"],
        "confidence": state["confidence"],
        "similar_questions": state["similar_questions"],
        "learning_examples": state["learning_examples"]
    }

def synthesize_with_memory(state: VectorAgentState) -> VectorAgentState:
    """
    ENHANCED SYNTHESIS: Create answers informed by examples
    """
    print("📝 Synthesizing answer with memory...")
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    
    # Get all relevant information
    original_question = state["messages"][0].content
    learning_examples = state.get("learning_examples", "")
    
    # Get tool results
    tool_results = []
    for msg in state["messages"]:
        if hasattr(msg, 'content') and any(keyword in str(msg.content) for keyword in ['Source', 'Wikipedia', 'Calculation']):
            tool_results.append(msg.content)
    
    synthesis_prompt = f"""
    Create a comprehensive answer using both research results and learning from examples.
    
    Original Question: {original_question}
    Question Type: {state.get('question_type', 'unknown')}
    Confidence: {state.get('confidence', 0.7)}
    
    Learning Examples:
    {learning_examples}
    
    Research Results:
    {chr(10).join(tool_results) if tool_results else 'No tool results available'}
    
    Provide an answer that:
    1. Directly answers the question
    2. Shows how you learned from similar examples
    3. Cites sources appropriately
    4. Indicates confidence level
    5. Explains your reasoning process
    """
    
    response = llm.invoke([HumanMessage(content=synthesis_prompt)])
    final_messages = state["messages"] + [response]
    
    return {
        "messages": final_messages,
        "research_notes": state["research_notes"] + f"\nSynthesis with memory completed",
        "question_type": state["question_type"],
        "sources_used": state["sources_used"] + ["vector_memory"],
        "confidence": state["confidence"],
        "similar_questions": state["similar_questions"],
        "learning_examples": state["learning_examples"]
    }

# ==========================================
# CONCEPT 6: Vector-Enhanced LangGraph Agent
# ==========================================

def create_vector_research_agent():
    """Create research agent with vector database memory"""
    print("🏗️  Building Vector-Enhanced Research Agent...")
    
    # Initialize vector database
    vector_db = QuestionVectorDB()
    
    # Create the state graph
    workflow = StateGraph(VectorAgentState)
    
    # Add nodes with vector database access
    workflow.add_node("find_examples", lambda state: find_similar_examples(state, vector_db))
    workflow.add_node("analyze", analyze_with_examples)
    workflow.add_node("research", research_with_examples)
    workflow.add_node("tools", ToolNode(tools))
    workflow.add_node("synthesize", synthesize_with_memory)
    
    # Define enhanced workflow
    workflow.add_edge(START, "find_examples")  # Start by finding similar examples
    workflow.add_edge("find_examples", "analyze")
    workflow.add_edge("analyze", "research")
    
    # Conditional routing for tools
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
    
    # Compile the workflow
    agent = workflow.compile()
    
    print("✅ Vector-Enhanced Research Agent Created!")
    return agent, vector_db

# ==========================================
# CONCEPT 7: Enhanced Agent Interface
# ==========================================

def ask_vector_agent(agent, question: str):
    """Ask the vector-enhanced research agent"""
    print(f"🚀 Vector Research Agent: '{question}'")
    print("=" * 70)
    
    # Initialize enhanced state
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "research_notes": "",
        "question_type": "",
        "sources_used": [],
        "confidence": 0.0,
        "similar_questions": [],
        "learning_examples": ""
    }
    
    try:
        final_state = agent.invoke(initial_state)
        
        # Display enhanced results
        print("\n📋 Enhanced Research Summary:")
        print("-" * 40)
        print(f"Question Type: {final_state.get('question_type', 'Unknown')}")
        print(f"Confidence: {final_state.get('confidence', 0.0):.2f}")
        print(f"Similar Examples Found: {len(final_state.get('similar_questions', []))}")
        print(f"Sources Used: {', '.join(final_state.get('sources_used', []))}")
        
        # Show similar questions
        if final_state.get('similar_questions'):
            print(f"\n🧠 Learning from Similar Questions:")
            for i, sim_q in enumerate(final_state['similar_questions'], 1):
                print(f"  {i}. {sim_q['question'][:60]}... (similarity: {sim_q['similarity_score']:.2f})")
        
        print("\n✅ Final Answer:")
        print("-" * 40)
        
        # Get final answer
        for msg in reversed(final_state["messages"]):
            if isinstance(msg, AIMessage) and not hasattr(msg, 'tool_calls'):
                print(msg.content)
                break
        
        print("\n" + "=" * 70)
        return final_state
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return None

if __name__ == "__main__":
    print("🚀 Vector Database Research Agent")
    print("=" * 50)
    
    # Check requirements
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Missing GROQ_API_KEY")
        exit(1)
    
    # Create vector-enhanced agent
    vector_agent, vector_db = create_vector_research_agent()
    
    # Test questions that should benefit from examples
    test_questions = [
        "What is the capital of Italy?",  # Similar to France question
        "Calculate 25 times 13 minus 4",  # Similar to math questions
        "What's new in machine learning?",  # Similar to AI developments
        "What is the area of a circle with radius 5?",  # Similar to circle area
        "Who invented the light bulb?",  # Similar to telephone question
    ]
    
    print("\n🧪 Testing Vector-Enhanced Agent:")
    print("-" * 50)
    
    for question in test_questions:
        ask_vector_agent(vector_agent, question)
        print("\n⏳ Processing next question...\n")
    
    print("🎉 Vector Database Agent Complete!")
    print("💡 Your agent now learns from similar questions!")
    print("📚 Try asking questions similar to the examples in the database!")