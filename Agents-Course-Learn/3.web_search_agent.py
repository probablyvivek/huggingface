"""
Step 3: Building a Research Agent with Web Search
================================================

New Concepts to Learn:
1. Web Search Tools - Access real-time information
2. Multiple Tool Types - Local vs External APIs
3. Information Synthesis - Combining search results
4. Research Agent Pattern - Search → Analyze → Answer
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool
import wikipedia 

# Load environment variables
load_dotenv()

# EXISTING TOOLS (from your working code)
@tool
def calculator(expression: str) -> str:
    """Calculate mathematical expressions safely."""
    try:
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

@tool  
def get_word_length(word: str) -> str:
    """Count the number of characters in a word."""
    return f"The word '{word}' has {len(word)} characters."

# NEW CONCEPT 1: Web Search Tool
# =============================
# This tool can search the internet for current information

@tool
def web_search(query: str) -> str:
    """
    Search the web for current information.
    
    Args:
        query: Search query (e.g., "latest news about AI", "weather in Tokyo")
    
    Returns:
        Search results with relevant information
    """
    try:
        from tavily import TavilyClient
        
        # Initialize Tavily client
        tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        
        # Perform search
        search_result = tavily.search(query, max_results=3)
        
        # Format results
        if search_result and 'results' in search_result:
            formatted_results = []
            for i, result in enumerate(search_result['results'][:3], 1):
                title = result.get('title', 'No title')
                content = result.get('content', 'No content')
                url = result.get('url', 'No URL')
                
                formatted_results.append(f"""
Result {i}:
Title: {title}
Content: {content[:200]}...
Source: {url}
""")
            
            return f"Web search results for '{query}':\n" + "\n".join(formatted_results)
        else:
            return f"No results found for '{query}'"
            
    except ImportError:
        return "Tavily package not installed. Run: uv add tavily-python"
    except Exception as e:
        return f"Search error: {e}"

# NEW CONCEPT 2: Wikipedia Tool  
# =============================
# For factual, encyclopedic information

@tool
def wikipedia_search(topic: str) -> str:
    """
    Search Wikipedia for factual information.
    
    Args:
        topic: Topic to search for (e.g., "Python programming", "Tokyo")
    
    Returns:
        Wikipedia summary and key information
    """
    try:
        import wikipedia
        
        # Search Wikipedia
        wikipedia.set_lang("en")
        summary = wikipedia.summary(topic, sentences=3)
        page = wikipedia.page(topic)
        
        return f"""
Wikipedia: {topic}
Summary: {summary}
URL: {page.url}
"""
    except ImportError:
        return "Wikipedia package not installed. Run: uv add wikipedia"
    except wikipedia.exceptions.DisambiguationError as e:
        # If multiple pages match, use the first one
        try:
            summary = wikipedia.summary(e.options[0], sentences=2)
            return f"Wikipedia: {topic}\nSummary: {summary}\n(Note: Multiple pages found, showing first match)"
        except:
            return f"Multiple Wikipedia pages found for '{topic}': {e.options[:5]}"
    except wikipedia.exceptions.PageError:
        return f"No Wikipedia page found for '{topic}'"
    except Exception as e:
        return f"Wikipedia error: {e}"

def demonstrate_web_search():
    """
    CONCEPT 3: Web Search in Action
    ==============================
    - Search for current information not in training data
    - Handle real-time queries
    """
    print("🌐 Web Search Example")
    print("-" * 30)
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    tools = [calculator, web_search, wikipedia_search]
    llm_with_tools = llm.bind_tools(tools)
    
    questions = [
        "What's the latest news about SpaceX?",
        "What's the current weather in London?",
        "Tell me about Python programming language"
    ]
    
    for question in questions:
        print(f"🤔 Q: {question}")
        
        messages = [
            SystemMessage(content="""You are a research assistant. Use web search for current/recent information, 
            Wikipedia for factual/historical information, and calculator for math. Be helpful and accurate."""),
            HumanMessage(content=question)
        ]
        
        response = llm_with_tools.invoke(messages)
        
        # Using your working tool call pattern
        tool_calls = getattr(response, "tool_calls", None)
        if tool_calls:
            tool_call = tool_calls[0]
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            
            print(f"🔧 Using: {tool_name}")
            print(f"📝 Query: {tool_args}")
            
            # Execute the appropriate tool
            if tool_name == 'web_search':
                result = web_search.invoke(tool_args)
            elif tool_name == 'wikipedia_search':
                result = wikipedia_search.invoke(tool_args)
            elif tool_name == 'calculator':
                result = calculator.invoke(tool_args)
            else:
                result = "Unknown tool"
            
            print(f"🎯 Result: {result[:300]}...")  # Show first 300 chars
        else:
            print(f"💭 Direct answer: {response.content}")
        
        print("-" * 50)

def research_agent(question: str):
    """
    CONCEPT 4: Research Agent Pattern
    ================================
    - Analyze question type
    - Choose appropriate information source
    - Synthesize and present findings
    """
    print(f"🕵️‍♂️ Research Agent Working on: {question}")
    print("=" * 60)
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    tools = [calculator, web_search, wikipedia_search, get_word_length]
    llm_with_tools = llm.bind_tools(tools)
    
    # Step 1: Initial analysis and tool selection
    messages = [
        SystemMessage(content="""You are an expert research assistant. Analyze the question and:

1. For CURRENT/RECENT events, news, weather, prices → use web_search
2. For FACTUAL/HISTORICAL info, definitions, science → use wikipedia_search  
3. For MATH calculations → use calculator
4. For SIMPLE text operations → use get_word_length

Choose the most appropriate tool and provide a focused query."""),
        HumanMessage(content=question)
    ]
    
    response = llm_with_tools.invoke(messages)
    
    # Step 2: Execute tool if needed
    tool_calls = getattr(response, "tool_calls", None)
    if tool_calls:
        tool_call = tool_calls[0]
        tool_name = tool_call['name']
        tool_args = tool_call['args']
        
        print(f"🔍 Strategy: Using {tool_name}")
        print(f"📋 Query: {tool_args}")
        print()
        
        # Execute tool
        if tool_name == 'web_search':
            tool_result = web_search.invoke(tool_args)
        elif tool_name == 'wikipedia_search':
            tool_result = wikipedia_search.invoke(tool_args)
        elif tool_name == 'calculator':
            tool_result = calculator.invoke(tool_args)
        elif tool_name == 'get_word_length':
            tool_result = get_word_length.invoke(tool_args)
        else:
            tool_result = "Tool execution failed"
        
        print(f"📊 Raw Results:")
        print(tool_result[:500] + "..." if len(tool_result) > 500 else tool_result)
        print()
        
        # Step 3: Synthesize findings
        synthesis_messages = [
            SystemMessage(content="You are a research assistant. Based on the tool results, provide a clear, comprehensive answer to the original question. Include sources when relevant."),
            HumanMessage(content=f"Original question: {question}"),
            HumanMessage(content=f"Tool results: {tool_result}"),
            HumanMessage(content="Please provide a final answer based on these results.")
        ]
        
        final_response = llm.invoke(synthesis_messages)
        print("✅ Final Answer:")
        print(final_response.content)
        
    else:
        print("💭 Direct Response (no tools needed):")
        print(response.content)
    
    print("\n" + "=" * 60)

if __name__ == "__main__":
    print("🚀 Welcome to Web Search Agent!")
    print("=" * 50)
    
    # Check API keys
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY not found")
        exit(1)
    
    if not os.getenv("TAVILY_API_KEY"):
        print("⚠️  Warning: TAVILY_API_KEY not found - web search will be limited")
    
    print("✅ Running web search examples...")
    print()
    
    # Test specific capabilities
    print("🧪 Testing Research Agent:")
    print("-" * 30)
    
    # Test different types of questions
    test_questions = [
        "What is the capital of France?",  # Should use Wikipedia
        "What's 157 * 89?",  # Should use calculator  
        "What are the latest developments in AI safety research?"  # Should use web search
    ]
    
    for question in test_questions:
        research_agent(question)
        print()
    
    print("🎉 Your research agent is ready!")
    print("Try asking it about current events, calculations, or factual information!")