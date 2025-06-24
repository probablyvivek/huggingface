"""
Step 2: Building Agents with Tools
=================================

New Concepts to Learn:
1. Tools - Functions your AI can call
2. Function Calling - LLM decides which tool to use
3. Tool Results - How to handle tool outputs
4. Agent Loop - Question → Think → Use Tool → Answer
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.tools import tool

# Load environment variables
load_dotenv()

# CONCEPT 1: Defining Tools
# ========================
# Tools are Python functions that your AI can call
# The @tool decorator makes them available to the LLM

@tool
def calculator(expression: str) -> str:
    """
    Calculate mathematical expressions safely.
    
    Args:
        expression: A mathematical expression like "15 + 27" or "42 * 3"
    
    Returns:
        The result of the calculation
    """
    try:
        # Safe evaluation - only allows basic math
        result = eval(expression)
        return f"Result: {result}"
    except Exception as e:
        return f"Error: {e}"

@tool  
def get_word_length(word: str) -> str:
    """
    Count the number of characters in a word.
    
    Args:
        word: The word to count characters for
        
    Returns:
        The number of characters in the word
    """
    return f"The word '{word}' has {len(word)} characters."

@tool
def reverse_text(text: str) -> str:
    """
    Reverse the order of characters in text.
    
    Args:
        text: The text to reverse
        
    Returns:
        The text with characters in reverse order
    """
    return f"Reversed: {text[::-1]}"

def demonstrate_tool_calling():
    """
    CONCEPT 2: Tool Calling
    ======================
    - LLM can decide which tools to use based on the question
    - Tools are bound to the LLM using .bind_tools()
    - LLM returns tool calls instead of direct answers
    """
    print("🛠️  Tool Calling Example")
    print("-" * 30)
    
    # Initialize LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )
    
    # Bind tools to the LLM - now it can use them!
    tools = [calculator, get_word_length, reverse_text]
    llm_with_tools = llm.bind_tools(tools)
    
    # Ask a question that requires a tool
    messages = [
        SystemMessage(content="You are a helpful assistant. Use tools when needed to answer questions accurately."),
        HumanMessage(content="What is 25 * 8?")
    ]
    
    response = llm_with_tools.invoke(messages)
    
    print("Human: What is 25 * 8?")
    print(f"AI Response Type: {type(response)}")
    
    # Check if the LLM wants to use a tool
    # Safely access tool_calls, which may not exist on a BaseMessage
    tool_calls = getattr(response, "tool_calls", None)
    if tool_calls:
        print("🔧 AI decided to use a tool!")
        for tool_call in tool_calls:
            print(f"Tool: {tool_call['name']}")
            print(f"Arguments: {tool_call['args']}")
            
            # Execute the tool manually (we'll automate this later)
            if tool_call['name'] == 'calculator':
                result = calculator.invoke(tool_call['args'])
                print(f"Tool Result: {result}")
    else:
        print("AI didn't use any tools")
        print(f"Direct response: {response.content}")
    print()

def demonstrate_multiple_tools():
    """
    CONCEPT 3: Multiple Tools
    ========================
    - AI can choose from multiple available tools
    - It picks the right tool based on the question
    """
    print("🎯 Multiple Tools Example")
    print("-" * 30)
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    tools = [calculator, get_word_length, reverse_text]
    llm_with_tools = llm.bind_tools(tools)
    
    questions = [
        "How many letters are in the word 'python'?",
        "What's 144 divided by 12?", 
        "Can you reverse the text 'hello world'?"
    ]
    
    for question in questions:
        print(f"Q: {question}")
        
        messages = [
            SystemMessage(content="Use the appropriate tool to answer the question."),
            HumanMessage(content=question)
        ]
        
        response = llm_with_tools.invoke(messages)
        
        # Safely access tool_calls, which may not exist on a BaseMessage
        tool_calls = getattr(response, "tool_calls", None)
        if tool_calls:
            tool_call = tool_calls[0]  # Get first tool call
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            print(f"🔧 Using tool: {tool_name}")
            
            # Execute the appropriate tool
            if tool_name == 'calculator':
                result = calculator.invoke(tool_args)
            elif tool_name == 'get_word_length':
                result = get_word_length.invoke(tool_args)
            elif tool_name == 'reverse_text':
                result = reverse_text.invoke(tool_args)
            else:
                result = "Unknown tool"
                
            print(f"A: {result}")
        else:
            print(f"A: {response.content}")
        print()

def demonstrate_agent_loop():
    """
    CONCEPT 4: Simple Agent Loop
    ===========================
    - Question → Think → Use Tool → Answer
    - This is the basic pattern for all AI agents
    """
    print("🔄 Agent Loop Example")
    print("-" * 30)
    
    llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
    tools = [calculator, get_word_length, reverse_text]
    llm_with_tools = llm.bind_tools(tools)
    
    def simple_agent(question: str):
        """A simple agent that can use tools"""
        
        print(f"🤔 Question: {question}")
        
        # Step 1: Think about what tool to use
        messages = [
            SystemMessage(content="""You are a helpful assistant. 
            Analyze the question and use the appropriate tool if needed.
            Available tools: calculator, get_word_length, reverse_text"""),
            HumanMessage(content=question)
        ]
        
        response = llm_with_tools.invoke(messages)
        
        # Step 2: Use tool if needed
        tool_calls = getattr(response, "tool_calls", None)
        if tool_calls:
            tool_call = tool_calls[0]
            tool_name = tool_call['name']
            tool_args = tool_call['args']
            print(f"🔧 Using: {tool_name} with {tool_args}")
            
            # Execute tool
            if tool_name == 'calculator':
                tool_result = calculator.invoke(tool_args)
            elif tool_name == 'get_word_length':
                tool_result = get_word_length.invoke(tool_args)
            elif tool_name == 'reverse_text':
                tool_result = reverse_text.invoke(tool_args)
            
            print(f"🎯 Result: {tool_result}")
            
            # Step 3: Give final answer using tool result
            final_messages = messages + [
                response,  # Include the tool call
                HumanMessage(content=f"Tool result: {tool_result}. Please give a final answer.")
            ]
            
            final_response = llm.invoke(final_messages)
            print(f"✅ Final Answer: {final_response.content}")
            
        else:
            print(f"✅ Direct Answer: {response.content}")
        
        print("-" * 40)
    
    # Test the agent
    simple_agent("What's the square root of 144?")
    simple_agent("How many characters are in 'LangChain'?")
    simple_agent("What's the weather like today?")  # No tool needed

if __name__ == "__main__":
    print("🚀 Welcome to Agents with Tools!")
    print("=" * 50)
    print()
    
    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY not found")
        exit(1)
    
    print("✅ Running tool examples...")
    print()
    
    # Run all examples
    demonstrate_tool_calling()
    demonstrate_multiple_tools()
    demonstrate_agent_loop()
    
    print("🎉 Great! You now understand tools and basic agents.")
    print("Next: We'll add web search and build a research agent!")