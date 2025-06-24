"""
Step 1: Understanding LangChain Basics
=====================================

Key Concepts to Learn:
1. LLM (Large Language Model) - The "brain" of our agent
2. Messages - How we structure conversations 
3. Prompts - Instructions we give to the LLM
4. Chains - Connecting multiple steps together
"""

import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage

# Load environment variables from .env file
load_dotenv()

def basic_llm_example():
    """
    CONCEPT 1: Basic LLM Interaction
    ===============================
    - LLM is like a very smart function: input text → output text
    - We use ChatGroq as our LLM provider (fast and free)
    """
    print("🤖 Basic LLM Example")
    print("-" * 30)
    
    # Initialize the LLM
    llm = ChatGroq(
        model="llama-3.1-8b-instant",  # Fast, capable model
        temperature=0,  # 0 = deterministic, 1 = creative
    )
    
    # Simple string input
    response = llm.invoke("What is the capital of France?")
    print(f"Q: What is the capital of France?")
    print(f"A: {response.content}")
    print()

def messages_example():
    """
    CONCEPT 2: Messages - The LangChain Way
    ======================================
    - Instead of plain strings, LangChain uses "Message" objects
    - Different types: SystemMessage, HumanMessage, AIMessage
    - This gives structure and context to conversations
    """
    print("💬 Messages Example")
    print("-" * 30)
    
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0,
    )
    
    # Create structured messages
    messages = [
        SystemMessage(content="You are a helpful geography teacher. Be concise."),
        HumanMessage(content="What is the capital of Japan?")
    ]
    
    response = llm.invoke(messages)
    print("System: You are a helpful geography teacher. Be concise.")
    print("Human: What is the capital of Japan?")
    print(f"AI: {response.content}")
    print()

def conversation_example():
    """
    CONCEPT 3: Multi-turn Conversations
    ==================================
    - Keep track of conversation history
    - Each response becomes part of the context
    """
    print("🔄 Conversation Example")
    print("-" * 30)
    
    llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        temperature=0
    )
    
    # Start conversation
    messages = [
        SystemMessage(content="You are a helpful assistant. Keep responses brief."),
        HumanMessage(content="What's 15 + 27?")
    ]
    
    # First exchange
    response1 = llm.invoke(messages)
    print("Human: What's 15 + 27?")
    print(f"AI: {response1.content}")
    
    # Add AI response to conversation history
    messages.append(AIMessage(content=response1.content))
    
    # Continue conversation
    messages.append(HumanMessage(content="Now multiply that by 3"))
    response2 = llm.invoke(messages)
    print("Human: Now multiply that by 3")
    print(f"AI: {response2.content}")
    print()

def chain_example():
    """
    CONCEPT 4: Simple Chain
    ======================
    - Chain multiple LLM calls together
    - Output of one becomes input of next
    """
    print("⛓️  Chain Example")
    print("-" * 30)
    
    llm = ChatGroq(
        model="llama-3.1-8b-instant",
        temperature=0
    )
    
    # Step 1: Get a random fact
    step1_messages = [
        SystemMessage(content="Provide one interesting fact about space. Just the fact, nothing else."),
        HumanMessage(content="Tell me a space fact")
    ]
    fact_response = llm.invoke(step1_messages)
    
    print("Step 1 - Generate fact:")
    print(f"Fact: {fact_response.content}")
    
    # Step 2: Explain it simply  
    step2_messages = [
        SystemMessage(content="Explain this fact in simple terms a 10-year-old would understand."),
        HumanMessage(content=f"Explain this: {fact_response.content}")
    ]
    explanation_response = llm.invoke(step2_messages)
    
    print("\nStep 2 - Simplify explanation:")
    print(f"Simple explanation: {explanation_response.content}")
    print()

if __name__ == "__main__":
    print("🚀 Welcome to LangChain Basics!")
    print("=" * 50)
    print()
    
    # Check if API key exists
    if not os.getenv("GROQ_API_KEY"):
        print("❌ Error: GROQ_API_KEY not found in .env file")
        print("Please add it to your .env file:")
        print("GROQ_API_KEY=your_key_here")
        exit(1)
    
    print("✅ API key found! Running examples...")
    print()
    
    # Run all examples
    basic_llm_example()
    messages_example() 
    conversation_example()
    chain_example()
    
    print("🎉 Great! You now understand LangChain basics.")
    print("Next: We'll build a simple agent with tools!")