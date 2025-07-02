import asyncio
import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

@dataclass
class Configuration:
    llm_api_key: str
    llm_endpoint: str = "https://api.groq.com/openai/v1/chat/completions"
    llm_model: str = "llama-3.3-70b-versatile"
    servers_config_path: str = "servers_config.json"

class MCPServerManager:
    def __init__(self):
        self.servers = {}
        self.all_tools = []

    async def start_servers(self, servers_config):
        """Start all MCP servers and keep them running"""
        for name, config in servers_config["mcpServers"].items():
            print(f"🔌 Starting MCP server: {name}")
            
            server_params = StdioServerParameters(
                command=config["command"],
                args=config["args"],
                env=config.get("env")
            )
            
            try:
                # Store the context managers
                stdio_manager = stdio_client(server_params)
                read, write = await stdio_manager.__aenter__()
                
                session_manager = ClientSession(read, write)
                session = await session_manager.__aenter__()
                
                # Initialize and get tools
                await session.initialize()
                tools_response = await session.list_tools()
                tools = [tool.model_dump() for tool in tools_response.tools]
                
                # Store everything
                self.servers[name] = {
                    'session': session,
                    'tools': tools,
                    'stdio_manager': stdio_manager,
                    'session_manager': session_manager
                }
                self.all_tools.extend(tools)
                
                print(f"✅ {name} connected with {len(tools)} tools")
                for tool in tools:
                    print(f"   📦 {tool['name']}: {tool.get('description', 'No description')}")
                    
            except Exception as e:
                print(f"❌ Failed to start {name}: {e}")

    async def execute_tool(self, tool_name: str, arguments: dict):
        """Execute a tool on the appropriate MCP server"""
        for server_name, server_data in self.servers.items():
            for tool in server_data['tools']:
                if tool['name'] == tool_name:
                    try:
                        session = server_data['session']
                        result = await session.call_tool(tool_name, arguments)
                        return result.content
                    except Exception as e:
                        return f"Error executing {tool_name}: {e}"
        return f"Tool {tool_name} not found"

class LLMClient:
    def __init__(self, config: Configuration):
        self.config = config

    def chat_completion(self, messages: List[Dict], tools: Optional[List[Dict]] = None):
        """Send a chat completion request to the LLM"""
        headers = {
            "Authorization": f"Bearer {self.config.llm_api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.config.llm_model,
            "messages": messages,
            "temperature": 0.7
        }
        
        if tools and len(tools) > 0:
            payload["tools"] = tools
            payload["tool_choice"] = "auto"
        
        # Print payload for debugging
        # print("Payload:", json.dumps(payload, indent=2))
        response = requests.post(self.config.llm_endpoint, json=payload, headers=headers)
        try:
            response.raise_for_status()
        except Exception as e:
            print("❌ LLM API error:", response.status_code, response.text)
            raise
        return response.json()
        
def load_config() -> Configuration:
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise ValueError("LLM_API_KEY not found in environment variables")
    return Configuration(llm_api_key=api_key)

# Update your main() function's chat loop to this:
async def main():
    print("🤖 MCP Chatbot starting up...")
    config = load_config()
    print(f"✅ Config loaded, using model: {config.llm_model}")
    
    # Load and start MCP servers
    with open(config.servers_config_path) as f:
        servers_config = json.load(f)
    
    server_manager = MCPServerManager()
    await server_manager.start_servers(servers_config)
    
    # Initialize LLM client
    llm_client = LLMClient(config)
    
    print(f"\n🎉 All servers ready! Found {len(server_manager.all_tools)} total tools")
    print("Chat with your AI assistant (type 'quit' to exit):\n")
    
    # Conversation history
    messages = [{
        "role": "system", 
        "content": f"You are a helpful AI assistant with access to these tools: {', '.join(tool['name'] for tool in server_manager.all_tools)}"
    }]
    
    # Chat loop
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit']:
            break
            
        messages.append({"role": "user", "content": user_input})
        
        # Format tools for OpenAI API
        openai_tools = []
        for tool in server_manager.all_tools:
            openai_tools.append({
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": tool.get("inputSchema", {})
                }
            })
        
        try:
            response = llm_client.chat_completion(messages, openai_tools)
            assistant_message = response["choices"][0]["message"]
            
            # Check if the assistant wants to call tools
            if assistant_message.get("tool_calls"):
                print("🔧 Assistant is using tools...")
                
                # Execute each tool call
                for tool_call in assistant_message["tool_calls"]:
                    tool_name = tool_call["function"]["name"]
                    arguments = json.loads(tool_call["function"]["arguments"])
                    
                    print(f"   Calling {tool_name} with {arguments}")
                    result = await server_manager.execute_tool(tool_name, arguments)
                    print(f"   Result: {result}")
                    
                    # Add tool result to conversation
                    messages.append(assistant_message)
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": str(result)
                    })
                
                # Get final response after tool execution
                final_response = llm_client.chat_completion(messages)
                final_message = final_response["choices"][0]["message"]
                print(f"Assistant: {final_message.get('content', 'Done!')}")
                messages.append(final_message)
            else:
                # Regular response without tools
                print(f"Assistant: {assistant_message.get('content', 'Thinking...')}")
                messages.append(assistant_message)
                
        except Exception as e:
            print(f"❌ Error: {e}")

if __name__ == "__main__":
    asyncio.run(main())