import asyncio
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import requests
from dotenv import load_dotenv
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

load_dotenv()

@dataclass
class Configuration:
    # Ollama config  
    ollama_endpoint: str = "http://localhost:11434/api/generate"
    ollama_model: str = "llama3.2:latest"
    max_retries: int = 3
    timeout: int = 30

class MCPServerManager:
    def __init__(self):
        self.servers = {}
        self.all_tools = []

    async def start_servers(self, servers_config):
        """Start all MCP servers and keep them running"""
        for name, config in servers_config["mcpServers"].items():
            if config.get("disabled", False):
                print(f"⏭️ Skipping disabled server: {name}")
                continue
                
            print(f"🔌 Starting MCP server: {name}")
            
            # Add environment variable to suppress debug output
            env = config.get("env", {}).copy()
            env.update({
                "MCP_LOG_LEVEL": "error",
                "LOG_LEVEL": "error",
                "DISABLE_CONSOLE_OUTPUT": "true"
            })
            
            server_params = StdioServerParameters(
                command=config["command"],
                args=config["args"],
                env=env
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
                    print(f"   📦 {tool['name']}: {tool.get('description', 'No description')[:60]}...")
                    
            except Exception as e:
                print(f"❌ Failed to start {name}: {e}")

    async def execute_tool(self, tool_name: str, arguments: dict):
        """Execute a tool on the appropriate MCP server"""
        for server_name, server_data in self.servers.items():
            for tool in server_data['tools']:
                if tool['name'] == tool_name:
                    try:
                        session = server_data['session']
                        print(f"🔧 Executing {tool_name} on {server_name}...")
                        result = await session.call_tool(tool_name, arguments)
                        
                        # Format result nicely
                        if hasattr(result, 'content') and result.content:
                            if len(result.content) == 1:
                                return result.content[0].text if hasattr(result.content[0], 'text') else str(result.content[0])
                            else:
                                return '\n'.join(str(item) for item in result.content)
                        return str(result)
                        
                    except Exception as e:
                        error_msg = f"Error executing {tool_name}: {str(e)}"
                        print(f"❌ {error_msg}")
                        return error_msg
        return f"Tool {tool_name} not found"

    def get_tool_by_name(self, tool_name: str):
        """Get tool schema by name"""
        for tool in self.all_tools:
            if tool['name'] == tool_name:
                return tool
        return None

def load_config() -> Configuration:
    """Load configuration from environment"""
    return Configuration(
        ollama_endpoint=os.getenv("OLLAMA_ENDPOINT", "http://localhost:11434/api/generate"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen2.5"),  # Default to your current model
        max_retries=int(os.getenv("MAX_RETRIES", "3")),
        timeout=int(os.getenv("TIMEOUT", "30"))
    )

def load_servers_config(config_path: str = "mcp_config.json") -> Dict:
    """Load MCP servers configuration from JSON file"""
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        print(f"📄 Loaded server config from {config_path}")
        return config
    except FileNotFoundError:
        print(f"❌ Config file {config_path} not found")
        print("Creating a basic config file...")
        
        # Create a basic config if file doesn't exist
        basic_config = {
            "mcpServers": {
                "kite": {
                    "command": "npx",
                    "args": ["mcp-remote", "https://mcp.kite.trade/sse"],
                    "env": {"NODE_TLS_REJECT_UNAUTHORIZED": "0"},
                    "disabled": False,
                    "autoApprove": ["get_profile", "login", "get_holdings"]
                }
            }
        }
        
        with open(config_path, 'w') as f:
            json.dump(basic_config, f, indent=2)
        
        return basic_config
    except json.JSONDecodeError as e:
        print(f"❌ Invalid JSON in {config_path}: {e}")
        raise

def generate_system_prompt(server_manager) -> str:
    """Generate completely dynamic system prompt based on available servers and tools"""
    if not server_manager.all_tools:
        return "You are a helpful AI assistant. Chat normally with users."
    
    # Build server descriptions dynamically
    prompt_parts = ["You are a helpful AI assistant with access to multiple tool servers:"]
    
    server_info = []
    for server_name, server_data in server_manager.servers.items():
        tool_count = len(server_data['tools'])
        server_info.append(f"- {server_name}: {tool_count} tools available")
    
    prompt_parts.extend(server_info)
    prompt_parts.append(f"\nTotal: {len(server_manager.all_tools)} tools across {len(server_manager.servers)} servers")
    
    # Add usage guidelines
    prompt_parts.extend([
        "\nUse tools when users ask specific questions that require external data or actions.",
        "For general chat, respond normally without tools.",
        "Always be helpful and use the most appropriate tool for the user's request."
    ])
    
    return "\n".join(prompt_parts)

def generate_tool_keywords(server_manager) -> List[str]:
    """Generate dynamic keywords from actual tool names and server names"""
    keywords = set()
    
    # Add all server names as keywords
    for server_name in server_manager.servers.keys():
        keywords.add(server_name.lower())
        # Split compound server names (e.g., "n8n-mcp" -> ["n8n", "mcp"])
        keywords.update(part for part in server_name.lower().replace('-', ' ').replace('_', ' ').split())
    
    # Extract keywords from actual tool names
    for tool in server_manager.all_tools:
        tool_name = tool['name'].lower()
        
        # Add action words from tool names
        action_words = ['get', 'set', 'list', 'create', 'update', 'delete', 'search', 'find', 
                       'fetch', 'send', 'post', 'put', 'execute', 'run', 'start', 'stop',
                       'add', 'remove', 'modify', 'edit', 'check', 'validate', 'test']
        
        for action in action_words:
            if action in tool_name:
                keywords.add(action)
        
        # Add significant words from tool names (longer than 3 characters)
        words = tool_name.replace('_', ' ').replace('-', ' ').split()
        for word in words:
            if len(word) > 3 and word not in action_words:
                keywords.add(word)
    
    # Add universal helper keywords
    keywords.update(['help', 'show', 'tell', 'can you', 'help me', 'tool', 'tools', 'available'])
    
    return list(keywords)

# Complete replacement for the OllamaClient class in your mcp_chat_core.py

class OllamaClient:
    def __init__(self, config: Configuration):
        self.config = config

    def parse_tool_calls(self, content: str) -> List[Dict]:
        """Enhanced tool call parser that handles various formats"""
        tool_calls = []
        
        # Method 1: Look for JSON format tool calls
        json_patterns = [
            r'\{\s*"tool_call"\s*:\s*\{[^}]+\}\s*\}',
            r'\{\s*"name"\s*:\s*"[^"]+"\s*,\s*"arguments"\s*:\s*\{[^}]*\}\s*\}',
        ]
        
        for pattern in json_patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            for match in matches:
                try:
                    clean_match = match.replace('$n', '\n').strip()
                    
                    # Handle both formats
                    if '"tool_call":' in clean_match:
                        tool_request = json.loads(clean_match)
                        if 'tool_call' in tool_request:
                            tool_calls.append(tool_request['tool_call'])
                    else:
                        # Direct format
                        tool_calls.append(json.loads(clean_match))
                        
                except json.JSONDecodeError:
                    continue
        
        # Method 2: Smart text parsing for any tool mention
        if not tool_calls:
            tool_calls = self._parse_tool_from_text(content)
        
        return tool_calls

    def _parse_tool_from_text(self, content: str) -> List[Dict]:
        """Parse tool calls from natural text"""
        content_lower = content.lower()
        
        # Common tool patterns and their mappings
        tool_patterns = {
            # Kite patterns
            r'(?:get|show|fetch).*(?:holdings?|portfolio)': {'name': 'kite:get_holdings', 'args': {}},
            r'(?:get|show|fetch).*(?:profile|account)': {'name': 'kite:get_profile', 'args': {}},
            r'(?:get|show|fetch).*(?:positions?)': {'name': 'kite:get_positions', 'args': {}},
            r'(?:get|show|list).*(?:orders?)': {'name': 'kite:get_orders', 'args': {}},
            r'(?:get|show).*(?:margins?)': {'name': 'kite:get_margins', 'args': {}},
            
            # Notes patterns
            r'(?:list|show|get).*notes?': {'name': 'Notes (AppleScript):list_notes', 'args': {}},
            r'(?:create|add|new).*note': {'name': 'Notes (AppleScript):add_note', 'args': {}},
            
            # Chrome patterns  
            r'(?:open|go to|visit).*(?:url|website|page)': {'name': 'Chrome (AppleScript):open_url', 'args': {}},
            r'(?:list|show).*tabs?': {'name': 'Chrome (AppleScript):list_tabs', 'args': {}},
            r'(?:current|active).*tab': {'name': 'Chrome (AppleScript):get_current_tab', 'args': {}},
            
            # MongoDB patterns
            r'(?:list|show).*(?:databases?|collections?)': {'name': 'MongoDB:list-databases', 'args': {}},
            r'(?:find|search|query).*(?:mongo|database)': {'name': 'MongoDB:find', 'args': {}},
            
            # n8n patterns
            r'(?:list|show).*(?:workflows?|nodes?)': {'name': 'n8n-mcp:list_nodes', 'args': {}},
            r'(?:search|find).*(?:workflow|node)': {'name': 'n8n-mcp:search_nodes', 'args': {}},
            r'(?:create|make).*workflow': {'name': 'n8n-mcp:n8n_create_workflow', 'args': {}},
            
            # File system patterns
            r'(?:list|show).*(?:files?|directory|folder)': {'name': 'Filesystem:list_directory', 'args': {}},
            r'(?:read|show|cat).*file': {'name': 'Filesystem:read_file', 'args': {}},
            r'(?:search|find).*files?': {'name': 'Filesystem:search_files', 'args': {}},
        }
        
        for pattern, tool_info in tool_patterns.items():
            if re.search(pattern, content_lower):
                return [{'name': tool_info['name'], 'arguments': tool_info['args']}]
        
        return []

    def chat_completion(self, messages: List[Dict], tools: Optional[List[Dict]] = None, server_manager=None):
        """Enhanced chat completion with better tool calling"""
        headers = {"Content-Type": "application/json"}
        
        # Build a more effective prompt
        prompt_parts = []
        
        # Check if this looks like a tool request
        last_user_msg = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                last_user_msg = msg["content"].lower()
                break
        
        # Tool usage indicators
        tool_indicators = [
            'get', 'show', 'list', 'fetch', 'find', 'search', 'create', 'open', 'run',
            'holdings', 'profile', 'positions', 'orders', 'notes', 'files', 'tabs',
            'workflow', 'database', 'tools', 'my', 'current'
        ]
        
        seems_like_tool_request = any(indicator in last_user_msg for indicator in tool_indicators)
        
        # Add tool instructions if this looks like a tool request
        if seems_like_tool_request and tools and len(tools) > 0:
            prompt_parts.append("TOOL CALLING MODE ACTIVATED")
            prompt_parts.append("Available tools:")
            
            # Show relevant tools based on the request
            relevant_tools = self._get_relevant_tools(last_user_msg, tools)
            for tool in relevant_tools[:10]:  # Show top 10 relevant tools
                func = tool.get("function", {})
                name = func.get('name', 'unknown')
                desc = func.get('description', 'No description')
                prompt_parts.append(f"- {name}: {desc}")
            
            prompt_parts.append("\nTo use a tool, respond with ONLY this JSON format:")
            prompt_parts.append('{"tool_call": {"name": "exact_tool_name", "arguments": {}}}')
            prompt_parts.append("Replace 'exact_tool_name' with the actual tool name from the list above.")
            prompt_parts.append("Do NOT add any other text. Just the JSON.\n")
        
        # Add conversation history
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"Human: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")
        
        full_prompt = "\n\n".join(prompt_parts) + "\n\nAssistant:"
        
        payload = {
            "model": self.config.ollama_model,
            "prompt": full_prompt,
            "stream": False,
            "options": {
                "temperature": 0.1 if seems_like_tool_request else 0.3,
                "top_p": 0.9,
                "stop": ["\nHuman:", "\nUser:"],
                "num_predict": 200 if seems_like_tool_request else 1000,
            }
        }
        
        try:
            response = requests.post(
                self.config.ollama_endpoint,
                json=payload,
                headers=headers,
                timeout=self.config.timeout
            )
            response.raise_for_status()
            result = response.json()
            
            return {
                "choices": [{
                    "message": {
                        "role": "assistant",
                        "content": result.get("response", "").strip()
                    }
                }]
            }
        except requests.exceptions.Timeout:
            raise Exception("Request to Ollama timed out")
        except requests.exceptions.RequestException as e:
            raise Exception(f"Ollama API error: {e}")

    def _get_relevant_tools(self, query: str, tools: List[Dict]) -> List[Dict]:
        """Get tools most relevant to the user's query"""
        query_lower = query.lower()
        relevant = []
        
        # Score tools based on relevance
        for tool in tools:
            func = tool.get("function", {})
            name = func.get('name', '').lower()
            desc = func.get('description', '').lower()
            
            score = 0
            # Exact name matches
            if any(word in name for word in query_lower.split()):
                score += 10
            # Description matches
            if any(word in desc for word in query_lower.split() if len(word) > 3):
                score += 5
            # Category matches
            if 'kite' in query_lower and 'kite' in name:
                score += 15
            if 'note' in query_lower and 'notes' in name:
                score += 15
            if 'file' in query_lower and ('filesystem' in name or 'file' in name):
                score += 15
            if 'mongo' in query_lower and 'mongodb' in name:
                score += 15
            if 'chrome' in query_lower and 'chrome' in name:
                score += 15
            if 'workflow' in query_lower and 'n8n' in name:
                score += 15
            
            if score > 0:
                relevant.append((tool, score))
        
        # Sort by relevance score
        relevant.sort(key=lambda x: x[1], reverse=True)
        return [tool for tool, score in relevant]

    def change_model(self, model: str):
        """Change the Ollama model"""
        self.config.ollama_model = model
        print(f"🔄 Switched to model: {model}")


# Also add this helper function to better handle conversation turns
async def handle_conversation_turn(ollama_client, server_manager, messages, openai_tools):
    """Enhanced conversation handler with better tool detection"""
    try:
        # Get initial response
        response = ollama_client.chat_completion(messages, openai_tools, server_manager)
        assistant_message = response["choices"][0]["message"]
        content = assistant_message.get("content", "").strip()
        
        if not content:
            return "I'm not sure how to respond to that."
        
        print(f"🤖 Model response: {content[:100]}...")
        
        # Check for tool calls
        tool_calls = ollama_client.parse_tool_calls(content)
        
        if tool_calls:
            print(f"🔧 Detected {len(tool_calls)} tool call(s)")
            
            # Execute the first tool call
            tool_call = tool_calls[0]
            tool_name = tool_call.get("name")
            arguments = tool_call.get("arguments", {})
            
            if not tool_name:
                return "I detected a tool request but couldn't determine which tool to use."
            
            print(f"   📡 Executing {tool_name} with args: {arguments}")
            
            try:
                result = await server_manager.execute_tool(tool_name, arguments)
                
                # Add tool interaction to conversation
                messages.append({"role": "assistant", "content": f"I'm using the {tool_name} tool..."})
                messages.append({"role": "user", "content": f"Tool {tool_name} returned: {result}"})
                
                # Get final response based on tool result
                final_response = ollama_client.chat_completion(messages, [], server_manager)
                final_content = final_response["choices"][0]["message"].get("content", "")
                
                messages.append({"role": "assistant", "content": final_content})
                return final_content
                
            except Exception as e:
                error_msg = f"Error executing {tool_name}: {str(e)}"
                print(f"❌ {error_msg}")
                return f"I tried to use the {tool_name} tool but encountered an error: {str(e)}"
        else:
            # Check if this might have been a tool request that failed to parse
            last_user_msg = ""
            for msg in reversed(messages):
                if msg["role"] == "user":
                    last_user_msg = msg["content"].lower()
                    break
            
            tool_indicators = ['get', 'show', 'list', 'fetch', 'holdings', 'profile', 'notes', 'tools']
            if any(indicator in last_user_msg for indicator in tool_indicators):
                # Suggest available tools
                available_tools = [tool['function']['name'] for tool in openai_tools[:5]]
                tools_list = '\n'.join(f"- {tool}" for tool in available_tools)
                
                enhanced_response = f"{content}\n\nI noticed you might be looking for tool functionality. Here are some available tools:\n{tools_list}\n\nTry being more specific, like 'get my holdings' or 'show my profile'."
                messages.append({"role": "assistant", "content": enhanced_response})
                return enhanced_response
            
            # Regular conversation
            messages.append(assistant_message)
            return content
            
    except Exception as e:
        error_msg = f"❌ Error during conversation: {e}"
        print(error_msg)
        return error_msg

async def main():
    print("🤖 Advanced MCP + Ollama Chatbot")
    print("=" * 40)
    
    config = load_config()
    print(f"✅ Using Ollama model: {config.ollama_model}")
    print(f"🔗 Endpoint: {config.ollama_endpoint}")
    
    # Load server configuration from JSON file
    servers_config = load_servers_config("mcp_config.json")
    
    # Show which servers will be loaded
    enabled_servers = [name for name, conf in servers_config["mcpServers"].items() 
                      if not conf.get("disabled", False)]
    disabled_servers = [name for name, conf in servers_config["mcpServers"].items() 
                       if conf.get("disabled", False)]
    
    if enabled_servers:
        print(f"📡 Will load servers: {', '.join(enabled_servers)}")
    if disabled_servers:
        print(f"⏸️ Disabled servers: {', '.join(disabled_servers)}")
    
    # Initialize components
    server_manager = MCPServerManager()
    await server_manager.start_servers(servers_config)
    ollama_client = OllamaClient(config)
    
    if not server_manager.all_tools:
        print("⚠️ No tools available. Running in chat-only mode.")
    else:
        print(f"\n🎉 Ready! {len(server_manager.all_tools)} tools available")
    
    print("\nCommands:")
    print("  'quit' or 'exit' - Exit the program")
    print("  'model <name>' - Switch Ollama model")
    print("  'tools' - List available tools")
    print("  'clear' - Clear conversation history")
    print("-" * 40)
    
    # Conversation state - now completely dynamic!
    messages = [{
        "role": "system",
        "content": generate_system_prompt(server_manager)
    }]
    
    # Main chat loop
    while True:
        try:
            user_input = input("\n💬 You: ").strip()
            
            if not user_input:
                continue
                
            if user_input.lower() in ['quit', 'exit', 'bye']:
                print("👋 Goodbye!")
                break
            
            # Handle special commands
            if user_input.lower().startswith('model '):
                model_name = user_input[6:].strip()
                ollama_client.change_model(model_name)
                continue
            elif user_input.lower() == 'tools':
                if server_manager.all_tools:
                    print("\n📋 Available Tools:")
                    for tool in server_manager.all_tools:
                        print(f"  • {tool['name']}: {tool.get('description', 'No description')}")
                else:
                    print("No tools available")
                continue
            elif user_input.lower() == 'clear':
                messages = messages[:1]  # Keep system message
                print("🧹 Conversation cleared")
                continue
            
            # Add user message
            messages.append({"role": "user", "content": user_input})
            
            # Format tools for the model
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
            
            # Handle the conversation turn
            response = await handle_conversation_turn(
                ollama_client, server_manager, messages, openai_tools
            )
            
            print(f"\n🤖 Assistant: {response}")
            
        except KeyboardInterrupt:
            print("\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Unexpected error: {e}")

if __name__ == "__main__":
    asyncio.run(main())