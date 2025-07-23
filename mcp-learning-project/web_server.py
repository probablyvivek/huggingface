import asyncio
import json
import os
from typing import Dict, List, Optional
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn

# Import your existing classes
from mcp_chat_core import (  # Replace with your actual filename
    MCPServerManager, 
    OllamaClient, 
    Configuration, 
    load_config, 
    load_servers_config,
    generate_system_prompt
)

# Pydantic models for API requests
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage] = []

class MCPServerConfig(BaseModel):
    command: str
    args: List[str]
    env: Dict[str, str] = {}
    disabled: bool = False
    autoApprove: List[str] = []

class UpdateServersRequest(BaseModel):
    servers: Dict[str, MCPServerConfig]

class ModelChangeRequest(BaseModel):
    model: str

# Global state - in production, you'd want proper state management
app_state = {
    "server_manager": None,
    "ollama_client": None,
    "config": None,
    "conversation_history": []
}

app = FastAPI(title="MCP Chat API")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    """Initialize MCP servers and Ollama client on startup"""
    print("🚀 Starting MCP Chat API...")
    
    # Load configuration
    config = load_config()
    servers_config = load_servers_config("mcp_config.json")
    
    # Initialize components
    server_manager = MCPServerManager()
    await server_manager.start_servers(servers_config)
    ollama_client = OllamaClient(config)
    
    # Store in global state
    app_state["server_manager"] = server_manager
    app_state["ollama_client"] = ollama_client
    app_state["config"] = config
    app_state["conversation_history"] = [{
        "role": "system",
        "content": generate_system_prompt(server_manager)
    }]
    
    print(f"✅ API ready with {len(server_manager.all_tools)} tools")

# Serve the chat.html file
@app.get("/")
async def serve_chat_ui():
    """Serve the chat.html file"""
    if os.path.exists("chat.html"):
        return FileResponse("chat.html", media_type="text/html")
    else:
        return {"error": "chat.html file not found. Please create it in the same directory as this server."}

@app.get("/api/status")
async def get_status():
    """Get current system status"""
    server_manager = app_state["server_manager"]
    config = app_state["config"]
    
    return {
        "status": "ready",
        "model": config.ollama_model if config else "unknown",
        "servers": len(server_manager.servers) if server_manager else 0,
        "tools": len(server_manager.all_tools) if server_manager else 0
    }

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Handle chat messages"""
    server_manager = app_state["server_manager"]
    ollama_client = app_state["ollama_client"]
    
    if not server_manager or not ollama_client:
        raise HTTPException(status_code=500, detail="System not initialized")
    
    try:
        # Build conversation history
        messages = app_state["conversation_history"].copy()
        
        # Add any provided history
        for msg in request.history[-10:]:  # Keep last 10 messages for context
            messages.append({"role": msg.role, "content": msg.content})
        
        # Add current user message
        messages.append({"role": "user", "content": request.message})
        
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
        
        # Get initial response
        response = ollama_client.chat_completion(messages, openai_tools, server_manager)
        assistant_message = response["choices"][0]["message"]
        content = assistant_message.get("content", "").strip()
        
        if not content:
            return {"response": "I'm not sure how to respond to that.", "used_tools": []}
        
        # Check for tool calls
        tool_calls = ollama_client.parse_tool_calls(content)
        used_tools = []
        
        if tool_calls:
            print(f"🔧 Found {len(tool_calls)} tool call(s)")
            
            # Execute the first tool call
            tool_call = tool_calls[0]
            tool_name = tool_call.get("name")
            arguments = tool_call.get("arguments", {})
            
            if tool_name:
                print(f"   📡 Calling {tool_name} with {arguments}")
                result = await server_manager.execute_tool(tool_name, arguments)
                used_tools.append({"name": tool_name, "arguments": arguments, "result": result})
                
                # Add tool interaction to conversation
                messages.append({"role": "assistant", "content": f"I'm calling {tool_name}..."})
                messages.append({"role": "user", "content": f"Tool {tool_name} returned: {result}"})
                
                # Get final response based on tool result
                final_response = ollama_client.chat_completion(messages, [], server_manager)
                final_content = final_response["choices"][0]["message"].get("content", "")
                
                return {"response": final_content, "used_tools": used_tools}
        
        return {"response": content, "used_tools": used_tools}
        
    except Exception as e:
        print(f"❌ Chat error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/servers")
async def get_servers():
    """Get current MCP server configuration"""
    try:
        servers_config = load_servers_config("mcp_config.json")
        return {"servers": servers_config["mcpServers"]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/servers")
async def update_servers(request: UpdateServersRequest):
    """Update MCP server configuration"""
    try:
        # Convert request to the format expected by load_servers_config
        servers_config = {
            "mcpServers": {
                name: config.dict() for name, config in request.servers.items()
            }
        }
        
        # Save to file
        with open("mcp_config.json", "w") as f:
            json.dump(servers_config, f, indent=2)
        
        # Restart servers (in production, you'd want more sophisticated reloading)
        server_manager = MCPServerManager()
        await server_manager.start_servers(servers_config)
        
        # Update global state
        app_state["server_manager"] = server_manager
        app_state["conversation_history"] = [{
            "role": "system", 
            "content": generate_system_prompt(server_manager)
        }]
        
        return {"status": "updated", "tools": len(server_manager.all_tools)}
        
    except Exception as e:
        print(f"❌ Server update error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/tools")
async def get_tools():
    """Get available tools"""
    server_manager = app_state["server_manager"]
    if not server_manager:
        return {"tools": []}
    
    return {"tools": server_manager.all_tools}

@app.post("/api/model")
async def change_model(request: ModelChangeRequest):
    """Change the Ollama model"""
    ollama_client = app_state["ollama_client"]
    if not ollama_client:
        raise HTTPException(status_code=500, detail="Ollama client not initialized")
    
    try:
        ollama_client.change_model(request.model)
        app_state["config"].ollama_model = request.model
        return {"status": "changed", "model": request.model}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/clear")
async def clear_conversation():
    """Clear conversation history"""
    server_manager = app_state["server_manager"]
    app_state["conversation_history"] = [{
        "role": "system",
        "content": generate_system_prompt(server_manager) if server_manager else "You are a helpful AI assistant."
    }]
    return {"status": "cleared"}

if __name__ == "__main__":
    print("🌐 Starting MCP Chat Web Server...")
    print("📱 Chat UI will be served from: chat.html")
    print("🔗 Open: http://localhost:8000")
    print("🔧 API docs at: http://localhost:8000/docs")
    
    uvicorn.run(
        "web_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )