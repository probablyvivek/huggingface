from smolagents import CodeAgent, DuckDuckGoSearchTool, InferenceClientModel
import os
from huggingface_hub import login
token = os.getenv("HF_Token")
login(token=token)

agent = CodeAgent(tools=[DuckDuckGoSearchTool()], model=InferenceClientModel())

agent.run("Search for the best music recommendations for a party at the Wayne's mansion.")
