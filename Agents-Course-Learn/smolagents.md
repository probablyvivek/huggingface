 
 ## [Building effective agents - By Anthropic](https://www.anthropic.com/engineering/building-effective-agents)

### The Big Picture
There are two main types:

**Workflows**: Like following a recipe step-by-step (very predictable)

**Agents**: Like having a smart friend who figures out their own way to solve problems (more flexible but unpredictable)

**Key Advice**: Keep It Simple
Don't build a complex agent when a simple solution works! It's like using a calculator instead of building a robot to do basic math.

Common Patterns They See Working:
* Prompt Chaining: Break big tasks into smaller steps

Like writing an essay: first make an outline, then write each section, then edit

* Routing: Send different types of questions to different specialists

Like how a hospital sends you to different doctors based on your symptoms

* Parallelization: Do multiple things at once

Like having several friends each research different parts of a group project

* Orchestrator-Workers: One "boss" AI delegates tasks to "worker" AIs

Like a project manager assigning different team members different tasks

* Evaluator-Optimizer: One AI does the work, another AI critiques it, repeat

Like having a friend review your essay and suggest improvements

When to Use Agents

When the problem is too complex/unpredictable for step-by-step workflows
When you need flexibility and can tolerate some mistakes
When simpler solutions aren't good enough

Bottom Line
Start simple, test everything, and only add complexity when you actually need it. Most problems don't need fancy AI agents - sometimes a well-crafted prompt is enough!
Think of it like choosing between a Swiss Army knife vs. a toolbox - agents are powerful but more complex and expensive to use.
 
 
 
 
 
 
 Appendix:
 1. [Building Effective Agents By Anthropic](https://www.anthropic.com/engineering/building-effective-agents)
 2. [Basic Workflows By Anthropic](https://github.com/anthropics/anthropic-cookbook/blob/main/patterns/agents/basic_workflows.ipynb)
 