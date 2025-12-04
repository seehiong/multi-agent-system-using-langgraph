import os
import json
import asyncio
import sys
from typing import Annotated, Sequence, TypedDict
from langchain_core.messages import (
    HumanMessage, AIMessage, BaseMessage, ToolMessage
)

# Fix Windows event loop issue
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI
from toolbox_langchain import ToolboxClient
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages

# ----------------------------- 
# Global state for MCP tools
# ----------------------------- 
_mcp_tools = None
_toolbox_client = None
_client_lock = asyncio.Lock()

async def get_mcp_tools():
    """Lazy load MCP tools on first use with connection pooling"""
    global _mcp_tools, _toolbox_client
    
    async with _client_lock:  # Prevent concurrent initialization
        if _mcp_tools is None:
            try:
                # Move blocking ToolboxClient to separate thread
                def _create_client():
                    return ToolboxClient("http://127.0.0.1:5000")
                
                _toolbox_client = await asyncio.to_thread(_create_client)
                
                # Try aload_toolset first, fall back to load_toolset
                if hasattr(_toolbox_client, 'aload_toolset'):
                    _mcp_tools = await asyncio.wait_for(
                        _toolbox_client.aload_toolset(),
                        timeout=30
                    )
                else:
                    _mcp_tools = await asyncio.wait_for(
                        _toolbox_client.load_toolset(),
                        timeout=30
                    )
                print(f"Loaded {len(_mcp_tools)} MCP tools")
            except asyncio.TimeoutError:
                print("MCP tools loading timed out")
                _mcp_tools = []
            except Exception as e:
                print(f"Failed to load MCP tools: {e}")
                _mcp_tools = []
    
    return _mcp_tools

# ----------------------------- 
# Tavily
# ----------------------------- 
tavily = TavilySearch(max_results=5)

# ----------------------------- 
# Custom amenity search tool
# ----------------------------- 
@tool
def search_hdb_amenities(query: str) -> str:
    """Search for amenities near an HDB block or town using Tavily.
    
    Example queries:
    - "amenities near Block 123 Ang Mo Kio"
    - "schools near Bukit Panjang"
    - "supermarkets near Tampines"
    - "parks near Woodlands"
    """
    formatted_query = f"amenities around {query}, Singapore HDB, nearby facilities, schools, supermarkets, malls, transport"
    results = tavily.invoke(formatted_query)
    return results

# ----------------------------- 
# STATE
# ----------------------------- 
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]

# ----------------------------- 
# PROMPTS
# ----------------------------- 
sql_prompt = ChatPromptTemplate.from_messages([
    ("system", """SQL Agent. You answer only questions requiring HDB resale data or stats.
Use MCP SQL tools ONLY. Always return structured, correct SQL tool calls."""),
    MessagesPlaceholder("scratch_pad")
])

amenities_prompt = ChatPromptTemplate.from_messages([
    ("system", """Amenities Agent. Use search_hdb_amenities first. Use Tavily for deeper context.
You answer: amenities, schools, malls, MRT, supermarkets near a block or town."""),
    MessagesPlaceholder("scratch_pad")
])

web_prompt = ChatPromptTemplate.from_messages([
    ("system", """Web Research Agent. Use Tavily search for general knowledge or non-HDB topics."""),
    MessagesPlaceholder("scratch_pad")
])

# ----------------------------- 
# MODEL FACTORY
# ----------------------------- 
MODEL = "moonshotai/kimi-k2-thinking"

def build_model():
    return ChatOpenAI(
        api_key=os.getenv("OPENROUTER_API_KEY"),
        base_url="https://openrouter.ai/api/v1",
        model=MODEL,
    )

# ----------------------------- 
# AGENTS (built dynamically with tools)
# ----------------------------- 
async def build_agents(mcp_tools):
    """Build agents with available tools"""
    sql_agent = sql_prompt | build_model().bind_tools(mcp_tools)
    amenities_agent = amenities_prompt | build_model().bind_tools([search_hdb_amenities, tavily])
    web_agent = web_prompt | build_model().bind_tools([tavily])
    return sql_agent, amenities_agent, web_agent

# ----------------------------- 
# ROUTER NODE
# ----------------------------- 
async def router_node(state: AgentState):
    """Route to appropriate agent"""
    router_model = build_model()
    supervisor_prompt = ChatPromptTemplate.from_messages([
        ("system", """You are the Supervisor. ROUTE the user query to one agent:
- SQL_AGENT for resale/transaction/statistics queries
- AMENITIES_AGENT for amenities/schools/MRT/malls
- WEB_AGENT for general queries

Respond with ONLY one of: SQL_AGENT AMENITIES_AGENT WEB_AGENT"""),
        MessagesPlaceholder("messages")
    ])
    
    chain = supervisor_prompt | router_model
    decision = await chain.ainvoke({"messages": state["messages"]})
    route = decision.content.strip().upper()
    
    if route not in {"SQL_AGENT", "AMENITIES_AGENT", "WEB_AGENT"}:
        route = "WEB_AGENT"
    
    return {"next_agent": route}

# ----------------------------- 
# TOOL EXECUTION NODE
# ----------------------------- 
async def execute_tool_with_retry(tool, tool_name, args, max_retries=2):
    """Execute tool with exponential backoff retry"""
    for attempt in range(max_retries + 1):
        try:
            print(f"    Attempt {attempt + 1}/{max_retries + 1}...")
            
            if hasattr(tool, 'ainvoke'):
                result = await asyncio.wait_for(
                    tool.ainvoke(args),
                    timeout=20
                )
            else:
                result = await asyncio.wait_for(
                    asyncio.to_thread(tool.invoke, args),
                    timeout=20
                )
            return result
            
        except asyncio.TimeoutError:
            print(f"    Timeout on attempt {attempt + 1}")
            if attempt < max_retries:
                wait_time = 2 ** attempt
                print(f"    Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise
        except Exception as e:
            print(f"    Error: {str(e)}")
            if attempt < max_retries and "Network" in str(e):
                wait_time = 2 ** attempt
                print(f"    Retrying in {wait_time}s...")
                await asyncio.sleep(wait_time)
            else:
                raise

async def tool_node(state: AgentState):
    """Execute tool calls from agent"""
    last = state["messages"][-1]
    outputs = []
    
    mcp_tools = await get_mcp_tools()
    all_tools = mcp_tools + [tavily, search_hdb_amenities]
    tools_by_name = {t.name: t for t in all_tools}
    
    print(f"\nTOOL_NODE: Found {len(last.tool_calls)} tool call(s)")
    print(f"Available tools: {list(tools_by_name.keys())}")
    
    for tc in last.tool_calls:
        tool_name = tc["name"]
        print(f"\n  Executing: {tool_name}")
        print(f"    Args: {tc['args']}")
        
        try:
            tool = tools_by_name.get(tool_name)
            if not tool:
                result = f"Tool '{tool_name}' not found. Available: {list(tools_by_name.keys())}"
                print(f"    {result}")
            else:
                print(f"    Tool type: {type(tool)}")
                result = await execute_tool_with_retry(tool, tool_name, tc["args"])
                print(f"    Success.")
            
            if isinstance(result, str):
                content = result
            elif isinstance(result, dict):
                content = json.dumps(result)
            elif isinstance(result, list):
                content = json.dumps(result)
            else:
                content = str(result)
            
            outputs.append(ToolMessage(
                content=content,
                name=tool_name,
                tool_call_id=tc["id"],
            ))
            print(f"    ToolMessage created ({len(content)} chars)")
            
        except Exception as e:
            error_msg = f"Error executing {tool_name}: {str(e)}"
            print(f"    {error_msg}")
            import traceback
            traceback.print_exc()
            
            outputs.append(ToolMessage(
                content=error_msg,
                name=tool_name,
                tool_call_id=tc["id"],
            ))
    
    print(f"\n  Returning {len(outputs)} ToolMessage(s)")
    return {"messages": outputs}

# ----------------------------- 
# SPECIALIST NODES
# ----------------------------- 
async def run_agent(agent, state):
    response = await agent.ainvoke({"scratch_pad": state["messages"]})
    return {"messages": [response]}

async def sql_agent_node(state: AgentState):
    mcp_tools = await get_mcp_tools()
    agent = sql_prompt | build_model().bind_tools(mcp_tools)
    return await run_agent(agent, state)

async def amenities_agent_node(state: AgentState):
    agent = amenities_prompt | build_model().bind_tools([search_hdb_amenities, tavily])
    return await run_agent(agent, state)

async def web_agent_node(state: AgentState):
    agent = web_prompt | build_model().bind_tools([tavily])
    return await run_agent(agent, state)

# ----------------------------- 
# GRAPH BUILD
# ----------------------------- 
def build_graph():
    workflow = StateGraph(AgentState, async_mode=True)
    
    workflow.add_node("ROUTER", router_node)
    workflow.add_node("SQL_AGENT", sql_agent_node)
    workflow.add_node("AMENITIES_AGENT", amenities_agent_node)
    workflow.add_node("WEB_AGENT", web_agent_node)
    workflow.add_node("TOOLS", tool_node)
    
    def next_step(state: AgentState):
        last = state["messages"][-1]
        has_calls = hasattr(last, 'tool_calls') and len(last.tool_calls) > 0
        print(f"\nnext_step: Last message type={type(last).__name__}, has_tool_calls={has_calls}")
        return "TOOLS" if has_calls else END
    
    workflow.add_conditional_edges("SQL_AGENT", next_step, {"TOOLS": "TOOLS", END: END})
    workflow.add_conditional_edges("AMENITIES_AGENT", next_step, {"TOOLS": "TOOLS", END: END})
    workflow.add_conditional_edges("WEB_AGENT", next_step, {"TOOLS": "TOOLS", END: END})
    
    workflow.add_conditional_edges(
        "ROUTER",
        lambda out: out["next_agent"],
        {
            "SQL_AGENT": "SQL_AGENT",
            "AMENITIES_AGENT": "AMENITIES_AGENT",
            "WEB_AGENT": "WEB_AGENT",
        },
    )
    
    workflow.add_edge("TOOLS", "ROUTER")
    workflow.set_entry_point("ROUTER")
    
    return workflow.compile()

graph = build_graph()

# ----------------------------- 
# MAIN
# ----------------------------- 
async def main(user_query: str):
    """Run the graph with a user query"""
    print(f"\nQuery: {user_query}\n")
    
    initial_state = {"messages": [HumanMessage(content=user_query)]}
    
    async for output in graph.astream(initial_state):
        print("State:", output)
    
    return graph.get_state(initial_state)

if __name__ == "__main__":
    query = "What amenities are near Bukit Panjang?"
    asyncio.run(main(query))
