import streamlit as st
import asyncio
import json
from mcp.client.sse import sse_client
from mcp.client.session import ClientSession

# Page Configuration
st.set_page_config(
    page_title="V-Agents MCP Tester",
    page_icon="🤖",
    layout="wide"
)

st.title("🛠️ V-Agents MCP Tool Tester")
st.markdown("""
This utility connects to your running **MCP Server** via SSE (Server-Sent Events) 
to list available agents and test tool execution.
""")

# --- SIDEBAR CONFIGURATION ---
with st.sidebar:
    st.header("🔌 Connection Settings")
    server_url = st.text_input("MCP Server URL", value="http://localhost:8252/mcp/sse")
    
    st.header("👤 Context Simulation")
    st.info("These values are passed as arguments to the tool, simulating the frontend context.")
    user_id = st.text_input("User ID", value="1490")
    firm_id = st.number_input("Firm ID", value=5, min_value=1)

# --- ASYNC HELPERS ---
async def list_available_tools(url):
    """Connects to the server and fetches the list of available tools."""
    try:
        async with sse_client(url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                tools_result = await session.list_tools()
                return tools_result.tools
    except Exception as e:
        return f"Error: {str(e)}"

async def call_agent_tool(url, tool_name, question, uid, fid):
    """Calls a specific tool with the provided arguments."""
    try:
        async with sse_client(url) as streams:
            async with ClientSession(streams[0], streams[1]) as session:
                await session.initialize()
                
                # Construct the arguments exactly as the server expects
                arguments = {
                    "question": question,
                    "user_id": uid,
                    "firm_id": fid
                }
                
                result = await session.call_tool(tool_name, arguments)
                return result
    except Exception as e:
        return f"Error: {str(e)}"

# --- MAIN INTERFACE ---

# 1. Tool Discovery Section
st.subheader("1. Tool Discovery")
col1, col2 = st.columns([1, 3])
with col1:
    if st.button("🔄 Connect & List Tools", type="primary"):
        with st.spinner("Connecting to MCP Server..."):
            tools = asyncio.run(list_available_tools(server_url))
            st.session_state['tools'] = tools

if 'tools' in st.session_state:
    tools = st.session_state['tools']
    if isinstance(tools, str) and tools.startswith("Error"):
        st.error(f"Failed to connect: {tools}")
        st.warning("Make sure your 'server.py' is running on port 8252.")
    else:
        st.success(f"Found {len(tools)} Tool(s)")
        
        # Create a cleaner list for display
        tool_options = {t.name: t for t in tools}
        selected_tool_name = st.selectbox("Select Tool to Test", list(tool_options.keys()))
        
        # Display Tool Details
        if selected_tool_name:
            tool_info = tool_options[selected_tool_name]
            with st.expander("View Tool Schema", expanded=False):
                st.json(tool_info.inputSchema)
            
            st.markdown(f"**Description:** {tool_info.description}")

            # 2. Testing Section
            st.divider()
            st.subheader(f"2. Test '{selected_tool_name}'")
            
            user_query = st.text_area("Enter your question:", value="Can you create a beginner workout plan?", height=100)
            
            if st.button("🚀 Run Tool", type="primary"):
                if not user_query:
                    st.warning("Please enter a question.")
                else:
                    with st.spinner(f"Asking {selected_tool_name}..."):
                        # Execute the tool call
                        result = asyncio.run(call_agent_tool(
                            server_url, 
                            selected_tool_name, 
                            user_query, 
                            user_id, 
                            firm_id
                        ))
                        
                        st.subheader("Results")
                        
                        if isinstance(result, str) and result.startswith("Error"):
                            st.error(result)
                        else:
                            # Parse MCP Result content
                            # MCP returns a list of content objects (text or image)
                            if hasattr(result, 'content') and isinstance(result.content, list):
                                for content in result.content:
                                    if content.type == 'text':
                                        st.markdown(content.text)
                                    elif content.type == 'image':
                                        st.image(content.data)
                            else:
                                st.json(result)

else:
    st.info("Click 'Connect & List Tools' to discover your Agents.")

# Footer
st.markdown("---")
st.caption("V-Agents MCP Debugger | Running on Streamlit")