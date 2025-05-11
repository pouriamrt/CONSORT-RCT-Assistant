from __future__ import annotations
import os, json, operator, asyncio
from typing import TypedDict, Annotated, Union, List
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain.agents import create_openai_tools_agent
from langchain_core.agents import AgentAction, AgentFinish
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langgraph.graph import StateGraph, END
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_community.tools import QuerySQLDatabaseTool
import re
from sqlalchemy.exc import SQLAlchemyError, DBAPIError

# ----------  shared singletons ---------- #
LLM = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    temperature=0,
    timeout=30,
    streaming=True,
)

SQL_DB = SQLDatabase.from_uri(os.getenv("PG_URI"))
EXECUTE_QUERY = QuerySQLDatabaseTool(db=SQL_DB)

# ----------  build SQL helper chains up‑front ---------- #
SQL_TEMPLATE = """
You are an expert data analyst writing SQL for a **PostgreSQL** database.

---
SCHEMA
{table_info}
---

**Task**

• Write the simplest SQL query that answers the user question below.  
• Do **NOT** add comments, markdown, or explanations.  
• Limit results to {top_k} rows unless the question explicitly asks for all rows.  
• Do not use `LIMIT` in your query for questions about counts.
• If the question requests a count, use `COUNT(*) AS total`.
• Pay attention to the database schema.

**Question**

{input}

**Respond with SQL only**
"""

SQL_PROMPT = PromptTemplate.from_template(SQL_TEMPLATE.strip())
WRITE_QUERY = create_sql_query_chain(LLM, SQL_DB, prompt=SQL_PROMPT)

ANSWER_PROMPT = PromptTemplate.from_template(
    """Given the user question, the SQL query, and its result, answer the question.

Question: {question}
SQL: {query}
Result: {result}
Answer:"""
)
ANSWER_CHAIN = ANSWER_PROMPT | LLM | StrOutputParser()


VALIDATION_PROMPT = PromptTemplate.from_template("""
    You are checking the answer based on the question.  
    If it does **not** fully address the user's question,
    reply exactly with the string «REVISE: write the exact original output here».
    Otherwise reply «OK».
    ---
    Question: {question}
    Answer draft: {answer}
    """)

# ----------  tools ---------- #
def make_rag_tool(chain):
    @tool("rag")
    async def rag(query: str, session_id: str):
        """Answer questions with the RAG pipeline."""
        # full output from create_retrieval_chain has BOTH keys
        res = await chain.ainvoke(
            {"input": query},
            config={"configurable": {"session_id": session_id}},
        )
        # keep answer + docs so downstream code can use them
        return {"answer": res["answer"], "context": res["context"]}
    return rag

MAX_RETRIES = 2
TIMEOUT = 15
def preprocess_sql(sql: str) -> str:
    sql = re.sub(r"```.*?```", "", sql, flags=re.S)
    sql = re.sub(r"COUNT\(\s*\)", "COUNT(*)", sql, flags=re.I)
    sql = sql.rstrip(" ;\n\t")
    return re.sub(r"\s{2,}", " ", sql).strip()

@tool("sql")
async def sql_tool(query: str, session_id: str):
    """Answer questions that can be converted to SQL."""
    plain_q = query.rsplit("session_id", 1)[0].strip()

    # 1) LLM ➜ SQL
    sql = await asyncio.wait_for(
        WRITE_QUERY.ainvoke({"question": plain_q}),
        timeout=TIMEOUT,
    )
    sql = preprocess_sql(sql)

    # 2) execute with automatic repair
    for attempt in range(MAX_RETRIES):
        try:
            result = await EXECUTE_QUERY.ainvoke({"query": sql})
            break
        except SQLAlchemyError as err:
            if attempt == MAX_RETRIES - 1:
                raise
            sql = (await LLM.ainvoke(
                f"""ONLY return a corrected SQL query—no prose.

    --- question ---
    {plain_q}

    --- bad SQL ---
    {sql}

    --- error ---
    {err}"""
            )).content.strip("` ")
            sql = preprocess_sql(sql)

    # 3) craft natural‑language answer
    return await ANSWER_CHAIN.ainvoke(
        {"question": plain_q, "query": sql, "result": result}
    )


TOOLS = []  # runnable‐specific list – we fill later

# ----------  state for LangGraph ---------- #
class AgentState(TypedDict):
    input: str
    agent_out: Union[AgentAction, AgentFinish, str, None]
    intermediate_steps: Annotated[List, operator.add]
    verdict: str
    retries: int
    revision_reason: str

def build_graph(conversational_rag_chain):
    # inject the RAG tool now that we have the chain instance
    TOOLS.clear()
    TOOLS.extend(
        [
            make_rag_tool(conversational_rag_chain),
            sql_tool,
        ]
    )

    prompt = hub.pull("hwchase17/openai-functions-agent")
    prompt.messages[0].prompt.template = (
        "You are a helpful assistant. Choose the *single best* tool "
        "based on the question."
    )

    query_agent = create_openai_tools_agent(LLM, TOOLS, prompt)

    # ---- LangGraph nodes ---- #
    async def run_query_agent(state: AgentState):
        input_text = state["input"]
        if state.get("revision_reason"):
            input_text = f"{input_text}\n\nNote for revision: {state['revision_reason']}"
        new_state = dict(state)
        new_state["input"] = input_text
        return {"agent_out": await query_agent.ainvoke(new_state)}

    async def _run_tool(tool_fn, state: AgentState):
        action = state["agent_out"]
        call = action[-1].message_log[-1].additional_kwargs["tool_calls"][-1]
        args = json.loads(call["function"]["arguments"])
        args["session_id"] = args.get("session_id", state["input"][-36:])
        result = await tool_fn.ainvoke(args)
        return {"agent_out": result}

    async def execute_rag(state):
        return await _run_tool(TOOLS[0], state)

    async def execute_sql(state):
        return await _run_tool(TOOLS[1], state)

    async def handle_error(state):
        q = state["input"]
        out = await LLM.ainvoke(
            f"You are a helpful assistant. Ignore any session ID.\n\nQUESTION: {q}"
        )
        return {"agent_out": out.content}

    def router(state):
        if isinstance(state["agent_out"], list):
            return state["agent_out"][-1].tool
        return "error"

    def _get_answer_text(agent_out) -> str:
        if isinstance(agent_out, str):
            return agent_out
        if isinstance(agent_out, dict):
            return agent_out.get("answer", str(agent_out))
        # AgentFinish → .return_values or .output
        try:
            return agent_out.return_values["output"]
        except Exception:
            return str(agent_out)
        
    def extract_revision_reason(verdict: str) -> str:
        if verdict.startswith("REVISE:"):
            return verdict[len("REVISE:"):].strip()
        return ""
    
    async def validate_answer(state: AgentState):
        answer_text = _get_answer_text(state["agent_out"])
        verdict = await (VALIDATION_PROMPT | LLM | StrOutputParser()).ainvoke(
            {"question": state["input"], "answer": answer_text}
        )
        revision_reason = extract_revision_reason(verdict)
        new_retries = state.get("retries", 0) + (1 if revision_reason else 0)
        return {
            "verdict": verdict,
            "revision_reason": revision_reason,
            "retries": new_retries,
        }

    def route_validation(state):
        if state.get("revision_reason") and state["retries"] < 3:
            return "revise"
        return "done"

    # ---- build the graph ---- #
    g = StateGraph(AgentState)
    g.add_node("query_agent", run_query_agent)
    g.add_node("rag", execute_rag)
    g.add_node("sql", execute_sql)
    g.add_node("error", handle_error)
    g.add_node("validate", validate_answer)

    g.set_entry_point("query_agent")
    g.add_conditional_edges("query_agent", router, {"rag": "rag", "sql": "sql", "error": "error"})
    g.add_edge("rag", "validate")
    g.add_edge("sql", "validate")
    g.add_conditional_edges("validate", route_validation,
                            {"revise": "query_agent", "done": END})
    g.add_edge("error", END)
    return g.compile()

# exposed function
def graph_construct(conversational_rag_chain):
    return build_graph(conversational_rag_chain)
