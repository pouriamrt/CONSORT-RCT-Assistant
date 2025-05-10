import operator
import json
from langchain_core.tools import tool
from langchain.agents import create_openai_tools_agent
from langchain import hub
from typing import TypedDict, Annotated, List, Union
from langchain_core.agents import AgentAction, AgentFinish
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_community.utilities import SQLDatabase
from langchain.chains import create_sql_query_chain
from langchain_core.prompts import PromptTemplate
from langchain_community.tools.sql_database.tool import QuerySQLDataBaseTool
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from operator import itemgetter


def graph_construct(conversational_rag_chain, model="gpt-4.1-mini"):
    llm = ChatOpenAI(model=model, temperature=0)
    
    @tool("rag")
    def rag_tool(query: str, session_id: str):
        """Returns the answer to the question with searching for information from the vectorstore."""
        
        out = conversational_rag_chain.invoke({"input": query}, 
                                            config={
                                                "configurable": {"session_id": session_id}
                                            },
        )
        
        return out


    @tool("sql")
    def sql_tool(query: str, session_id: str):
        """Returns the answer to the questions which is transformable to a sql query."""
        
        db = SQLDatabase.from_uri("postgresql+psycopg://bsituser:M4pbcMDsbm30zDV6@awseb-e-mmtzduxdgy-stack-awsebrdsdatabase-a1ggrejgeign.cp5mioiwgdbp.ca-central-1.rds.amazonaws.com:5432/user_db")
        
        execute_query = QuerySQLDataBaseTool(db=db)
        
        seed_prompt = """
        Given an input question, create a syntactically correct POSTGRESQL query to run.

        Question: "Question here"
        SQLQuery: "SQL Query to run"

        """
        
        restrictions = """
        Format all numeric response ###,###,###,###.
        Only return relevant columns to the question.
        If a table or column does not exist, return table or column could not be found.
        Only type the SQL query and do not write ```sql or anything else before the SQL query statement.
        Question: {input}
        Top_k: {top_k}
        Table Info: {table_info}
        """
        
        prompt = seed_prompt + restrictions
        PROMPT = PromptTemplate(
            input_variables=["input", "top_k", "table_info"], template=prompt
        )
        
        write_query = create_sql_query_chain(llm, db, prompt=PROMPT)
        
        query = " ".join(query.split("session_id")[:-1])[:-4]
        
        answer_prompt = PromptTemplate.from_template(
        """Given the following user question, corresponding SQL query, 
        and SQL result, answer the user question.

        Question: {question}
        SQL Query: {query}
        SQL Result: {result}
        Answer: """
        )
        
        answer = answer_prompt | llm | StrOutputParser()
        
        chain = (
            RunnablePassthrough.assign(query=write_query).assign(
                result=itemgetter("query") | execute_query
            )
            | answer
        )
        
        output = chain.invoke({"question": query})
        
        return output


    tools = [rag_tool, sql_tool]

    prompt = hub.pull("hwchase17/openai-functions-agent")
    prompt.messages[0].prompt.template = "You are a helpful assistant. Decide carefully which tool to use based on the tools description and the question."
    
    query_agent_runnable = create_openai_tools_agent(
        llm=llm,
        tools=tools,
        prompt=prompt
    )


    def run_query_agent(state: list):
        print("> run_query_agent")
        agent_out = query_agent_runnable.invoke(state)
        return {"agent_out": agent_out}

    def execute_search(state: list):
        print("> execute_search")
        action = state["agent_out"]
        tool_call = action[-1].message_log[-1].additional_kwargs["tool_calls"][-1]
        
        out = rag_tool.invoke(
            json.loads(tool_call["function"]["arguments"]), 
        )
        
        #return {"intermediate_steps": [{"search": str(out)}]}
        return {"agent_out": out}

    def router(state: list):
        print("> router")
        if isinstance(state["agent_out"], list):
            return state["agent_out"][-1].tool
        else:
            return "error"

    def execute_sql(state: list):
        print("> execute_sql")
        action = state["agent_out"]
        tool_call = action[-1].message_log[-1].additional_kwargs["tool_calls"][-1]
        d = json.loads(tool_call["function"]["arguments"])
        d['query'] = state["input"]
        
        out = sql_tool.invoke(
            d, 
        )
        
        return {"agent_out": out}

    # we use the same forced final_answer LLM call to handle incorrectly formatted
    # output from our query_agent
    def handle_error(state: list):
        print("> handle_error")
        query = state["input"]
        prompt = f"""You are a helpful assistant, answer the user's question. 
        Ignore anything about the session id if there was any in the query.

        QUESTION: {query}
        """
        out = llm.invoke(prompt)
        return {"agent_out": out.content}


    class AgentState(TypedDict):
        input: str
        agent_out: Union[AgentAction, AgentFinish, None]
        intermediate_steps: Annotated[list[tuple[AgentAction, str]], operator.add]
        

    graph = StateGraph(AgentState)

    graph.add_node("query_agent", run_query_agent)
    graph.add_node("rag", execute_search)
    graph.add_node("sql", execute_sql)
    graph.add_node("error", handle_error)

    graph.set_entry_point("query_agent")

    # conditional edges are controlled by our router
    graph.add_conditional_edges(
        "query_agent",  # where in graph to start
        router,  # function to determine which node is called
        {
            "rag": "rag",
            "sql": "sql",
            "error": "error",
        }
    )
    
    graph.add_edge("rag", END)
    graph.add_edge("sql", END)
    graph.add_edge("error", END)


    # memory = SqliteSaver.from_conn_string(":memory:")
    # memory = MemorySaver()

    # runnable = graph.compile(checkpointer=memory)
    runnable = graph.compile()
    
    return runnable

