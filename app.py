from __future__ import annotations
import os, warnings, logging, asyncio
from typing import Dict, Optional, List, Tuple
import json
import chainlit as cl
from chainlit.types import ThreadDict
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import (
    create_history_aware_retriever,
    create_retrieval_chain,
)
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.retrievers import ContextualCompressionRetriever, SelfQueryRetriever
from langchain.chains.query_constructor.base import AttributeInfo
from langchain.retrievers.document_compressors import LLMChainFilter
from langchain_community.vectorstores import PGVector
from langchain_core.runnables import RunnableConfig
from langchain.output_parsers.json import SimpleJsonOutputParser
from langsmith import traceable
from aiocache import cached

from agent_graph import graph_construct

logging.basicConfig(level=logging.WARNING)
warnings.filterwarnings("ignore")

# ----------  env & singletons ---------- #
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
PG_CONN = os.getenv("PG_VECTOR_CONN")
COLLECTION = os.getenv("PG_VECTOR_COLLECTION", "state_of_union_vectors")

LLM = ChatOpenAI(
    model=os.getenv("OPENAI_MODEL", "gpt-4.1-mini"),
    temperature=0,
    streaming=True,
)
UTILITY_LLM = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
EMB = OpenAIEmbeddings()

VECTORSTORE = PGVector(
    embedding_function=EMB,
    collection_name=COLLECTION,
    connection_string=PG_CONN,
)

# ----------  helper prompts ---------- #
CONTEXTUALIZE_SYS = """Given a chat history and the latest user question
(which may reference prior context) rewrite it as a *stand‑alone* question.
Return only the rewritten question."""

CONTEXTUALIZE_PROMPT = ChatPromptTemplate.from_messages(
    [("system", CONTEXTUALIZE_SYS), MessagesPlaceholder("chat_history"), ("human", "{input}")]
)

QA_SYS = """You are an assistant for medical research questions.
Answer from the supplied context. You can deduce the answer from the context as well. If unsure say
"I don't know from the given documents.".

{context}"""
QA_PROMPT = ChatPromptTemplate.from_messages(
    [("system", QA_SYS), MessagesPlaceholder("chat_history"), ("human", "{input}")]
)

# ----------  document grader ---------- #
GRADE_PROMPT = ChatPromptTemplate.from_template(
    """You are grading how relevant a document is to a question.
Score 0 (irrelevant) to 5 (highly relevant).

QUESTION: {question}
DOCUMENT: {doc}

Score only a single integer."""
)

@traceable(name="GradeDocs")
@cached(ttl=300)
async def grade_documents(question: str, docs: List[str]) -> List[Tuple[str, int]]:
    tasks = [LLM.ainvoke(GRADE_PROMPT.format(question=question, doc=d.page_content)) for d in docs]
    scores = [int((await t).content.strip()[:1]) for t in asyncio.as_completed(tasks)]
    return list(zip(docs, scores))

def filter_by_grade(docs_scored, threshold: int = 2):
    return [d for d, s in docs_scored if s >= threshold]

# ----------  hallucination checker ---------- #
json_parser = SimpleJsonOutputParser()

CHECK_PROMPT = ChatPromptTemplate.from_template(
    """You are checking if the ANSWER is fully supported by CONTEXT.
Respond **only** with valid JSON containing these keys:

  "score"       – float in [0,1] (1 = perfectly grounded)
  "consistent"  – 1 or 0
  "explanation" – short string

Return nothing else.

<context>
{context}
</context>

<answer>
{answer}
</answer>"""
)

async def hallucination_guard(answer: str, ctx_docs):
    ctx_text = "\n\n".join(d.page_content for d in ctx_docs)[:6000]

    llm_json_mode = {"response_format": {"type": "json_object"}}
    raw = await UTILITY_LLM.ainvoke(
        CHECK_PROMPT.format(context=ctx_text, answer=answer),
        **llm_json_mode,
    )
    
    try:
        js = await json_parser.aparse(raw.content)   # reliable JSON parse
        score = float(js.get("score", 0))
    except Exception:
        # Guardrail: treat any parsing failure as low support
        score, js = 0.0, {"consistent": 0, "explanation": str(raw), "score": 0}

    return score, js

# ----------  build the RAG runnable ---------- #
def build_runnable():
    # 1 . retriever with self‑query
    metadata_info = [
        AttributeInfo(
            name="Publication Year",
            description="The year that the paper was published.",
            type="integer",
        ),
        AttributeInfo(
            name="Date Added",
            description="The year that the paper was added to the collection.",
            type="integer",
        ),
        AttributeInfo(
            name="Author",
            description="Authors of the paper, it could be couple of people.",
            type="string",
        ),
        AttributeInfo(
            name="Title", 
            description="Title of the paper that the paper is about.", 
            type="string",
        ),
        AttributeInfo(
            name="Cleaned_Abs", 
            description="Abstract of the paper that the paper is about.", 
            type="string",
        ),
        AttributeInfo(
            name="Population", 
            description="Whether the Population is mentioned in the paper. P flag in PICOS is the string True or False.", 
            type="string",
        ),
        AttributeInfo(
            name="Intervention", 
            description="Whether the Intervention is mentioned in the paper. I flag in PICOS is the string True or False.", 
            type="string",
        ),
        AttributeInfo(
            name="Comparator", 
            description="Whether the Comparator is mentioned in the paper. C flag in PICOS is the string True or False.", 
            type="string",
        ),
        AttributeInfo(
            name="Outcome", 
            description="Whether the Outcome is mentioned in the paper. O flag in PICOS is the string True or False.", 
            type="string",
        ),
        AttributeInfo(
            name="Study Design", 
            description="Whether the Study Design is mentioned in the paper. S flag in PICOS is the string True or False.", 
            type="string",
        ),
        AttributeInfo(
            name="Qualification", 
            description="Whether the paper is PICOS compliant or not. It is the string Qualified or Not Qualified.", 
            type="string",
        ),
    ]
    base_retriever = SelfQueryRetriever.from_llm(
        LLM,
        VECTORSTORE,
        "medical research papers",
        metadata_info,
        search_kwargs={"k": 8},
    )

    # 2 . LLM filter (grade ≥ 2)
    _llm_filter = LLMChainFilter.from_llm(LLM)

    compression_retriever = ContextualCompressionRetriever(
        base_compressor=_llm_filter,
        base_retriever=base_retriever,
    )

    # 3 . history‑aware query transform
    hist_retriever = create_history_aware_retriever(
        LLM, compression_retriever, CONTEXTUALIZE_PROMPT
    )

    # 4 . answer chain
    qa_chain = create_stuff_documents_chain(LLM, QA_PROMPT)

    # 5 . assemble retrieval+answer
    rag_chain = create_retrieval_chain(hist_retriever, qa_chain)

    # 6 . wrap with message history
    from langchain_core.runnables.history import RunnableWithMessageHistory

    store = {}
    def _get_hist(sid): return store.setdefault(sid, ConversationBufferMemory().chat_memory)

    rag_with_hist = RunnableWithMessageHistory(
        rag_chain,
        _get_hist,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )

    # 7 . LangGraph agent with hallucination node
    runnable_graph = graph_construct(rag_with_hist)

    async def _run(payload, config=None):
        res = await runnable_graph.ainvoke(payload, config=config)
        agent_out = res["agent_out"]
        if isinstance(agent_out, dict):
            answer   = agent_out.get("answer", "")
            ctx_docs = agent_out.get("context", [])
        else:                      # it's already a string
            answer   = str(agent_out)
            ctx_docs = []
            
        if not ctx_docs:
            return {"text": answer, "context": ctx_docs}
        
        score, report = await hallucination_guard(answer, ctx_docs)
        if score < 0.5:
            answer += (
                f"\n\n⚠️ *Low source‑support score ({score:.2f}). "
                "Consider re‑phrasing or requesting sources.*"
            )
        return {"text": answer, "context": ctx_docs}

    return _run

# ----------  Chainlit callbacks ---------- #
@cl.oauth_callback
def oauth_callback(provider_id, token, raw_user_data, default_user):
    return default_user

@cl.password_auth_callback
def auth_callback(username, password):
    return cl.User(identifier="admin", metadata={"role": "admin"}) if (username, password) == ("admin", "admin") else None

@cl.on_chat_start
async def start():
    cl.user_session.set("runner", build_runnable())
    usr = cl.user_session.get("user")
    await cl.Message(f"Hello {usr.identifier.split('@')[0]}! 👋").send()

@cl.on_message
async def handle(message: cl.Message):
    runner = cl.user_session.get("runner")
    usr = cl.user_session.get("user")

    # --- stream intermediate tokens & show sources -------------------
    answer_prefix_tokens = ["**Answer:**\n\n"]

    langchain_cb = cl.LangchainCallbackHandler(
        stream_final_answer=True,
        answer_prefix_tokens=answer_prefix_tokens,
    )

    # 1. invoke the graph with the callback handler for streaming
    res = await runner(
        {"input": f"{message.content}\n\nsession_id:{usr.identifier}"},
        config=RunnableConfig(callbacks=[langchain_cb]),
    )

    # 2. `res` is already the fully‑formed answer string we build in _run()
    #    plus any warning from the hallucination guard.
    answer, ctx_docs = res["text"], res.get("context", [])

    # 3. build a “Sources” block (simple markdown list)
    if ctx_docs:
        src_lines = []
        for i, doc in enumerate(ctx_docs, 1):
            title = (
                doc.metadata.get("title")                # prefer explicit title
                or doc.metadata.get("source")            # or a “source” field
                or doc.page_content.strip()[:120] + "…"  # fallback: snippet
            )
            src_lines.append(f"{i}. *{title}*")
        answer += "\n\n**Sources:**\n" + "\n".join(src_lines)

    await cl.Message(answer).send()
