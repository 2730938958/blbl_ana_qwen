from __future__ import annotations

import json
from typing import Any, List, Optional

from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.tools import Tool

from .local_qwen import Qwen25LLM
from .pipeline import basic_stats
from .vector_store import search_comments
from .web_search import serpapi_search


REACT_TEMPLATE = """You are an analyst assistant. You must decide whether to answer using Bilibili comments, web search, or both.

You have access to the following tools:
{tools}

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Action Input/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Constraints:
- If question is about comment-area slang/梗/what does X mean: prioritize comments_retriever.
- If question asks for lowest price / official response / news: use web_search.
- If question asks to combine both (e.g. user complaints + official response): use both.

Question: {input}
Thought: {agent_scratchpad}
"""


def build_llm() -> Qwen25LLM:
    """构建本地 Qwen LLM，不依赖任何 API Key。"""
    return Qwen25LLM(temperature=0.2, max_new_tokens=512)


def make_agent_executor(
    *,
    llm: Qwen25LLM,
    vector_store: Any,
    docs_for_stats: Optional[List[Any]] = None,
) -> AgentExecutor:
    def comments_retriever(q: str) -> str:
        hits = search_comments(vector_store, q, k=6)
        payload = []
        for doc, score in hits:
            payload.append(
                {
                    "score": float(score),
                    "text": doc.page_content,
                    "meta": doc.metadata,
                }
            )
        return json.dumps(payload, ensure_ascii=False)

    def stats_summarizer(_: str) -> str:
        if not docs_for_stats:
            return json.dumps({"error": "no_docs_for_stats"}, ensure_ascii=False)
        return json.dumps(basic_stats(docs_for_stats), ensure_ascii=False)

    def web_search_tool(q: str) -> str:
        # 这里仍然可以使用 SerpAPI；如果你不想用全网搜索，也可以在调用前判断是否启用。
        return json.dumps(serpapi_search(q, cfg=None, k=5), ensure_ascii=False)

    tools = [
        Tool(
            name="comments_retriever",
            func=comments_retriever,
            description="Retrieve semantically similar Bilibili comments for a query. Input should be the user question.",
        ),
        Tool(
            name="stats_summarizer",
            func=stats_summarizer,
            description="Get basic stats: total, hot ratio, sentiment distribution, intent distribution. Input can be empty string.",
        ),
        Tool(
            name="web_search",
            func=web_search_tool,
            description="Search the web for up-to-date info (price, official response, news). Input should be a concise search query.",
        ),
    ]

    prompt = PromptTemplate.from_template(REACT_TEMPLATE)
    agent = create_react_agent(llm, tools, prompt)
    return AgentExecutor(agent=agent, tools=tools, verbose=False, handle_parsing_errors=True, max_iterations=5)

