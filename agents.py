"""
Agent orchestration layer.

- MCP server: exposes discrete "tools" (get_current_time, convert_temperature,
  lookup_course, gpa_calculator, define_term, ...) over HTTP.
- A2A server: exposes higher-level "skills" run by separate agents
  (quiz generation, summarization, translation) over HTTP.
- route(): a lightweight intent router that decides whether a user
  utterance should hit a specific MCP tool, an A2A agent skill, or fall
  back to the Pinecone RAG chatbot for open-ended department questions.

Both servers are already deployed independently (see config.py for URLs).
This module just talks to them and adds an in-memory response cache for
MCP calls to avoid redundant network round-trips within a session.
"""
import re
import requests

from config import MCP_SERVER_URL, A2A_SERVER_URL
from rag import ask_rag

mcp_cache: dict = {}


def call_mcp(tool_name: str, arguments: dict | None = None) -> str:
    """Call an MCP tool, with simple in-memory caching by (tool, args)."""
    arguments = arguments or {}
    cache_key = f"{tool_name}_{arguments}"
    if cache_key in mcp_cache:
        return mcp_cache[cache_key]
    try:
        r = requests.post(
            f"{MCP_SERVER_URL}/call",
            json={"tool_name": tool_name, "arguments": arguments},
            timeout=30,
        )
        r.raise_for_status()
        result = r.json().get("result", "No result")
    except requests.RequestException as e:
        result = f"(MCP tool '{tool_name}' unavailable: {e})"
    mcp_cache[cache_key] = result
    return result


def call_a2a(agent: str, skill: str, text: str) -> str:
    """Call an A2A agent skill (quiz / summarizer / translator)."""
    try:
        r = requests.post(
            f"{A2A_SERVER_URL}/{agent}/execute/{skill}",
            json={"text": text},
            timeout=30,
        )
        r.raise_for_status()
        return r.json().get("result", "No result")
    except requests.RequestException as e:
        return f"(A2A agent '{agent}' unavailable: {e})"


def route(user_input: str, thread_id: str = "default-session") -> str:
    """Decide which tool/agent/RAG path should answer this utterance."""
    text = user_input.lower()

    if any(w in text for w in ["quiz", "test me", "quiz me"]):
        topic = re.sub(r"quiz me on|quiz me about|quiz me|test me on", "", text).strip()
        return call_a2a("quiz", "quiz", topic)

    if any(w in text for w in ["summarize", "shorten", "summary"]):
        return call_a2a("summarizer", "summarize", user_input)

    if any(w in text for w in ["translate", "spanish", "in spanish"]):
        return call_a2a("translator", "translate", user_input)

    if any(w in text for w in ["time", "date", "what time"]):
        return call_mcp("get_current_time")

    if any(w in text for w in ["fahrenheit", "celsius", "convert", "temperature"]):
        nums = re.findall(r"\d+\.?\d*", user_input)
        value = float(nums[0]) if nums else 0
        unit = "F" if "fahrenheit" in text or "f to c" in text else "C"
        return call_mcp("convert_temperature", {"value": value, "from_unit": unit})

    if any(w in text for w in ["gpa", "grade", "grades"]):
        grades = [{"grade": "A", "credits": 3}, {"grade": "B", "credits": 3}]
        return call_mcp("gpa_calculator", {"grades": grades})

    if any(w in text for w in ["cis", "course", "class"]):
        match = re.search(r"CIS\s?\d+", user_input, re.IGNORECASE)
        course_code = match.group(0).upper() if match else "CIS 360"
        return call_mcp("lookup_course", {"course_code": course_code})

    if any(w in text for w in ["define", "what is", "meaning"]):
        term = re.sub(r"define|what is|meaning of", "", text).strip()
        return call_mcp("define_term", {"term": term})

    return ask_rag(user_input, thread_id=thread_id)
