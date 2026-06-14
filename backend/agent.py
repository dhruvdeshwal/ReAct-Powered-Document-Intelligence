"""
agent.py
--------
ReAct (Reasoning + Acting) agent.

Flow:
  User Query -> LLM produces "Thought" + "Action" + "Action Input"
             -> Agent executes the Action (a tool)
             -> Tool result becomes "Observation"
             -> Looped back into the LLM until it produces "Final Answer"

Tools available to the agent:
  - search_documents(query): semantic search over indexed chunks
  - list_documents(): list indexed source files
  - summarize(text): ask the LLM to summarize a block of text

Also supports simple query decomposition for multi-part questions.
"""

import os
import re
import json
from groq import Groq
from dotenv import load_dotenv
load_dotenv()

from vector_store import VectorStore

LLM_MODEL = os.getenv("LLM_MODEL", "llama-3.3-70b-versatile")
MAX_ITERATIONS = int(os.getenv("MAX_ITERATIONS", 5))

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
_client = Groq(api_key=GROQ_API_KEY) if GROQ_API_KEY else None


# ---------------------------------------------------------------------------
# LLM call wrapper
# ---------------------------------------------------------------------------
def call_llm(prompt: str, system: str = "", temperature: float = 0.0) -> str:
    """Send a prompt to the Groq-hosted LLM and return the text response."""
    if _client is None:
        raise RuntimeError("GROQ_API_KEY not set. Add it to your .env file.")

    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    response = _client.chat.completions.create(
        model=LLM_MODEL,
        messages=messages,
        temperature=temperature,
    )
    return response.choices[0].message.content.strip()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------
class Tools:
    """Container for tools the ReAct agent can invoke."""

    def __init__(self, vector_store: VectorStore):
        self.vector_store = vector_store

    def search_documents(self, query: str) -> str:
        """Semantic search over indexed document chunks. Returns formatted context."""
        results = self.vector_store.search(query)
        if not results:
            return "No relevant information found in the indexed documents."

        formatted = []
        for r in results:
            src = r["metadata"].get("source", "unknown")
            page = r["metadata"].get("page", -1)
            page_info = f", page {page}" if page and page != -1 else ""
            formatted.append(f"[Source: {src}{page_info}]\n{r['text']}")

        return "\n\n".join(formatted)

    def list_documents(self, _: str = "") -> str:
        """List all currently indexed documents."""
        docs = self.vector_store.list_documents()
        if not docs:
            return "No documents have been indexed yet."
        return "Indexed documents: " + ", ".join(docs)

    def summarize(self, text: str) -> str:
        """Ask the LLM to summarize a block of text."""
        prompt = f"Summarize the following text concisely:\n\n{text}"
        return call_llm(prompt)

    def dispatch(self, action: str, action_input: str) -> str:
        """Route an action name to the corresponding tool method."""
        action = action.strip().lower()
        if action == "search_documents":
            return self.search_documents(action_input)
        elif action == "list_documents":
            return self.list_documents(action_input)
        elif action == "summarize":
            return self.summarize(action_input)
        else:
            return f"Unknown action '{action}'. Valid actions are: search_documents, list_documents, summarize."


# ---------------------------------------------------------------------------
# ReAct prompt template
# ---------------------------------------------------------------------------
REACT_SYSTEM_PROMPT = """You are a document analysis assistant that answers questions using the ReAct \
(Reasoning + Acting) methodology.

You have access to the following tools:
- search_documents: Search the indexed documents for relevant information. Input: a search query string.
- list_documents: List all documents currently indexed. Input: empty string.
- summarize: Summarize a block of text. Input: the text to summarize.

Use the following strict format for EVERY step:

Thought: <your reasoning about what to do next>
Action: <one of: search_documents, list_documents, summarize>
Action Input: <the input to the action>

After you receive an "Observation:", continue with another Thought/Action/Action Input,
OR if you have enough information, respond with:

Thought: <your final reasoning>
Final Answer: <the complete answer to the user's question, grounded in the observations>

Rules:
- Always start with a "Thought:".
- Only output ONE Thought/Action/Action Input block per turn (stop after "Action Input:").
- Never fabricate information that wasn't in an Observation.
- If documents don't contain the answer, say so honestly in the Final Answer.
"""


# ---------------------------------------------------------------------------
# Parsing helpers
# ---------------------------------------------------------------------------
def _parse_step(llm_output: str) -> dict:
    """
    Parse an LLM output into one of:
      {"type": "final", "answer": "..."}
      {"type": "action", "thought": "...", "action": "...", "action_input": "..."}
      {"type": "unknown", "raw": "..."}
    """
    final_match = re.search(r"Final Answer:\s*(.*)", llm_output, re.DOTALL)
    if final_match:
        return {"type": "final", "answer": final_match.group(1).strip()}

    thought_match = re.search(r"Thought:\s*(.*?)(?=\nAction:|\Z)", llm_output, re.DOTALL)
    action_match = re.search(r"Action:\s*(.*?)(?=\n|\Z)", llm_output)
    input_match = re.search(r"Action Input:\s*(.*)", llm_output, re.DOTALL)

    if action_match and input_match:
        return {
            "type": "action",
            "thought": thought_match.group(1).strip() if thought_match else "",
            "action": action_match.group(1).strip(),
            "action_input": input_match.group(1).strip(),
        }

    return {"type": "unknown", "raw": llm_output}


# ---------------------------------------------------------------------------
# Query decomposition (for multi-part questions)
# ---------------------------------------------------------------------------
def decompose_query(query: str) -> list[str]:
    """
    Ask the LLM to break a complex query into simpler sub-questions.
    Returns a list of sub-questions (or the original query as a single-item list
    if decomposition isn't needed / fails).
    """
    prompt = f"""Analyze this user question. If it contains multiple distinct sub-questions
or requires comparing/combining information about different topics, break it into
a numbered list of simple, self-contained sub-questions. If it's already simple,
just return it as a single item.

Question: {query}

Respond ONLY with a JSON array of strings, e.g. ["sub-question 1", "sub-question 2"]"""

    try:
        raw = call_llm(prompt)
        raw = re.sub(r"```json|```", "", raw).strip()
        sub_questions = json.loads(raw)
        if isinstance(sub_questions, list) and all(isinstance(s, str) for s in sub_questions):
            return sub_questions
    except Exception:
        pass

    return [query]


# ---------------------------------------------------------------------------
# ReAct Agent
# ---------------------------------------------------------------------------
class ReActAgent:
    def __init__(self, vector_store: VectorStore, max_iterations: int = MAX_ITERATIONS):
        self.tools = Tools(vector_store)
        self.max_iterations = max_iterations

    def run(self, query: str, verbose: bool = True) -> dict:
        """
        Run the ReAct loop for a single (sub-)question.

        Returns:
            {
                "answer": "...",
                "trace": [ {step dicts...} ]
            }
        """
        trace = []
        scratchpad = ""

        for step in range(self.max_iterations):
            prompt = f"Question: {query}\n\n{scratchpad}\nThought:"
            llm_output = call_llm(prompt, system=REACT_SYSTEM_PROMPT)
            parsed = _parse_step("Thought:" + llm_output if not llm_output.startswith("Thought") else llm_output)

            if parsed["type"] == "final":
                trace.append({"step": step, "type": "final", "answer": parsed["answer"]})
                return {"answer": parsed["answer"], "trace": trace}

            if parsed["type"] == "action":
                observation = self.tools.dispatch(parsed["action"], parsed["action_input"])
                trace.append({
                    "step": step,
                    "type": "action",
                    "thought": parsed["thought"],
                    "action": parsed["action"],
                    "action_input": parsed["action_input"],
                    "observation": observation,
                })
                scratchpad += (
                    f"Thought: {parsed['thought']}\n"
                    f"Action: {parsed['action']}\n"
                    f"Action Input: {parsed['action_input']}\n"
                    f"Observation: {observation}\n"
                )
                continue

            # Unknown format - try to salvage by treating raw output as final answer
            trace.append({"step": step, "type": "unparsed", "raw": llm_output})
            return {"answer": llm_output, "trace": trace}

        # Max iterations hit - force a final answer using gathered observations
        final_prompt = (
            f"Question: {query}\n\n{scratchpad}\n"
            "Based on the observations above, provide a Final Answer now:\nFinal Answer:"
        )
        answer = call_llm(final_prompt, system=REACT_SYSTEM_PROMPT)
        trace.append({"step": self.max_iterations, "type": "final_forced", "answer": answer})
        return {"answer": answer, "trace": trace}

    def run_with_decomposition(self, query: str) -> dict:
        """
        Decompose a complex query into sub-questions, run each through the ReAct loop,
        then synthesize a final combined answer.
        """
        sub_questions = decompose_query(query)

        if len(sub_questions) == 1:
            return self.run(query)

        sub_results = []
        full_trace = []
        for sq in sub_questions:
            result = self.run(sq)
            sub_results.append({"question": sq, "answer": result["answer"]})
            full_trace.append({"sub_question": sq, "trace": result["trace"]})

        synthesis_prompt = "Original question: " + query + "\n\n"
        synthesis_prompt += "Here are answers to its sub-questions:\n"
        for sr in sub_results:
            synthesis_prompt += f"- Q: {sr['question']}\n  A: {sr['answer']}\n"
        synthesis_prompt += "\nCombine these into one coherent, complete answer to the original question."

        final_answer = call_llm(synthesis_prompt)

        return {
            "answer": final_answer,
            "trace": full_trace,
            "sub_questions": sub_questions,
            "sub_results": sub_results,
        }


if __name__ == "__main__":
    # Quick standalone test (requires GROQ_API_KEY set and an indexed document)
    vs = VectorStore()
    agent = ReActAgent(vs)
    result = agent.run_with_decomposition("What documents are indexed and who created python?")
    print("ANSWER:", result["answer"])
    print("\nTRACE:")
    for t in result["trace"]:
        print(t)