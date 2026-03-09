"""
Conversation Memory (Episodic Memory)
=======================================
Manages multi-turn conversation history for stateful agent interactions.
Provides rolling summarisation to keep the context window bounded while
preserving essential information from earlier turns.

Where This Fits in the MAS Architecture
-----------------------------------------
Position in the multi-tier memory stack:

    Working Memory      — per-execution scratchpad
    Conversation Memory — per-session dialogue history (THIS FILE)
         ↕ LangGraph add_messages reducer + MemorySaver
    Long-Term Memory    — persistent knowledge base (long_term_memory.py)

The episodic memory problem:
    In a multi-turn conversation, each new turn adds messages to history.
    Left unchecked, the history grows indefinitely, eventually exceeding
    the LLM's context window (typically 8K–128K tokens).

    Even before hitting the hard limit, long histories are expensive:
    every token in context is billed. A 50-turn conversation can
    accumulate thousands of tokens of history that the LLM mostly ignores.

Rolling summarisation strategy:
    Instead of sending the full history on every turn, we:
        1. Keep the last N messages intact ("recent window")
        2. Summarise everything before the window into a single paragraph
        3. Prepend the summary as a SystemMessage on future turns

    The agent sees: [SUMMARY OF TURNS 1-10] + [TURNS 11-14] + [NEW TURN]
    Not: [TURN 1] + [TURN 2] + ... + [TURN 14] + [NEW TURN]

    This keeps the context bounded while preserving continuity.

Alternative strategies (commented for comparison):
    Token-based windowing  — count tokens instead of messages (tiktoken)
    Importance scoring     — keep high-importance messages longer
    Selective forgetting   — drop low-value exchanges (greetings, acks)

LangGraph integration:
    This module works with LangGraph's add_messages reducer and
    MemorySaver checkpointer. The summarise_history() method is called
    inside a LangGraph node to compress the messages state.

    See scripts/memory_management/conversation_memory.py for the full
    pattern showing how this integrates with StateGraph.

Pattern script:
    scripts/memory_management/conversation_memory.py  — Pattern 4

Usage:
    from memory.conversation_memory import ConversationMemory

    cm = ConversationMemory(summarise_after=4, history_window=6)
    if cm.should_summarise(state["messages"]):
        updated_messages, new_summary = cm.summarise_history(
            messages=state["messages"],
            old_summary=state.get("summary", ""),
            llm=get_llm(),
        )
        return {"messages": updated_messages, "summary": new_summary}
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


# ============================================================
# ConversationMemory Class
# ============================================================

class ConversationMemory:
    """
    Manages conversation history with rolling summarisation.

    Concept — Why a class for conversation memory:
        Summarisation involves configurable thresholds (when to summarise,
        how many messages to keep) and stateless operations (compress
        messages, build the summary prompt). Encapsulating these in a
        class makes the policy explicit and testable.

        The class is STATELESS about the actual messages — it does not
        store messages internally. It provides METHODS that operate on
        the message lists stored in LangGraph state.

    Args:
        summarise_after: Number of messages that triggers summarisation.
            When len(messages) > summarise_after, should_summarise() is True.
            Default: 6 (3 user+AI turn pairs)
        history_window: Number of recent messages to keep intact after
            summarisation. Everything before the window is compressed.
            Default: 4 (2 recent turn pairs)
        summarise_system_prompt: System prompt used when calling the LLM
            to generate the summary. Override for domain customisation.
    """

    DEFAULT_SYSTEM_PROMPT = "You are a medical conversation summariser."
    DEFAULT_SUMMARISE_PROMPT_TEMPLATE = (
        "Summarise the following conversation history into a concise paragraph. "
        "Capture the key medical facts, decisions made, medications discussed, "
        "and any pending actions. Do not include greetings or pleasantries.\n\n"
        "{previous_summary}"
        "Messages to summarise:\n"
        "{messages_text}"
    )

    def __init__(
        self,
        summarise_after: int = 6,
        history_window: int = 4,
        summarise_system_prompt: str | None = None,
    ):
        self.summarise_after = summarise_after
        self.history_window = history_window
        self.summarise_system_prompt = (
            summarise_system_prompt or self.DEFAULT_SYSTEM_PROMPT
        )

    # ──────────────────────────────────────────────
    # Core Methods
    # ──────────────────────────────────────────────

    def should_summarise(self, messages: list) -> bool:
        """
        Check if the conversation history is long enough to need summarisation.

        Concept — Message count vs token count:
            Counting messages is simple but imprecise. A message with a
            long clinical history is much more expensive than a one-word ack.
            Production systems use token counting (e.g., tiktoken) for
            more accurate context window management.

            Message counting is the right starting point for learning —
            easy to understand and works well for consistent message lengths.

        Args:
            messages: List of LangChain message objects from state.

        Returns:
            True if len(messages) > self.summarise_after.
        """
        return len(messages) > self.summarise_after

    def window_messages(self, messages: list) -> tuple[list, list]:
        """
        Split messages into "old" (to summarise) and "recent" (to keep).

        The recent window size is controlled by self.history_window.
        Everything before the window is "old" and gets summarised.

        Args:
            messages: Full conversation history.

        Returns:
            Tuple of (old_messages, recent_messages).
            old_messages: Messages to include in the summary.
            recent_messages: Messages to keep in the context window.

        Example:
            history_window = 4
            messages = [m1, m2, m3, m4, m5, m6]
            → old_messages = [m1, m2]
            → recent_messages = [m3, m4, m5, m6]
        """
        if len(messages) <= self.history_window:
            return [], messages

        old_messages = messages[:-self.history_window]
        recent_messages = messages[-self.history_window:]
        return old_messages, recent_messages

    def summarise_history(
        self,
        messages: list,
        old_summary: str,
        llm: Any,
        max_message_preview: int = 200,
    ) -> tuple[list, str]:
        """
        Compress old messages into a rolling summary using the LLM.

        Concept — Rolling summarisation:
            Each call to this method extends an existing summary with
            the newly-old messages. The result is a single paragraph
            that replaces all old messages with their essence.

            The LLM sees:
                [Previous summary: ...]  (if exists)
                [Messages to summarise: msg1, msg2, ...]
            And produces a new summary paragraph.

            The returned message list has the summary as a SystemMessage
            prepended to the recent messages. This gives the LLM context
            about the conversation history without the token overhead of
            the full history.

        Args:
            messages: Full current message list.
            old_summary: Existing summary from previous summarisation calls.
                Empty string on first summarisation.
            llm: Any LangChain chat model for generating the summary.
            max_message_preview: Max chars per message in the summary prompt.

        Returns:
            Tuple of (new_message_list, new_summary_text).
            new_message_list: [SystemMessage(summary)] + recent_messages
            new_summary_text: The new summary string for storing in state.
        """
        from langchain_core.messages import HumanMessage, SystemMessage

        old_messages, recent_messages = self.window_messages(messages)

        if not old_messages:
            logger.debug("No old messages to summarise")
            return messages, old_summary

        # Build the summary prompt
        previous_summary_section = ""
        if old_summary:
            previous_summary_section = f"Previous summary:\n{old_summary}\n\n"

        messages_text = ""
        for msg in old_messages:
            # Detect message role by class name (avoids importing all message types)
            role = type(msg).__name__.replace("Message", "").replace("AI", "AI").replace("Human", "Human")
            content = msg.content if hasattr(msg, "content") else str(msg)
            messages_text += f"  {role}: {content[:max_message_preview]}\n"

        prompt_text = self.DEFAULT_SUMMARISE_PROMPT_TEMPLATE.format(
            previous_summary=previous_summary_section,
            messages_text=messages_text,
        )

        summary_response = llm.invoke([
            SystemMessage(content=self.summarise_system_prompt),
            HumanMessage(content=prompt_text),
        ])

        new_summary = summary_response.content
        logger.info(
            f"ConversationMemory: summarised {len(old_messages)} messages "
            f"into {len(new_summary)} char summary"
        )

        # Build the new message list: summary as context + recent messages
        summary_context = SystemMessage(
            content=f"CONVERSATION HISTORY SUMMARY:\n{new_summary}"
        )
        new_message_list = [summary_context] + recent_messages

        return new_message_list, new_summary

    # ──────────────────────────────────────────────
    # Convenience: One-step check and summarise
    # ──────────────────────────────────────────────

    def maybe_summarise(
        self,
        messages: list,
        old_summary: str,
        llm: Any,
    ) -> tuple[list, str]:
        """
        Conditionally summarise only if should_summarise() returns True.

        Convenience wrapper used in graph nodes to handle both branches
        with a single call.

        Args:
            messages: Current message list.
            old_summary: Existing summary or empty string.
            llm: LangChain chat model.

        Returns:
            Tuple of (messages, summary). If summarisation was not needed,
            returns (messages, old_summary) unchanged.

        Usage in a LangGraph node:
            def check_and_summarise_node(state):
                cm = ConversationMemory()
                new_msgs, new_summary = cm.maybe_summarise(
                    messages=state["messages"],
                    old_summary=state.get("summary", ""),
                    llm=get_llm(),
                )
                return {"messages": new_msgs, "summary": new_summary}
        """
        if not self.should_summarise(messages):
            return messages, old_summary

        return self.summarise_history(messages, old_summary, llm)
