"""This module provides a utility to wrap tools with human-in-the-loop (HIL) review functionality.

enabling human intervention in tool execution workflows compatible with Agent Inbox UIs.
"""

from typing import Any, Callable, Dict, Optional

from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool
from langchain_core.tools import tool as create_tool
from langgraph.types import interrupt


def add_human_in_the_loop(
    tool_to_wrap: Callable | BaseTool,
    *,
    interrupt_config: Optional[Dict[str, bool]] = None,
) -> BaseTool:
    """Wrap a tool to support human-in-the-loop review, compatible with Agent Inbox.

    This function takes a tool (either a callable or a BaseTool instance) and
    returns a new BaseTool. When this new tool is invoked, it first triggers
    a human-in-the-loop (HIL) interrupt, pausing the graph's execution.
    The interrupt payload is structured to be compatible with UIs like Agent Inbox,
    allowing a human user to review, accept, edit, or respond to the intended
    tool call.

    Args:
        tool_to_wrap: The tool or callable to be wrapped with HIL.
        interrupt_config: Optional. A dictionary to configure HIL behavior,
            specifying which actions (e.g., "allow_accept", "allow_edit")
            are permitted. If None, a default configuration is used.

    Returns:
        A new BaseTool instance that incorporates the HIL step.
    """
    if not isinstance(tool_to_wrap, BaseTool):
        tool_instance = create_tool(tool_to_wrap)
    else:
        tool_instance = tool_to_wrap

    # Default interrupt configuration.
    # These keys are expected by agent-chat-ui's isAgentInboxInterruptSchema.
    default_config: Dict[str, bool] = {
        "allow_accept": True,
        "allow_edit": True,
        "allow_respond": True,
        "allow_ignore": True,  # Frontend schema checks for this key
    }

    final_interrupt_config = (
        interrupt_config if interrupt_config is not None else default_config
    )

    @create_tool(
        tool_instance.name,  # Pass name as a positional argument
        description=tool_instance.description,
        args_schema=tool_instance.args_schema,
    )
    async def call_tool_with_interrupt(
        config: RunnableConfig, **tool_input: Any
    ) -> Any:
        """Invoke the wrapped tool after a human-in-the-loop interrupt."""
        # Construct the interrupt request payload.
        # This structure must match what
        # agent-chat-ui/src/lib/agent-inbox-interrupt.ts expects.
        interrupt_payload: Dict[str, Any] = {
            "action_request": {
                "action": tool_instance.name,
                "args": tool_input,
            },
            "config": final_interrupt_config,
            "description": f"Please review the tool call to '{tool_instance.name}'.",
        }

        # interrupt() expects a list of payloads and returns a list of responses.
        # We send one interrupt, so we expect one response from that interrupt.
        response_list = interrupt([interrupt_payload])

        if not response_list:
            # This case should ideally not happen if interrupt() behaves as expected.
            raise ValueError("Interrupt did not return a response.")

        response_from_human = response_list[
            0
        ]  # Get the first (and typically only) response

        response_type = response_from_human.get("type")
        response_args = response_from_human.get("args")

        if response_type == "accept":
            # User approved, run the original tool asynchronously
            return await tool_instance.ainvoke(tool_input, config=config)
        elif response_type == "edit":
            # User edited the arguments.
            # Agent Inbox sends edited args under response_args["args"].
            if response_args and "args" in response_args:
                edited_tool_input = response_args["args"]
                return await tool_instance.ainvoke(edited_tool_input, config=config)
            else:
                raise ValueError(
                    "Edit response type received, but valid 'args' were not found in the response."
                )
        elif response_type == "response":
            # User provided a direct response to be used as tool output.
            # Agent Inbox typically sends the response string directly in response_args.
            return response_args
        elif response_type == "ignore":
            # User chose to ignore the tool call.
            return f"Tool call '{tool_instance.name}' was ignored by the user."
        else:
            raise ValueError(
                f"Unsupported interrupt response type received: '{response_type}'"
            )

    return call_tool_with_interrupt
