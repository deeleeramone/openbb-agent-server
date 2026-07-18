# Explanation

This section is intentionally small. It contains architecture contracts and internals that do not fit well in API docs.

## Pages in this section

1. [Wire-protocol contract](wire-contract.md)
	 Workspace to server request/response contract (`agents.json` and `/v1/query`), event types, and ordering.
2. [SSE adapter internals](adapter.md)
	 How DeepAgents stream events are converted into OpenBB SSE payloads.
3. [Canvas rendering contract](canvas.md)
	 How the desktop canvas updates, renders content, and handles TVChart/document modes.

## Where everything else is

- [User Guides](../guides/index.md)
	End-to-end workflows: Workspace integration, widgets, memory, multimodal, background jobs.
- [Advanced Configuration](../operating/index.md)
	Profiles, auth, persistence, memory, observability.
- [Developing Plugins](../developing/index.md)
	Tool/model/middleware/subagent/auth extension guides.
- [API Reference](../reference/index.md)
	Module/class/function signatures generated from code docstrings.
