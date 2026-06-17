# Explanation

Deep-dive notes and contracts that go beyond the auto-generated [API Reference](../reference/index.md). The API Reference renders module/class/function signatures and docstrings; the pages here carry the load-bearing *prose* — wire-level contracts, translation internals, and architectural constraints — that a signature alone cannot express.

- [**Wire-protocol contract**](wire-contract.md) — the explicit Workspace ⇄ agent-server contract: `agents.json` + `/v1/query` request/response shapes, every artifact type, and emission ordering.
- [**SSE adapter internals**](adapter.md) — how DeepAgents stream events are translated into OpenBB SSE: thinking/prose routing, citation de-duplication and relevance filtering, Harmony-format leak suppression.
- [**Canvas rendering contract**](canvas.md) — PyWry's three rendering paths, the TVChart toolbar dual-dispatch shim, the datafeed-backed Symbol Search / Compare wiring, and MIME-dispatched document rendering.
