"""OpenBB Workspace agent wire-protocol schemas.

Pydantic models for both directions of the ``/v1/query`` exchange:
:class:`QueryRequest` (with its :class:`ChatMessage`, :class:`WidgetsBag`
and :class:`UploadedFile` parts) is the request body, and the
:data:`SSEEvent` union is the Server-Sent-Events stream the endpoint
replies with — :class:`MessageChunkSSE` (text deltas),
:class:`StatusUpdateSSE` (reasoning steps), :class:`FunctionCallSSE`
(client-side calls), :class:`MessageArtifactSSE` (inline artifacts) and
:class:`CitationCollectionSSE` (end-of-run citations). Request models set
``extra="allow"`` so unrecognised Workspace fields survive the round
trip.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_serializer

MessageRole = Literal["human", "ai", "tool"]


class ChatMessage(BaseModel):
    """One message in the multi-turn conversation history.

    Mirrors the Workspace copilot message shape; unrecognised keys are
    preserved (``extra="allow"``).

    Attributes
    ----------
    role : MessageRole
        Author of the message — ``"human"``, ``"ai"`` or ``"tool"``.
    content : str or dict[str, Any] or None
        Message body: plain text for ordinary turns, or a structured
        payload for non-text messages.
    tool_call_id : str or None
        Identifier linking a ``"tool"`` result back to the call that
        produced it.
    function : str or None
        Name of the function/tool the message records, for tool
        invocations and their results.
    input_arguments : dict[str, Any] or None
        Arguments the function was invoked with, when applicable.
    data : list[Any] or None
        Structured data carried with the message (e.g. widget result
        rows).
    agent_id : str or None
        Identifier of the sub-agent that produced the message, in
        multi-agent runs.
    """

    model_config = ConfigDict(extra="allow")

    role: MessageRole
    content: str | dict[str, Any] | None = None
    tool_call_id: str | None = None
    function: str | None = None
    input_arguments: dict[str, Any] | None = None
    data: list[Any] | None = None
    agent_id: str | None = None


class WidgetParam(BaseModel):
    """One parameter advertised by a widget.

    Attributes
    ----------
    name : str
        Parameter name as defined by the widget.
    type : str or None
        Declared parameter type, when provided.
    current_value : Any
        Value currently selected for the parameter in the UI.
    """

    model_config = ConfigDict(extra="allow")

    name: str
    type: str | None = None
    current_value: Any = None


class WidgetSpec(BaseModel):
    """One Workspace-attached widget context entry.

    Attributes
    ----------
    uuid : str or None
        Unique identifier of the widget instance.
    widget_id : str or None
        Identifier of the widget template/type.
    name : str or None
        Display name of the widget.
    type : str or None
        Widget type/category.
    origin : str or None
        Origin app or data source the widget belongs to.
    description : str or None
        Human-readable description of the widget.
    params : list[WidgetParam] or dict[str, Any]
        Parameters configured on the widget, as a list of
        :class:`WidgetParam` or a raw mapping.
    data : Any
        The widget's resolved data payload, when Workspace inlines it.
    """

    model_config = ConfigDict(extra="allow")

    uuid: str | None = None
    widget_id: str | None = None
    name: str | None = None
    type: str | None = None
    origin: str | None = None
    description: str | None = None
    params: list[WidgetParam] | dict[str, Any] = Field(default_factory=list)
    data: Any = None

    @property
    def id(self) -> str:
        """Return a stable lookup key (``uuid`` if set, else ``widget_id``).

        Returns
        -------
        str
            The widget's ``uuid`` when present, otherwise its
            ``widget_id``, or an empty string when neither is set.
        """
        return self.uuid or self.widget_id or ""


class WidgetsBag(BaseModel):
    """Widget context attached to a request, grouped by priority tier.

    The server flattens all three tiers into the run context; the tiers
    only convey the Workspace's relative prioritisation.

    Attributes
    ----------
    primary : list[WidgetSpec]
        Highest-priority widgets the user has put in focus.
    secondary : list[WidgetSpec]
        Supporting widgets on the active dashboard.
    extra : list[WidgetSpec]
        Additional widgets supplied as background context.
    """

    model_config = ConfigDict(extra="allow")

    primary: list[WidgetSpec] = Field(default_factory=list)
    secondary: list[WidgetSpec] = Field(default_factory=list)
    extra: list[WidgetSpec] = Field(default_factory=list)


class UploadedFile(BaseModel):
    """One user-uploaded file (PDF / image / spreadsheet / raw).

    The bytes arrive either inlined as base64 or by reference via a URL.

    Attributes
    ----------
    name : str
        Original file name, used for display and type inference.
    mime : str or None
        MIME type when known; ``None`` falls back to extension/content
        sniffing.
    data_base64 : str or None
        Base64-encoded file contents when the bytes are inlined.
    url : str or None
        Location to fetch the file from when it is not inlined.
    """

    model_config = ConfigDict(extra="allow")

    name: str
    mime: str | None = None
    data_base64: str | None = None
    url: str | None = None


class QueryRequest(BaseModel):
    """Body of ``POST /v1/query``.

    Unrecognised keys are preserved (``extra="allow"``) so newer
    Workspace fields pass through untouched.

    Attributes
    ----------
    messages : list[ChatMessage]
        Full multi-turn history, with the new human message last.
    widgets : WidgetsBag
        Widget context the user attached to the turn.
    uploaded_files : list[UploadedFile]
        Files attached to the turn.
    api_keys : dict[str, Any]
        Provider API keys forwarded by Workspace, keyed by provider.
    api_urls : dict[str, Any]
        Provider base URLs forwarded by Workspace, keyed by provider.
    workspace_options : dict[str, Any]
        Per-user custom feature toggles, normalised to an
        option-id-keyed mapping by :meth:`_coerce_workspace_options`.
    timezone : str or None
        IANA timezone for localizing the run.
    context : list[dict[str, Any]] or None
        Additional free-form context entries supplied by the client.
    urls : list[str] or None
        URLs the user asked the agent to consider.
    force_web_search : bool or None
        When ``True``, force a web search regardless of heuristics.
    workspace_state : dict[str, Any] or None
        Snapshot of relevant Workspace UI state.
    tools : list[dict[str, Any]] or None
        Client-side tool declarations the Workspace UI can execute.
    """

    model_config = ConfigDict(extra="allow")

    messages: list[ChatMessage]
    widgets: WidgetsBag = Field(default_factory=WidgetsBag)
    uploaded_files: list[UploadedFile] = Field(default_factory=list)

    api_keys: dict[str, Any] = Field(default_factory=dict)
    api_urls: dict[str, Any] = Field(default_factory=dict)

    workspace_options: dict[str, Any] = Field(default_factory=dict)

    timezone: str | None = None

    context: list[dict[str, Any]] | None = None
    urls: list[str] | None = None
    force_web_search: bool | None = None
    workspace_state: dict[str, Any] | None = None
    tools: list[dict[str, Any]] | None = None

    @field_validator("workspace_options", mode="before")
    @classmethod
    def _coerce_workspace_options(cls, value: Any) -> Any:
        """Normalise ``workspace_options`` to an option-id-keyed dict."""
        if value is None:
            return {}
        if isinstance(value, list):
            return {str(item): True for item in value}
        return value


class BaseSSE(BaseModel):
    """Base class for every SSE event variant.

    Attributes
    ----------
    event : str
        The SSE event name the Workspace client dispatches on; each
        subclass pins this to a constant.
    data : Any
        The event payload; each subclass narrows it to a concrete type.
    """

    event: str
    data: Any


class MessageChunkSSEData(BaseModel):
    """Payload for the ``copilotMessageChunk`` event.

    Attributes
    ----------
    delta : str
        The next chunk of streamed assistant text.
    """

    delta: str


class MessageChunkSSE(BaseSSE):
    """Streaming text delta from the model.

    Attributes
    ----------
    data : MessageChunkSSEData
        The text-delta payload for this chunk.
    """

    event: Literal["copilotMessageChunk"] = "copilotMessageChunk"
    data: MessageChunkSSEData


StatusEventType = Literal["INFO", "SUCCESS", "WARNING", "ERROR"]


class StatusUpdateSSEData(BaseModel):
    """Payload for the ``copilotStatusUpdate`` event.

    Attributes
    ----------
    eventType : StatusEventType
        Severity of the update — ``"INFO"``, ``"SUCCESS"``,
        ``"WARNING"`` or ``"ERROR"``.
    message : str
        Human-readable status line shown to the user.
    group : Literal["reasoning"]
        Group the update belongs to; always ``"reasoning"``.
    details : list[dict[str, Any] or str] or None
        Optional structured or textual detail lines.
    artifacts : list[ClientArtifact] or None
        Optional artifacts to surface alongside the status.
    hidden : bool
        When ``True``, the update is recorded but not shown in the UI.
    """

    eventType: StatusEventType
    message: str
    group: Literal["reasoning"] = "reasoning"
    details: list[dict[str, Any] | str] | None = None
    artifacts: list[ClientArtifact] | None = None
    hidden: bool = False


class StatusUpdateSSE(BaseSSE):
    """A reasoning-step status update.

    Attributes
    ----------
    data : StatusUpdateSSEData
        The status payload.
    """

    event: Literal["copilotStatusUpdate"] = "copilotStatusUpdate"
    data: StatusUpdateSSEData


FunctionName = Literal[
    "get_widget_data",
    "get_extra_widget_data",
    "get_params_options",
    "add_widget_to_dashboard",
    "add_generative_widget",
    "update_widget_in_dashboard",
    "assign_tasks_to_agents",
    "execute_agent_tool",
    "manage_navigation_bar",
    "get_skill_content",
]


class FunctionCallSSEData(BaseModel):
    """Payload for the ``copilotFunctionCall`` event.

    Attributes
    ----------
    function : FunctionName
        Name of the client-side function the Workspace UI must execute.
    input_arguments : dict[str, Any]
        Arguments to invoke the function with.
    extra_state : dict[str, Any] or None
        Opaque state echoed back by the client on the follow-up turn.
    """

    function: FunctionName
    input_arguments: dict[str, Any] = Field(default_factory=dict)
    extra_state: dict[str, Any] | None = None


class FunctionCallSSE(BaseSSE):
    """A client-side function call the Workspace UI must execute.

    Emitting this event ends the current run; the client executes the
    function and starts a new turn with the result.

    Attributes
    ----------
    data : FunctionCallSSEData
        The function name and arguments to execute.
    """

    event: Literal["copilotFunctionCall"] = "copilotFunctionCall"
    data: FunctionCallSSEData


ArtifactType = Literal[
    "text",
    "table",
    "chart",
    "code",
    "html",
]


class ClientArtifact(BaseModel):
    """The single artifact shape Workspace consumes.

    Attributes
    ----------
    type : ArtifactType
        Artifact kind — ``"text"``, ``"table"``, ``"chart"``, ``"code"``
        or ``"html"``.
    name : str
        Short display name.
    description : str
        Human-readable description of the artifact.
    uuid : str
        Unique identifier of the artifact.
    content : str or list[dict[str, Any]]
        The artifact body: text/HTML markup, or row records for tables
        and charts.
    chart_params : dict[str, Any] or None
        Chart rendering options, when ``type`` is ``"chart"``.
    """

    type: ArtifactType
    name: str
    description: str
    uuid: str
    content: str | list[dict[str, Any]]
    chart_params: dict[str, Any] | None = None


class MessageArtifactSSE(BaseSSE):
    """An artifact the chat panel renders inline.

    Attributes
    ----------
    data : ClientArtifact
        The artifact to render.
    """

    event: Literal["copilotMessageArtifact"] = "copilotMessageArtifact"
    data: ClientArtifact


class CitationHighlightBoundingBox(BaseModel):
    """Pixel bounding box for a quoted span on a PDF page.

    Attributes
    ----------
    text : str
        The quoted text the box encloses.
    page : int
        Zero-based page index the box is on.
    x0 : float
        Left edge, in page pixels.
    top : float
        Top edge, in page pixels.
    x1 : float
        Right edge, in page pixels.
    bottom : float
        Bottom edge, in page pixels.
    """

    text: str
    page: int
    x0: float
    top: float
    x1: float
    bottom: float


class SourceInfo(BaseModel):
    """Where a citation came from.

    Attributes
    ----------
    type : Literal["widget", "direct retrieval", "web", "artifact"]
        Class of source the citation points at.
    uuid : str or None
        Identifier of the source widget/artifact, when applicable.
    origin : str or None
        Origin app or data source.
    widget_id : str or None
        Identifier of the source widget template/type.
    name : str or None
        Display name of the source.
    description : str or None
        Human-readable description of the source.
    metadata : dict[str, Any]
        Additional source metadata.
    citable : bool
        Whether the source may be surfaced as a clickable citation.
    """

    model_config = ConfigDict(extra="allow")

    type: Literal["widget", "direct retrieval", "web", "artifact"]
    uuid: str | None = None
    origin: str | None = None
    widget_id: str | None = None
    name: str | None = None
    description: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)
    citable: bool = True


class Citation(BaseModel):
    """One source attribution.

    Attributes
    ----------
    id : str
        Unique identifier of the citation.
    source_info : SourceInfo
        Where the cited material came from.
    details : list[dict[str, Any]] or None
        Optional structured detail entries about the citation.
    quote_bounding_boxes : list[list[CitationHighlightBoundingBox]] or None
        Per-quote groups of bounding boxes highlighting the cited text;
        omitted from the wire payload when ``None``.
    """

    id: str
    source_info: SourceInfo
    details: list[dict[str, Any]] | None = None
    quote_bounding_boxes: list[list[CitationHighlightBoundingBox]] | None = None

    @model_serializer(mode="wrap")
    def _drop_empty_bboxes(self, handler):  # type: ignore[no-untyped-def]
        """Omit ``quote_bounding_boxes`` from the wire payload when null."""
        data = handler(self)
        if data.get("quote_bounding_boxes") is None:
            data.pop("quote_bounding_boxes", None)
        return data


class CitationCollection(BaseModel):
    """Payload for the ``copilotCitationCollection`` event.

    Attributes
    ----------
    citations : list[Citation]
        The citations gathered over the run.
    """

    citations: list[Citation]


class CitationCollectionSSE(BaseSSE):
    """A batch of citations emitted at end-of-run.

    Attributes
    ----------
    data : CitationCollection
        The collected citations.
    """

    event: Literal["copilotCitationCollection"] = "copilotCitationCollection"
    data: CitationCollection


SSEEvent = (
    MessageChunkSSE
    | StatusUpdateSSE
    | FunctionCallSSE
    | MessageArtifactSSE
    | CitationCollectionSSE
)


StatusUpdateSSEData.model_rebuild()
