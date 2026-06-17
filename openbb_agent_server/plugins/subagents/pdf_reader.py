"""PDF reader subagent."""

from __future__ import annotations

SYSTEM_PROMPT = """\
You are a PDF-reading subagent. The user has uploaded one or more PDFs.

Approach
--------
1. List uploaded files via ``ls /uploads/``.
2. For each relevant PDF, call the ``pdf_extract`` tool to get
   page-keyed text plus bounding boxes for any phrases you cite.
3. Answer the user's question based on the extracted text.
4. ALWAYS emit citations via the ``custom`` channel with shape:

   {"type": "citations", "citations": [
     {"source": "<filename>", "text": "<exact quote>",
      "highlight": {"page_number": <int>, "bounding_box": [x0,y0,x1,y1]}},
   ]}

Never invent quotes or bounding boxes. If the PDF is empty / unreadable,
say so.
"""


class PdfReaderSubAgent:
    """Declare the PDF-reading subagent and its routing metadata.

    This is a static descriptor used by the subagent registry to expose a
    PDF-focused assistant. It carries no behaviour of its own; the runtime
    reads its attributes to decide when to delegate and which tools and
    prompt to use.

    Attributes
    ----------
    name : str
        Registry key for the subagent (``"pdf_reader"``).
    description : str
        Human-readable hint describing when to route to this subagent
        (questions about, summaries of, or citations from uploaded PDFs).
    system_prompt : str
        System prompt instructing the subagent to list uploads, extract
        page-keyed text via ``pdf_extract``, and emit grounded citations.
    tools : tuple of str
        Names of tools the subagent may call (``("pdf_extract",)``).
    model : str or None
        Optional model override; ``None`` means use the default model.
    """

    name = "pdf_reader"
    description = (
        "Use when the user asks about content of an uploaded PDF, or asks "
        "to summarise / quote / cite a document they've attached."
    )
    system_prompt = SYSTEM_PROMPT
    tools: tuple[str, ...] = ("pdf_extract",)
    model: str | None = None
