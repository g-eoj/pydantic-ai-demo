import asyncio
import markdownify
import os
import requests_cache
import re
import time

from langchain_text_splitters import RecursiveCharacterTextSplitter
from mcp.server.fastmcp import FastMCP
from playwright.async_api import async_playwright
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.openai import OpenAIModel, OpenAIModelSettings
from pydantic_ai.providers.openai import OpenAIProvider
from transformers import AutoTokenizer
from typing import List


# define output types for note taking agents
class Note(BaseModel):
    text: str | None
    is_relevant: bool


# convert a web page or PDF to markdown that is easily consumable by an LLM
async def get_md(path: str) -> str:
    md = ""
    async with async_playwright() as playwright:
        browser = await playwright.firefox.launch(
            firefox_user_prefs={
                "pdfjs.disabled": False,
                "browser.download.open_pdf_attachments_inline": True,
                "browser.link.open_newwindow": 1,
            }
        )
        page = await browser.new_page()
        await page.goto(path, wait_until="commit")
        time.sleep(3)
        for frame in page.frames:
            try:
                # try loading the pdf viewer
                content = await frame.inner_html("id=viewer", timeout=1000)
            except Exception:
                content = await frame.page.inner_html("body")
            md += (
                markdownify.markdownify(
                    content,
                    strip=["a"],
                    heading_style="ATX",
                    table_infer_header=True,
                )
                + "\n\n"
            )
        await browser.close()
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md


async def run_note_agent(chunk: str, model: OpenAIModel, query: str) -> Note:
    instructions = (
        f"Look for information relevant to '{query}' in this chunk of markdown:\n\n{chunk}\n\n"
        "Now, only using the chunk of markdown, make a note of any relevant information. "
        "If you don't find any relevant informtation, then don't make a note."
    )
    agent = Agent(
        model,
        output_type=Note,
        instructions=instructions,
    )
    run = await agent.run()
    return run.output


async def run_summary_agent(model: OpenAIModel, notes: List[Note], query: str) -> Note:
    notes_text = [note.text for note in notes]
    instructions = (
        f"Combine these notes to help answer '{query}'. "
        "If none of the notes are relevant, then don't make a note. "
        "Check if the notes have contradictory information. "
        f"If they do, explain the contradictions:\n\n{notes_text}"
    )
    agent = Agent(
        model,
        output_type=Note,
        instructions=instructions,
    )
    run = await agent.run()
    return run.output



# configure model
model = OpenAIModel(
    "Qwen/Qwen3-8B-FP8",
    provider=OpenAIProvider(
        api_key=os.getenv("LLM_API_KEY", ""),
        base_url="http://localhost:8000/v1",
    ),
    settings=OpenAIModelSettings(
        extra_body={"chat_template_kwargs": {"enable_thinking": False}}
    ),
)

# split text based on token count
tokenizer = AutoTokenizer.from_pretrained(model.model_name)
split = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    separators=["\n#+", "\n\n", "\n", " ", ""],
    is_separator_regex=True,
    tokenizer=tokenizer,
    chunk_overlap=10000,
    chunk_size=20000,
).split_text

# cache search results
session = requests_cache.CachedSession(
    cache_name="google_search_cache",
    allowable_methods=["POST"],
)

# initialize MCP server
mcp = FastMCP("Web Access")

# add tools
@mcp.tool(title="Search")
def search(query: str) -> str:
    """Search the web for links."""
    url = "https://google.serper.dev/search"
    result = session.post(
        url=url,
        data={"q": query, "num": 5},
        headers={"X-API-KEY": os.getenv("SERPER_API_TOKEN")},
    )
    result.raise_for_status()
    result = result.json()["organic"]
    text = "\n"
    for r in result:
        text += f"- [{r['title']}]({r['link']})\n\tSnippet: {r.get('snippet', '')}\n"
    return text


@mcp.tool(title="Search Papers")
def search_papers(query: str) -> str:
    """Search the web for papers."""
    url = "https://google.serper.dev/scholar"
    result = session.post(
        url=url,
        data={"q": query, "num": 5},
        headers={"X-API-KEY": os.getenv("SERPER_API_TOKEN")},
    )
    result.raise_for_status()
    result = result.json()["organic"]
    text = "\n"
    for r in result:
        pdf_url = r.get("pdfUrl", None)
        html_url = r.get("htmlUrl", None)
        link_url = r.get("link", None)
        link = pdf_url or html_url or link_url
        text += f"- [{r['title']}]({link}): {r.get('snippet', '')}...\n\tPublication Information: {r.get('publicationInfo', '')}\n"
    return text


@mcp.tool(title="Read")
async def read(query: str, url: str) -> Note:
    """Converts a web page or PDF to markdown, then answers your query about the markdown content."""
    md = await get_md(url)
    notes = []
    tasks = [run_note_agent(chunk, model, query) for chunk in split(md)[:5]]
    for task in asyncio.as_completed(tasks):
        note = await task
        if note.is_relevant:
            notes.append(note)
    if len(notes) > 1:
        return await run_summary_agent(model, notes, query)
    if len(notes) == 1:
        return notes[0]
    return Note(text=None, is_relevant=False)


if __name__ == "__main__":
    mcp.run()
