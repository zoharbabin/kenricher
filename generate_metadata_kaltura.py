#!/usr/bin/env python
"""
generate_metadata_kaltura.py - Process Kaltura media entries to generate and update metadata using OpenAI.
"""

import argparse
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, TimeoutError
from enum import Enum
from typing import List, Optional, Dict, Any

import requests
from dotenv import load_dotenv
from rich.console import Console
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.theme import Theme
from rich.prompt import Confirm
from rich.table import Table
from pydantic import BaseModel

from openai import OpenAI
from KalturaClient import KalturaClient
from KalturaClient.Plugins.Core import (
    KalturaMediaEntryFilter,
    KalturaFilterPager,
    KalturaMediaType,
    KalturaMediaEntry
)
from KalturaClient.Plugins.Caption import KalturaCaptionAssetFilter
from KalturaClient.exceptions import KalturaException

from kaltura_utils import KalturaClientsManager, create_custom_logger

# --- Environment Variables Loading and Validation ---
script_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(script_dir, ".env")
if os.path.exists(env_path):
    load_dotenv(dotenv_path=env_path, override=True)
else:
    print("No .env file found in the script directory; using global environment variables.")

required_vars = ["KALTURA_PARTNER_ID", "KALTURA_ADMIN_SECRET", "KALTURA_SERVICE_URL", "OPENAI_API_KEY"]
missing_vars = [var for var in required_vars if not os.getenv(var)]
if missing_vars:
    sys.exit(f"Error: Missing required environment variables: {', '.join(missing_vars)}")

# --- Pydantic Models for API Responses ---
class ChunkSummary(BaseModel):
    summary: str


class CombinedSummary(BaseModel):
    combined_summary: str


class VideoMetadata(BaseModel):
    title: str
    tags: List[str]
    description: str


# --- OpenAI Model Enumeration ---
class OpenAIModel(Enum):
    GPT_4O = {"name": "gpt-4o-2024-08-06", "context_window": 128000}
    GPT_4O_MINI = {"name": "gpt-4o-mini-2024-07-18", "context_window": 128000}
    O1_PREVIEW = {"name": "o1-preview-2024-09-12", "context_window": 128000}

    def get_name(self) -> str:
        return self.value["name"]

    def get_context_window(self) -> int:
        return self.value["context_window"]


# --- Retrieve Configuration Constants ---
KALTURA_PARTNER_ID = os.getenv("KALTURA_PARTNER_ID")
KALTURA_ADMIN_SECRET = os.getenv("KALTURA_ADMIN_SECRET")
KALTURA_SERVICE_URL = os.getenv("KALTURA_SERVICE_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

SELECTED_MODEL = OpenAIModel.GPT_4O_MINI
MODEL_NAME = SELECTED_MODEL.get_name()
TEMPERATURE = 0.2

# --- Setup Rich Console and Logger ---
custom_theme = Theme({
    "info": "cyan",
    "warning": "yellow",
    "error": "bold red",
    "done": "bold green",
    "title": "bold magenta",
    "step": "bold cyan"
})
console = Console(theme=custom_theme)
logger = create_custom_logger(logging.getLogger(__name__))


# --- Kaltura Metadata Generator Class ---
class KalturaMetadataGenerator:
    """
    Class to process Kaltura media entries and generate metadata using OpenAI.
    """

    def __init__(self, kaltura_client: KalturaClient, openai_client: OpenAI,
                 model_name: str, temperature: float, simulation: bool = False,
                 debug: bool = False, auto_update: bool = False) -> None:
        """
        Initialize the metadata generator.
        """
        self.client = kaltura_client
        self.openai_client = openai_client
        self.model_name = model_name
        self.temperature = temperature
        self.simulation = simulation
        self.debug = debug
        self.auto_update = auto_update
        self.session = requests.Session()
        if self.debug:
            logger.debug("KalturaMetadataGenerator initialized with simulation=%s, debug=%s, auto_update=%s",
                         simulation, debug, auto_update)
        else:
            logger.info("KalturaMetadataGenerator initialized.")

    def fetch_entry(self, entry_id: str) -> Optional[KalturaMediaEntry]:
        """
        Fetch an entry by its ID with progress updates.
        """
        logger.info("Starting fetch_entry for ID: %s", entry_id)
        start_time = time.time()
        try:
            result = self.client.media.get(entry_id)
            elapsed = time.time() - start_time
            logger.info("Fetched entry for ID: %s in %.2f seconds", entry_id, elapsed)
            return result
        except Exception as e:
            logger.error("Error fetching entry for ID %s: %s", entry_id, e)
            return None

    def get_all_video_entries(self, id_in: Optional[str] = None,
                              category_ancestor_id_in: Optional[str] = None) -> List[KalturaMediaEntry]:
        """
        Fetch all video/audio entries from Kaltura using optional filters.
        """
        pager = KalturaFilterPager()
        pager.pageSize = 500
        pager.pageIndex = 1
        entry_filter = KalturaMediaEntryFilter()
        entry_filter.mediaTypeIn = f"{KalturaMediaType.VIDEO},{KalturaMediaType.AUDIO}"
        if id_in:
            entry_filter.idIn = id_in
        if category_ancestor_id_in:
            entry_filter.categoryAncestorIdIn = category_ancestor_id_in
        all_entries: List[KalturaMediaEntry] = []
        while True:
            try:
                result = self.client.media.list(entry_filter, pager)
                if self.debug:
                    logger.debug("Fetched %d entries in page %d", len(result.objects), pager.pageIndex)
            except KalturaException as e:
                logger.error("Failed to fetch entries: %s", e)
                break
            if not result.objects:
                break
            all_entries.extend(result.objects)
            if len(result.objects) < pager.pageSize:
                break
            pager.pageIndex += 1
        return all_entries

    def get_captions_url(self, entry_id: str) -> Optional[str]:
        """
        Retrieve the URL of the caption asset for a given entry.
        """
        caption_filter = KalturaCaptionAssetFilter()
        caption_filter.entryIdEqual = entry_id
        try:
            result = self.client.caption.captionAsset.list(caption_filter)
        except KalturaException as e:
            logger.error("Error fetching caption assets for entry %s: %s", entry_id, e)
            return None
        if not result.objects:
            return None
        asset_id = result.objects[0].id
        try:
            url = self.client.caption.captionAsset.getUrl(asset_id)
            if self.debug:
                logger.debug("Caption URL for asset %s: %s", asset_id, url)
            return url
        except KalturaException as e:
            logger.error("Error fetching caption URL for asset %s: %s", asset_id, e)
            return None

    def download_captions(self, url: str) -> str:
        """
        Download captions from the given URL using a persistent session.
        """
        try:
            with console.status("[step]Downloading captions...[/step]", spinner="dots"):
                response = self.session.get(url, timeout=10)
                response.raise_for_status()
                if self.debug:
                    logger.debug("Downloaded captions (first 100 chars): %s", response.text[:100])
                return response.text
        except requests.RequestException as e:
            logger.error("Failed to download captions from %s: %s", url, e)
            console.print(f"[warning]Failed to download captions from {url}: {e}[/warning]")
            return ""

    @staticmethod
    def chunk_text(text: str, context_window: int, reserved_tokens: int = 1000) -> List[str]:
        """
        Break text into chunks for processing.
        """
        max_chunk_size = context_window - reserved_tokens
        return [text[i:i + max_chunk_size] for i in range(0, len(text), max_chunk_size)]

    def call_openai_api(self, messages: List[Dict[str, str]], response_format: Any) -> Optional[Any]:
        """
        Helper method to call the OpenAI API.
        """
        if self.debug:
            logger.debug("Calling OpenAI API with messages: %s", messages)
        try:
            response = self.openai_client.beta.chat.completions.parse(
                model=self.model_name,
                messages=messages,
                temperature=self.temperature,
                response_format=response_format
            )
            parsed = response.choices[0].message.parsed
            if self.debug:
                logger.debug("OpenAI API response parsed: %s", parsed)
            return parsed
        except Exception as e:
            logger.error("OpenAI API call failed: %s", e)
            return None

    def summarize_chunk(self, chunk: str) -> str:
        """
        Summarize a transcript chunk using OpenAI.
        """
        system_msg = (
            "You are a helpful assistant that creates a concise summary of the given transcript segment. "
            "Return the summary as JSON only, following this schema:\n"
            '{"summary": "<concise summary>"}'
        )
        user_msg = (
            "Summarize the following segment in under 250 words. Return only JSON with a single key 'summary':\n\n"
            f"{chunk}"
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        result = self.call_openai_api(messages, ChunkSummary)
        return result.summary if result and hasattr(result, "summary") else ""

    def combine_summaries(self, summaries: List[str]) -> str:
        """
        Combine multiple summaries into a cohesive summary.
        """
        combined_text = "\n".join(f"- {s}" for s in summaries)
        system_msg = (
            "You are a helpful assistant that combines partial summaries into one cohesive summary. "
            "Return the final result as JSON only, following this schema:\n"
            '{"combined_summary": "<cohesive summary>"}'
        )
        user_msg = (
            "Combine these partial summaries into one cohesive summary under 300 words. Return only JSON:\n\n"
            f"{combined_text}"
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        result = self.call_openai_api(messages, CombinedSummary)
        return result.combined_summary if result and hasattr(result, "combined_summary") else ""

    def generate_metadata(self, final_summary: str, original_title: str,
                          feedback: Optional[str] = None) -> Optional[VideoMetadata]:
        """
        Generate metadata (title, tags, description) from the final summary and original title.
        """
        if feedback:
            user_msg = (
                f"Original title: '{original_title}'\n\n"
                f"Summary:\n\n{final_summary}\n\n"
                f"Feedback: {feedback}\n\n"
                "Generate new metadata in JSON only, addressing the feedback:\n"
                "Keys: title, tags (as JSON array), description."
            )
        else:
            user_msg = (
                f"Original title: '{original_title}'\n\n"
                f"Summary:\n\n{final_summary}\n\n"
                "Generate metadata strictly as JSON (no extra text) with keys: title, tags (as JSON array), description."
            )
        system_msg = (
            "You are a video content metadata generator. Return your answer in JSON only with the following schema:\n"
            '{\n'
            '  "title": "string",\n'
            '  "tags": ["string", ... up to 10],\n'
            '  "description": "string"\n'
            '}\n\n'
            "Formatting rules:\n"
            "- Title: Concise, reflect main subjects discussed in the transcript, and incorporate key structure elements from the original title.\n"
            "   - Dates in the title should be placed between the primary topic and any colons or details.\n"
            "   - Use title case for the new title.\n"
            "- NEVER use quotes in the title.\n"
            "- Tags: Up to 15 relevant keywords, comma-separated, all lowercase if possible, no duplicates.\n"
            "- Description: Detailed, factual description that accurately reflects the video content.\n"
            "   - Focus on specific details and key points actually discussed in the video.\n"
            "   - Avoid generic statements or unneeded superlatives.\n"
            "No extra text, just the JSON object."
        )
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg}
        ]
        result = self.call_openai_api(messages, VideoMetadata)
        return result if result and hasattr(result, "title") else None

    def update_entry_metadata(self, entry_id: str, title: str, description: str, tags: List[str]) -> bool:
        """
        Update the metadata of a Kaltura entry.
        """
        try:
            entry = KalturaMediaEntry()
            entry.name = title
            entry.description = description
            entry.tags = ",".join(tags) if tags else ""
            self.client.media.update(entry_id, entry)
            return True
        except KalturaException as e:
            logger.error("Error updating entry %s: %s", entry_id, e)
            console.print(f"[error]Failed to update entry {entry_id}: {e}[/error]")
            return False

    def process_captions(self, captions: str, context_window: int) -> str:
        """
        Process captions: chunk, summarize, and combine into a final summary.
        """
        chunks = self.chunk_text(captions, context_window)
        console.print(f"[info]Total chunks: {len(chunks)}[/info]")
        if not chunks:
            return ""
        summaries = []
        max_workers = min(20, len(chunks))
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console, transient=True
        ) as progress:
            task = progress.add_task("Summarizing chunks...", total=len(chunks))
            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_chunk = {executor.submit(self.summarize_chunk, chunk): chunk for chunk in chunks}
                for future in as_completed(future_to_chunk):
                    summary = future.result()
                    if summary:
                        summaries.append(summary)
                    progress.update(task, advance=1)
        if not summaries:
            logger.error("No summaries were generated from chunks.")
            return ""
        return self.combine_summaries(summaries)

    def process_entry(self, entry: Dict[str, Any]) -> None:
        """
        Process a single Kaltura entry to generate and optionally update metadata.
        """
        # If entry is not a dict, convert to dict with keys 'id' and 'name'
        if isinstance(entry, dict):
            entry_id = entry.get("id")
            entry_name = entry.get("name")
            idx = entry.get("idx", 0)
            total = entry.get("total", 0)
        else:
            entry_id = getattr(entry, "id", None)
            entry_name = getattr(entry, "name", None)
            idx = 0
            total = 0

        console.print(Panel.fit(f"Processing entry {idx}/{total}: [bold]{entry_id}[/bold]", style="title"))
        if not entry_id or not entry_name:
            logger.error("Entry missing 'id' or 'name': %s", entry)
            console.print("[warning]Entry missing 'id' or 'name'. Skipping.[/warning]")
            return

        caption_url = self.get_captions_url(entry_id)
        if not caption_url:
            console.print("[warning]No captions found.[/warning]")
            return
        captions = self.download_captions(caption_url)
        if not captions.strip():
            console.print("[warning]Empty captions.[/warning]")
            return
        context_window = SELECTED_MODEL.get_context_window()
        final_summary = self.process_captions(captions, context_window)
        if not final_summary.strip():
            console.print("[warning]No summary generated, possibly empty content.[/warning]")
            return

        feedback: Optional[str] = None
        while True:
            metadata = self.generate_metadata(final_summary, entry_name, feedback)
            if not metadata:
                console.print("[warning]Failed to generate metadata.[/warning]")
                return

            table = Table(
                title=f"Generated Metadata for entryId: {entry_id}",
                show_header=True,
                header_style="bold magenta"
            )
            table.add_column("Field", style="cyan")
            table.add_column("Value", style="green")
            table.add_row("Title", metadata.title)
            table.add_row("Description", metadata.description)
            table.add_row("Tags", ",".join(metadata.tags) if metadata.tags else "")
            console.print(table)

            if self.simulation:
                console.print("[info]Simulation mode ON — skipping update.[/info]")
                break
            elif self.auto_update or Confirm.ask("Would you like to update the entry with this metadata?"):
                if self.update_entry_metadata(entry_id, metadata.title, metadata.description, metadata.tags):
                    console.print("[done]Entry updated successfully![/done]")
                else:
                    console.print("[error]Failed to update entry.[/error]")
                break
            else:
                feedback = console.input("[step]Please provide feedback for regeneration (or press Enter to skip):[/step] ")
                if not feedback.strip():
                    console.print("[info]Skipping entry...[/info]")
                    break
                console.print("[info]Regenerating metadata with feedback...[/info]")

    def process_entries(self, entries: List[Dict[str, Any]]) -> None:
        """
        Process multiple Kaltura entries.
        """
        for entry in entries:
            self.process_entry(entry)
        console.print("[done]All entries processed.[/done]")


def main() -> None:
    """
    Main execution flow.
    """
    parser = argparse.ArgumentParser(description="Generate metadata on Kaltura entries.")
    parser.add_argument("--entry_ids", help="Comma-separated list of entry IDs to process.")
    parser.add_argument("--category_ids", help="Comma-separated list of parent category IDs to process entries from.")
    parser.add_argument("--simulate", action="store_true", help="Simulation mode: do not update entries.")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode for verbose output.")
    parser.add_argument("--privileges", default="", help="Specify privileges for the KS (e.g., 'disableentitlement').")
    parser.add_argument("--auto-update", action="store_true", help="Automatically update entries without confirmation.")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.setLevel(logging.DEBUG)
        logger.debug("Debug mode enabled.")
    else:
        logging.getLogger().setLevel(logging.INFO)
        logger.setLevel(logging.INFO)

    # Ensure that --entry_ids and --category_ids are not both provided.
    if args.entry_ids and args.category_ids:
        sys.exit("Error: Cannot provide both --entry_ids and --category_ids. Please choose one.")

    console.print("[bold blue]Starting Kaltura Metadata Generation Tool...[/bold blue]")
    console.print("[info]Initializing Kaltura clients...[/info]")
    source_client_params = {
        "service_url": KALTURA_SERVICE_URL,
        "partner_id": KALTURA_PARTNER_ID,
        "partner_secret": KALTURA_ADMIN_SECRET
    }
    dest_client_params = source_client_params.copy()
    clients_manager = KalturaClientsManager(
        should_log=args.debug,
        kaltura_user_id="admin",
        session_duration=86400,
        session_privileges=args.privileges,
        source_client_params=source_client_params,
        dest_client_params=dest_client_params
    )
    console.print("[info]Kaltura clients initialized.[/info]")

    console.print("[info]Initializing OpenAI client...[/info]")
    openai_client = OpenAI(api_key=OPENAI_API_KEY)
    console.print("[info]OpenAI client initialized.[/info]")

    metadata_generator = KalturaMetadataGenerator(
        kaltura_client=clients_manager.source_client,
        openai_client=openai_client,
        model_name=MODEL_NAME,
        temperature=TEMPERATURE,
        simulation=args.simulate,
        debug=args.debug,
        auto_update=args.auto_update
    )

    # Determine which entries to process.
    if args.entry_ids:
        console.print("[info]Fetching specified entry IDs...[/info]")
        entries = metadata_generator.get_all_video_entries(id_in=args.entry_ids)
    elif args.category_ids:
        console.print("[info]Fetching entries for specified category IDs...[/info]")
        entries = metadata_generator.get_all_video_entries(category_ancestor_id_in=args.category_ids)
    else:
        with console.status("[bold green]Fetching all video entries...[/bold green]", spinner="dots"):
            entries = metadata_generator.get_all_video_entries()
    console.print(f"[info]Total entries fetched: {len(entries)}[/info]")

    # Convert entries (if not already dicts) to dicts for uniform processing.
    if entries and not isinstance(entries[0], dict):
        entries = [{"id": entry.id, "name": entry.name, "idx": idx, "total": len(entries)}
                   for idx, entry in enumerate(entries, start=1)]

    metadata_generator.process_entries(entries)


if __name__ == "__main__":
    main()
