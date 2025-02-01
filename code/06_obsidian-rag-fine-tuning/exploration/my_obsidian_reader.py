"""Obsidian reader class.
Parse Obsidian vault markdown files into Documents, with metadata, 
wiki links, and task handling.
"""

import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Set, Optional
from datetime import datetime

if TYPE_CHECKING:
    from langchain.docstore.document import Document as LCDocument

from llama_index.core.readers.base import BaseReader
from llama_index.readers.file import MarkdownReader
from llama_index.core.schema import Document


class ObsidianReader(BaseReader):
    """Utilities for loading data from an Obsidian Vault.

    Args:
        input_dir (str): Path to the vault.
        include_tasks (bool): Whether to include task items (- [ ] or - [x]).
            Defaults to True.
        task_type (Optional[str]): If set, adds a 'type' field to metadata
            for documents containing tasks.
    """

    def __init__(
        self,
        input_dir: str,
        include_tasks: bool = True,
        task_type: Optional[str] = None,
    ):
        """Initialize with input directory and task handling options."""
        self.input_dir = Path(input_dir)
        self.include_tasks = include_tasks
        self.task_type = task_type

        # Regex patterns
        self.wiki_link_pattern = re.compile(r"\[\[(.*?)\]\]")
        # Compile with re.MULTILINE so ^ matches at the start of each line.
        self.task_pattern = re.compile(r"^- \[[ xX]\](\s|$)", re.MULTILINE)

    def _get_file_metadata(self, filepath: Path) -> dict:
        """Extract metadata from a file.

        Args:
            filepath (Path): Path to the file

        Returns:
            dict: Dictionary containing file metadata
        """
        stats = filepath.stat()
        return {
            "filename": filepath.name,
            "folder_name": str(filepath.parent.relative_to(self.input_dir)),
            "created_at": datetime.fromtimestamp(stats.st_ctime).isoformat(),
            "modified_at": datetime.fromtimestamp(stats.st_mtime).isoformat(),
        }

    def _extract_wiki_links(self, content: str) -> Set[str]:
        """Extract all wiki links from content.

        Args:
            content (str): The markdown content to parse

        Returns:
            Set[str]: Set of unique wiki links found in the content
        """
        matches = self.wiki_link_pattern.findall(content)
        links = set()
        for match in matches:
            if "|" in match:
                actual_link = match.split("|")[0].strip()
            else:
                actual_link = match.strip()
            links.add(actual_link)
        return links

    def _process_content(self, content: str) -> str:
        """Process content to handle tasks based on configuration.

        This method will remove lines that exactly match a valid task
        (e.g. "- [ ] Some text" or "- [x] Some text") except when they appear
        inside code blocks (fenced with ```) or in table rows (lines starting with "|").

        Args:
            content (str): Raw content to process

        Returns:
            str: Processed content with tasks handled according to settings
        """
        if not self.include_tasks:
            in_code_block = False
            processed_lines = []
            for line in content.splitlines():
                stripped_line = line.lstrip()
                # Toggle code block state.
                if stripped_line.startswith("```"):
                    in_code_block = not in_code_block
                    processed_lines.append(line)
                    continue

                # Do not filter lines inside a code block.
                if in_code_block:
                    processed_lines.append(line)
                    continue

                # Do not filter table rows.
                if stripped_line.startswith("|"):
                    processed_lines.append(line)
                    continue

                # If the (left-stripped) line matches a valid task pattern, skip it.
                if self.task_pattern.match(stripped_line):
                    continue

                # Otherwise, include the line.
                processed_lines.append(line)
            return "\n".join(processed_lines)
        return content

    def _has_tasks(self, content: str) -> bool:
        """Check if content contains task items.

        Args:
            content (str): Content to check

        Returns:
            bool: True if content contains tasks, False otherwise
        """
        return bool(self.task_pattern.search(content))

    def load_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory with metadata and wiki links.

        Returns:
            List[Document]: List of documents with metadata
        """
        docs: List[Document] = []

        for dirpath, dirnames, filenames in os.walk(self.input_dir):
            # Skip hidden directories
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]

            for filename in filenames:
                if filename.endswith(".md"):
                    filepath = Path(os.path.join(dirpath, filename))

                    # Get metadata before loading content
                    metadata = self._get_file_metadata(filepath)

                    # Read the raw content
                    with open(filepath, "r", encoding="utf-8") as f:
                        raw_content = f.read()

                    # Extract wiki links and add to metadata
                    wiki_links = self._extract_wiki_links(raw_content)
                    metadata["wiki_links"] = list(wiki_links)

                    # Check for tasks and update metadata if needed.
                    if self.task_type and self._has_tasks(raw_content):
                        metadata["type"] = "task_type"

                    # Process content (remove tasks if configured)
                    processed_content = self._process_content(raw_content)

                    # Write processed content to temporary file for MarkdownReader
                    # if modified
                    if processed_content != raw_content:
                        temp_path = filepath.with_suffix(".temp.md")
                        temp_path.write_text(processed_content, encoding="utf-8")
                        content = MarkdownReader().load_data(temp_path)
                        temp_path.unlink()  # Clean up temp file
                    else:
                        content = MarkdownReader().load_data(filepath)

                    # Add metadata to each document from the file
                    for doc in content:
                        doc.metadata.update(metadata)
                        docs.append(doc)

        return docs

    def load_langchain_documents(self, **load_kwargs: Any) -> List["LCDocument"]:
        """Load data in LangChain document format."""
        docs = self.load_data(**load_kwargs)
        return [d.to_langchain_format() for d in docs]
