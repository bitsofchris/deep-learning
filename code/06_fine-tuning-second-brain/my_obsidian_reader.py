"""Obsidian reader class.
Parse Obsidian vault markdown files into Documents, with metadata and header-based splitting.
"""

import os
from pathlib import Path
from typing import TYPE_CHECKING, Any, List
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
    """

    def __init__(self, input_dir: str):
        """Initialize with input directory."""
        self.input_dir = Path(input_dir)

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

    def load_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
        """Load data from the input directory with metadata.

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

                    # Load content using MarkdownReader
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
