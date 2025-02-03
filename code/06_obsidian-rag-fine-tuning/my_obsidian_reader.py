import os
import re
from pathlib import Path
from typing import TYPE_CHECKING, Any, List, Tuple

if TYPE_CHECKING:
    from langchain.docstore.document import Document as LCDocument

from llama_index.core.readers.base import BaseReader
from llama_index.readers.file import MarkdownReader
from llama_index.core.schema import Document


class ObsidianReader(BaseReader):
    """
    ObsidianReader for loading data from an Obsidian Vault.
    Each Document includes metadata with file name, folder path,
    extracted wikilink and backlinks.

    Optionally, markdown tasks (e.g. checklist items like "- [ ] task") can be extracted
    into metadata and, if desired, removed from the main text.

    Args:
        input_dir (str): Path to the Obsidian vault.
        extract_tasks (bool): If True, extract tasks from the text and store them in metadata.
                              Default is False.
        remove_tasks_from_text (bool): If True and extract_tasks is True, remove the task
                                       lines from the main document text.
                                       Default is False.
    """

    def __init__(
        self,
        input_dir: str,
        extract_tasks: bool = False,
        remove_tasks_from_text: bool = False,
    ):
        self.input_dir = Path(input_dir)
        self.extract_tasks = extract_tasks
        self.remove_tasks_from_text = remove_tasks_from_text

    def load_data(self, *args: Any, **load_kwargs: Any) -> List[Document]:
        """
        Walks through the vault, loads each markdown file, and adds extra metadata
        (file name, folder path, wikilinks, backlinks, and optionally tasks).
        """
        docs: List[Document] = []
        # This map will hold: {target_note: [linking_note1, linking_note2, ...]}
        backlinks_map = {}

        for dirpath, dirnames, filenames in os.walk(self.input_dir):
            # Skip hidden directories.
            dirnames[:] = [d for d in dirnames if not d.startswith(".")]
            for filename in filenames:
                if filename.endswith(".md"):
                    filepath = os.path.join(dirpath, filename)
                    md_docs = MarkdownReader().load_data(Path(filepath))
                    for i, doc in enumerate(md_docs):
                        file_path_obj = Path(filepath)
                        note_name = file_path_obj.stem
                        doc.metadata["file_name"] = file_path_obj.name
                        doc.metadata["folder_path"] = str(file_path_obj.parent)
                        doc.metadata["folder_name"] = str(
                            file_path_obj.parent.relative_to(self.input_dir)
                        )
                        doc.metadata["note_name"] = note_name
                        wikilinks = self._extract_wikilinks(doc.text)
                        doc.metadata["wikilinks"] = wikilinks
                        # For each wikilink found in this document, record a backlink from this note.
                        for link in wikilinks:
                            # Each link is expected to match a note name (without .md)
                            backlinks_map.setdefault(link, []).append(note_name)

                        # Optionally, extract tasks from the text.
                        if self.extract_tasks:
                            tasks, cleaned_text = self._extract_tasks(doc.text)
                            doc.metadata["tasks"] = tasks
                            if self.remove_tasks_from_text:
                                md_docs[i] = Document(
                                    text=cleaned_text, metadata=doc.metadata
                                )
                    docs.extend(md_docs)

        # Now that we have processed all files, assign backlinks metadata.
        for doc in docs:
            note_name = doc.metadata.get("note_name")
            # If no backlinks exist for this note, default to an empty list.
            doc.metadata["backlinks"] = backlinks_map.get(note_name, [])
        return docs

    def load_langchain_documents(self, **load_kwargs: Any) -> List["LCDocument"]:
        """
        Loads data in the LangChain document format.
        """
        docs = self.load_data(**load_kwargs)
        return [d.to_langchain_format() for d in docs]

    def _extract_wikilinks(self, text: str) -> List[str]:
        """
        Extracts Obsidian wikilinks from the given text.

        Matches patterns like:
          - [[Note Name]]
          - [[Note Name|Alias]]

        Returns a list of unique wikilink targets (aliases are ignored).
        """
        pattern = r"\[\[([^\]]+)\]\]"
        matches = re.findall(pattern, text)
        links = []
        for match in matches:
            # If a pipe is present (e.g. [[Note|Alias]]), take only the part before it.
            target = match.split("|")[0].strip()
            links.append(target)
        return list(set(links))

    def _extract_tasks(self, text: str) -> Tuple[List[str], str]:
        """
        Extracts markdown tasks from the text.

        A task is expected to be a checklist item in markdown, for example:
            - [ ] Do something
            - [x] Completed task

        Returns a tuple:
            (list of task strings, text with task lines removed if removal is enabled).
        """
        # This regex matches lines starting with '-' or '*' followed by a checkbox.
        task_pattern = re.compile(
            r"^\s*[-*]\s*\[\s*(?:x|X| )\s*\]\s*(.*)$", re.MULTILINE
        )
        tasks = task_pattern.findall(text)
        cleaned_text = text
        if self.remove_tasks_from_text:
            cleaned_text = task_pattern.sub("", text)
        return tasks, cleaned_text
