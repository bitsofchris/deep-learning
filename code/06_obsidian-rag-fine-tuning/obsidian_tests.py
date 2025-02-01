import os
import pytest
from pathlib import Path
from typing import List

from llama_index.core.schema import Document

from my_obsidian_reader import ObsidianReader
from moc import build_notes_map, get_moc_and_linked_notes


# Helper function to create a markdown file in the given directory.
def create_markdown_file(directory: Path, file_name: str, content: str) -> Path:
    file_path = directory / file_name
    file_path.write_text(content, encoding="utf-8")
    return file_path


###########################################
# Feature 1: File Metadata (file_name & folder_path)
###########################################


def test_file_metadata(tmp_path: Path):
    """
    Test that a simple markdown file returns a document with correct file metadata.
    """
    content = "This is a simple document."
    create_markdown_file(tmp_path, "test.md", content)

    reader = ObsidianReader(input_dir=str(tmp_path))
    docs = reader.load_data()
    assert len(docs) == 1
    doc = docs[0]
    # Check that file metadata was added
    assert doc.metadata.get("file_name") == "test.md"
    # folder_path should match the temporary directory
    assert Path(doc.metadata.get("folder_path")).resolve() == tmp_path.resolve()


def test_file_metadata_nested(tmp_path: Path):
    """
    Test that a markdown file in a subdirectory gets the correct folder path metadata.
    """
    subdir = tmp_path / "subfolder"
    subdir.mkdir()
    content = "Nested file content."
    create_markdown_file(subdir, "nested.md", content)

    reader = ObsidianReader(input_dir=str(tmp_path))
    docs = reader.load_data()
    # Expect one document loaded from the nested directory
    assert len(docs) == 1
    doc = docs[0]
    assert doc.metadata.get("file_name") == "nested.md"
    assert Path(doc.metadata.get("folder_path")).resolve() == subdir.resolve()


###########################################
# Feature 2: Wikilink Extraction
###########################################


def test_wikilink_extraction(tmp_path: Path):
    """
    Test that wikilinks (including alias links) are extracted and stored in metadata.
    """
    content = "Refer to [[NoteOne]] and [[NoteTwo|Alias]] for more details."
    create_markdown_file(tmp_path, "wikilinks.md", content)

    reader = ObsidianReader(input_dir=str(tmp_path))
    docs = reader.load_data()
    assert len(docs) == 1
    doc = docs[0]
    wikilinks: List[str] = doc.metadata.get("wikilinks", [])
    # Order does not matter; both targets should be present.
    assert set(wikilinks) == {"NoteOne", "NoteTwo"}


def test_wikilink_extraction_duplicates(tmp_path: Path):
    """
    Test that duplicate wikilinks (with or without aliases) are only stored once.
    """
    content = "See [[Note]] and also [[Note|Alias]]."
    create_markdown_file(tmp_path, "dup_wikilinks.md", content)

    reader = ObsidianReader(input_dir=str(tmp_path))
    docs = reader.load_data()
    assert len(docs) == 1
    doc = docs[0]
    wikilinks: List[str] = doc.metadata.get("wikilinks", [])
    # Only one unique wikilink should be present.
    assert set(wikilinks) == {"Note"}


###########################################
# Feature 3: Tasks Extraction (without removal)
###########################################


def test_tasks_extraction(tmp_path: Path):
    """
    Test that markdown tasks are correctly extracted into metadata when removal is disabled.
    """
    content = (
        "Task list:\n"
        "- [ ] Task A\n"
        "Some intervening text\n"
        "- [x] Task B\n"
        "More text follows."
    )
    create_markdown_file(tmp_path, "tasks.md", content)

    reader = ObsidianReader(
        input_dir=str(tmp_path), extract_tasks=True, remove_tasks_from_text=False
    )
    docs = reader.load_data()
    assert len(docs) == 1
    doc = docs[0]
    tasks: List[str] = doc.metadata.get("tasks", [])
    # Check that tasks are extracted.
    assert "Task A" in tasks
    assert "Task B" in tasks
    # Since removal is disabled, the original text should still contain the task lines.
    assert "- [ ] Task A" in doc.text
    assert "- [x] Task B" in doc.text


###########################################
# Feature 4: Tasks Removal from Text
###########################################


def test_remove_tasks_from_text(tmp_path: Path):
    """
    Test that when removal is enabled, task lines are removed from the document text.
    """
    content = "Intro text\n" "- [ ] Task 1\n" "- [x] Task 2\n" "Conclusion text"
    create_markdown_file(tmp_path, "tasks_removed.md", content)

    reader = ObsidianReader(
        input_dir=str(tmp_path), extract_tasks=True, remove_tasks_from_text=True
    )
    docs = reader.load_data()
    assert len(docs) == 1
    doc = docs[0]
    tasks: List[str] = doc.metadata.get("tasks", [])
    assert "Task 1" in tasks
    assert "Task 2" in tasks
    # Ensure the task lines have been removed from the main text.
    assert "- [ ] Task 1" not in doc.text
    assert "- [x] Task 2" not in doc.text
    # Ensure that non-task text is still present.
    assert "Intro text" in doc.text
    assert "Conclusion text" in doc.text


def create_doc(text: str, file_name: str = None, wikilinks=None) -> Document:
    """
    Helper to create a Document with given text, file_name, and wikilinks.
    If wikilinks is not provided, an empty list is used.
    """
    if wikilinks is None:
        wikilinks = []
    metadata = {}
    if file_name:
        metadata["file_name"] = file_name
    metadata["wikilinks"] = wikilinks
    return Document(text=text, metadata=metadata)


###########################################
# Tests for build_notes_map
###########################################


def test_build_notes_map_normal():
    """
    Test that build_notes_map groups documents by their file name.
    Two documents with the same file name should be grouped under one key.
    """
    # Create three documents: two from "Note1.md" and one from "Note2.md"
    doc1 = create_doc("Content A", "Note1.md")
    doc2 = create_doc("Content B", "Note1.md")
    doc3 = create_doc("Content C", "Note2.md")
    docs = [doc1, doc2, doc3]

    notes_map = build_notes_map(docs)

    assert "Note1.md" in notes_map
    assert "Note2.md" in notes_map
    assert len(notes_map["Note1.md"]) == 2
    assert len(notes_map["Note2.md"]) == 1


def test_build_notes_map_missing_file_name():
    """
    Test that documents without a file_name in metadata are ignored in the resulting map.
    """
    # Create one document with file_name and one without.
    doc1 = create_doc("Content A", "Note1.md")
    doc2 = create_doc("Content B")  # No file_name provided
    docs = [doc1, doc2]

    notes_map = build_notes_map(docs)

    # Only the document with a file_name should appear.
    assert "Note1.md" in notes_map
    assert len(notes_map) == 1
    assert len(notes_map["Note1.md"]) == 1


###########################################
# Tests for get_moc_and_linked_notes
###########################################


def test_get_moc_and_linked_notes_valid():
    """
    Test that given an MOC note with valid wikilinks, the function returns
    a combined list of documents that includes the MOC and its linked notes.
    """
    # Create a MOC document that links to "Note1" and "Note2".
    moc_doc = create_doc("MOC Content", "MOC.md", wikilinks=["Note1", "Note2"])
    note1_doc = create_doc("Note1 Content", "Note1.md")
    note2_doc = create_doc("Note2 Content", "Note2.md")

    # Build the notes map as expected by get_moc_and_linked_notes.
    notes_map = {"MOC.md": [moc_doc], "Note1.md": [note1_doc], "Note2.md": [note2_doc]}

    # Provide a path to the MOC note (the function uses the basename)
    combined_docs = get_moc_and_linked_notes("some/path/MOC.md", notes_map)

    # Expect the combined list to contain documents from MOC, Note1, and Note2.
    file_names = [doc.metadata["file_name"] for doc in combined_docs]
    assert "MOC.md" in file_names
    assert "Note1.md" in file_names
    assert "Note2.md" in file_names
    assert len(combined_docs) == 3


def test_get_moc_and_linked_notes_missing_link(capsys):
    """
    Test that if the MOC's wikilinks include a note that does not exist
    in the notes_map, only the MOC documents are returned and a warning is printed.
    """
    # Create a MOC document that links to a non-existent note.
    moc_doc = create_doc("MOC Content", "MOC.md", wikilinks=["NonExistingNote"])
    notes_map = {
        "MOC.md": [moc_doc]
        # "NonExistingNote.md" is intentionally missing.
    }

    combined_docs = get_moc_and_linked_notes("MOC.md", notes_map)

    # The returned list should only contain the MOC document.
    file_names = [doc.metadata["file_name"] for doc in combined_docs]
    assert file_names == ["MOC.md"]

    # Capture and check the warning message.
    captured = capsys.readouterr().out
    assert "Warning: Linked note 'NonExistingNote.md' not found" in captured
