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


def test_get_moc_and_linked_notes_depth0():
    """
    Test that with depth=0, only the MOC note is returned,
    even if the MOC document contains wikilinks.
    """
    # Create a MOC document that links to Note1.
    moc_doc = create_doc("MOC content", "MOC.md", wikilinks=["Note1"])
    note1_doc = create_doc("Note1 content", "Note1.md")

    # Build the notes map with both the MOC and Note1.
    notes_map = {"MOC.md": [moc_doc], "Note1.md": [note1_doc]}

    # With depth=0, we expect only the MOC document.
    result = get_moc_and_linked_notes("some/path/MOC.md", notes_map, depth=0)

    assert len(result) == 1
    assert result[0].metadata["file_name"] == "MOC.md"


def test_get_moc_and_linked_notes_depth1():
    """
    Test that with depth=1, the MOC note and its directly linked note(s) are returned.
    """
    # Create a MOC document that links to Note1.
    moc_doc = create_doc("MOC content", "MOC.md", wikilinks=["Note1"])
    note1_doc = create_doc("Note1 content", "Note1.md")

    # Build the notes map with both the MOC and Note1.
    notes_map = {"MOC.md": [moc_doc], "Note1.md": [note1_doc]}

    # With depth=1, expect the MOC document and the Note1 document.
    result = get_moc_and_linked_notes("any/path/MOC.md", notes_map, depth=1)
    file_names = {doc.metadata["file_name"] for doc in result}

    assert file_names == {"MOC.md", "Note1.md"}
    assert len(result) == 2


def test_get_moc_and_linked_notes_depth2():
    """
    Test that with depth=2, the function returns the MOC note,
    its directly linked note, and the note linked from that note.
    """
    # Create a MOC document that links to Note1.
    moc_doc = create_doc("MOC content", "MOC.md", wikilinks=["Note1"])
    # Create Note1 that links to Note2.
    note1_doc = create_doc("Note1 content", "Note1.md", wikilinks=["Note2"])
    # Create Note2 with no further links.
    note2_doc = create_doc("Note2 content", "Note2.md")

    # Build the notes map.
    notes_map = {"MOC.md": [moc_doc], "Note1.md": [note1_doc], "Note2.md": [note2_doc]}

    # With depth=2, we expect all three notes.
    result = get_moc_and_linked_notes("any/path/MOC.md", notes_map, depth=2)
    file_names = {doc.metadata["file_name"] for doc in result}

    assert file_names == {"MOC.md", "Note1.md", "Note2.md"}
    assert len(result) == 3


def get_doc_by_note_name(docs, note_name: str):
    """
    Utility function to return the first document with the specified note name.
    """
    for doc in docs:
        if doc.metadata.get("note_name") == note_name:
            return doc
    return None


def test_single_backlink(tmp_path: Path):
    """
    Test a simple case where one note (A.md) links to another (B.md).

    Expected behavior:
      - Note A should have no backlinks.
      - Note B should have a backlink from A.
    """
    # Create two markdown files:
    # A.md links to B, while B.md contains no wikilinks.
    create_markdown_file(tmp_path, "A.md", "This is note A linking to [[B]].")
    create_markdown_file(tmp_path, "B.md", "This is note B with no links.")

    reader = ObsidianReader(input_dir=str(tmp_path))
    docs = reader.load_data()

    doc_a = get_doc_by_note_name(docs, "A")
    doc_b = get_doc_by_note_name(docs, "B")

    # Verify that doc_a exists and has no backlinks.
    assert doc_a is not None
    assert doc_a.metadata.get("backlinks") == []

    # Verify that doc_b exists and has a backlink from A.
    assert doc_b is not None
    assert doc_b.metadata.get("backlinks") == ["A"]


def test_multiple_backlinks(tmp_path: Path):
    """
    Test a scenario with multiple notes linking to a single note.

    Create three files:
      - A.md: links to B and C.
      - B.md: links to C.
      - C.md: contains no wikilinks.

    Expected behavior:
      - Note A should have no backlinks.
      - Note B should have a backlink from A.
      - Note C should have backlinks from both A and B.
    """
    create_markdown_file(tmp_path, "A.md", "Linking to [[B]] and [[C]].")
    create_markdown_file(tmp_path, "B.md", "Linking to [[C]].")
    create_markdown_file(tmp_path, "C.md", "No links here.")

    reader = ObsidianReader(input_dir=str(tmp_path))
    docs = reader.load_data()

    doc_a = get_doc_by_note_name(docs, "A")
    doc_b = get_doc_by_note_name(docs, "B")
    doc_c = get_doc_by_note_name(docs, "C")

    # Verify note A has no backlinks.
    assert doc_a is not None
    assert doc_a.metadata.get("backlinks") == []

    # Verify note B has a backlink from A.
    assert doc_b is not None
    assert doc_b.metadata.get("backlinks") == ["A"]

    # Note C should have backlinks from A and B.
    # Since file processing order might vary, we compare as sets.
    assert doc_c is not None
    backlinks_c = doc_c.metadata.get("backlinks")
    assert set(backlinks_c) == {"A", "B"}


def test_no_links(tmp_path: Path):
    """
    Test that a note with no outgoing links gets an empty backlinks list.
    """
    create_markdown_file(tmp_path, "A.md", "This is a note with no links.")
    reader = ObsidianReader(input_dir=str(tmp_path))
    docs = reader.load_data()

    doc_a = get_doc_by_note_name(docs, "A")
    assert doc_a is not None
    # Since no note links to A, its backlinks should be an empty list.
    assert doc_a.metadata.get("backlinks") == []
