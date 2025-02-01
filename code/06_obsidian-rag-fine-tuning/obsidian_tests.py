import os
import pytest
from pathlib import Path
from typing import List

# Adjust this import to where you have defined ObsidianReader.
from my_obsidian_reader import ObsidianReader


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
