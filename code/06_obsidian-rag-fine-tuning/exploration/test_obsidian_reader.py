import pytest
from pathlib import Path
import tempfile
from datetime import datetime
import time
from my_obsidian_reader import ObsidianReader


@pytest.fixture
def temp_obsidian_vault():
    """Create a temporary Obsidian vault with test files."""
    with tempfile.TemporaryDirectory() as temp_dir:
        # Create a simple vault structure
        vault_path = Path(temp_dir)

        # Create a subfolder
        notes_folder = vault_path / "notes"
        notes_folder.mkdir()

        # Create test files with wiki links
        test_files = [
            (
                vault_path / "root_note.md",
                """# Root Note
This is a root level note with a [[link to another note]].
It also has a link with an [[actual link|custom alias]].""",
            ),
            (
                notes_folder / "subfolder_note.md",
                """# Subfolder Note
This links back to the [[root note]] and has a [[new link]].""",
            ),
        ]

        # Write content to files
        for file_path, content in test_files:
            file_path.write_text(content)
            # Sleep briefly to ensure different timestamps
            time.sleep(0.1)

        yield vault_path


def test_metadata_extraction(temp_obsidian_vault):
    """Test that metadata is correctly extracted from files."""
    reader = ObsidianReader(str(temp_obsidian_vault))
    docs = reader.load_data()

    # We should have at least two documents (one from each file)
    assert len(docs) >= 2

    # Check that we have documents from both the root and subfolder
    root_docs = [doc for doc in docs if doc.metadata["filename"] == "root_note.md"]
    subfolder_docs = [
        doc for doc in docs if doc.metadata["filename"] == "subfolder_note.md"
    ]

    assert len(root_docs) > 0, "No documents found from root note"
    assert len(subfolder_docs) > 0, "No documents found from subfolder note"

    # Test metadata for root document
    root_doc = root_docs[0]
    assert root_doc.metadata["filename"] == "root_note.md"
    assert root_doc.metadata["folder_name"] == "."
    assert isinstance(root_doc.metadata["created_at"], str)
    assert isinstance(root_doc.metadata["modified_at"], str)

    # Test metadata for subfolder document
    subfolder_doc = subfolder_docs[0]
    assert subfolder_doc.metadata["filename"] == "subfolder_note.md"
    assert subfolder_doc.metadata["folder_name"] == "notes"
    assert isinstance(subfolder_doc.metadata["created_at"], str)
    assert isinstance(subfolder_doc.metadata["modified_at"], str)


def test_wiki_link_extraction(temp_obsidian_vault):
    """Test that wiki links are correctly extracted."""
    reader = ObsidianReader(str(temp_obsidian_vault))
    docs = reader.load_data()

    # Get documents by filename
    root_doc = next(doc for doc in docs if doc.metadata["filename"] == "root_note.md")
    subfolder_doc = next(
        doc for doc in docs if doc.metadata["filename"] == "subfolder_note.md"
    )

    # Test wiki links in root document
    assert "wiki_links" in root_doc.metadata
    root_links = set(root_doc.metadata["wiki_links"])
    assert "link to another note" in root_links
    assert "actual link" in root_links  # Should extract actual link from alias

    # Test wiki links in subfolder document
    assert "wiki_links" in subfolder_doc.metadata
    subfolder_links = set(subfolder_doc.metadata["wiki_links"])
    assert "root note" in subfolder_links
    assert "new link" in subfolder_links


def test_wiki_link_with_alias(temp_obsidian_vault):
    """Test that wiki links with aliases are correctly processed."""
    # Create a test file with various alias formats
    test_file = temp_obsidian_vault / "aliases.md"
    test_file.write_text(
        """# Aliases Test
[[basic link]]
[[link with spaces]]
[[actual|alias]]
[[complicated link|with spaces in alias]]
[[link with spaces|and alias with spaces]]
"""
    )

    reader = ObsidianReader(str(temp_obsidian_vault))
    docs = reader.load_data()

    # Get the aliases test document
    alias_doc = next(doc for doc in docs if doc.metadata["filename"] == "aliases.md")

    # Check extracted links
    links = set(alias_doc.metadata["wiki_links"])
    assert "basic link" in links
    assert "link with spaces" in links
    assert "actual" in links  # Should only include the actual link, not the alias
    assert "complicated link" in links
    assert "with spaces in alias" not in links  # Aliases should not be included


def test_timestamp_format(temp_obsidian_vault):
    """Test that timestamps are in valid ISO format."""
    reader = ObsidianReader(str(temp_obsidian_vault))
    docs = reader.load_data()

    for doc in docs:
        # Verify created_at is valid ISO format
        try:
            datetime.fromisoformat(doc.metadata["created_at"])
        except ValueError:
            pytest.fail(
                f"Invalid created_at timestamp format: {doc.metadata['created_at']}"
            )

        # Verify modified_at is valid ISO format
        try:
            datetime.fromisoformat(doc.metadata["modified_at"])
        except ValueError:
            pytest.fail(
                f"Invalid modified_at timestamp format: {doc.metadata['modified_at']}"
            )


def test_hidden_folder_exclusion(temp_obsidian_vault):
    """Test that hidden folders are excluded."""
    # Create a hidden folder with a markdown file
    hidden_folder = temp_obsidian_vault / ".hidden"
    hidden_folder.mkdir()
    hidden_file = hidden_folder / "hidden_note.md"
    hidden_file.write_text("# Hidden Note\nThis [[should not|be]] loaded.")

    reader = ObsidianReader(str(temp_obsidian_vault))
    docs = reader.load_data()

    # Verify no documents from hidden folder
    hidden_docs = [doc for doc in docs if ".hidden" in doc.metadata["folder_name"]]
    assert len(hidden_docs) == 0, "Documents from hidden folder should not be included"


def test_empty_vault(temp_obsidian_vault):
    """Test handling of an empty vault."""
    # Remove all files from the temporary vault
    for file_path in temp_obsidian_vault.rglob("*.md"):
        file_path.unlink()

    reader = ObsidianReader(str(temp_obsidian_vault))
    docs = reader.load_data()

    assert len(docs) == 0, "Empty vault should return empty document list"


def test_task_handling(temp_obsidian_vault):
    """Test different task handling configurations."""
    # Create a test file with tasks
    test_file = temp_obsidian_vault / "tasks.md"
    test_file.write_text(
        """# Tasks
Regular content here.
- [ ] Incomplete task
- [x] Complete task
## More Content
- [ ] Another task
Regular content again."""
    )

    # Test with default settings (include tasks)
    reader = ObsidianReader(str(temp_obsidian_vault))
    docs = reader.load_data()
    task_docs = [doc for doc in docs if doc.metadata["filename"] == "tasks.md"]

    # Tasks should be present
    assert any(
        "[ ]" in doc.text for doc in task_docs
    ), "Tasks should be included by default"

    # Test with tasks excluded
    reader_no_tasks = ObsidianReader(str(temp_obsidian_vault), include_tasks=False)
    docs_no_tasks = reader_no_tasks.load_data()
    task_docs_filtered = [
        doc for doc in docs_no_tasks if doc.metadata["filename"] == "tasks.md"
    ]

    # Tasks should be removed
    assert not any(
        "[ ]" in doc.text for doc in task_docs_filtered
    ), "Tasks should be excluded"
    assert not any(
        "[x]" in doc.text for doc in task_docs_filtered
    ), "Completed tasks should be excluded"

    # Regular content should remain
    assert any(
        "Regular content" in doc.text for doc in task_docs_filtered
    ), "Non-task content should remain"


def test_task_type_metadata(temp_obsidian_vault):
    """Test task type metadata assignment."""
    # Create files with and without tasks
    with_tasks = temp_obsidian_vault / "with_tasks.md"
    with_tasks.write_text(
        """# Content With Tasks
- [ ] Task 1
- [x] Task 2"""
    )

    no_tasks = temp_obsidian_vault / "no_tasks.md"
    no_tasks.write_text(
        """# Content Without Tasks
Just regular content here."""
    )

    # Initialize reader with task type
    reader = ObsidianReader(str(temp_obsidian_vault), task_type="task_list")
    docs = reader.load_data()

    # Check documents with tasks
    task_docs = [doc for doc in docs if doc.metadata["filename"] == "with_tasks.md"]
    assert all(
        "type" in doc.metadata for doc in task_docs
    ), "Documents with tasks should have type metadata"
    assert all(
        doc.metadata["type"] == "task_type" for doc in task_docs
    ), "Task type should match configured value"

    # Check documents without tasks
    no_task_docs = [doc for doc in docs if doc.metadata["filename"] == "no_tasks.md"]
    assert all(
        "type" not in doc.metadata for doc in no_task_docs
    ), "Documents without tasks should not have type metadata"


def test_complex_task_filtering(temp_obsidian_vault):
    """Test task filtering with complex content and formatting."""
    test_file = temp_obsidian_vault / "complex_tasks.md"
    test_file.write_text(
        """# Complex Tasks
- [ ] Task with **bold** and *italic*
- [x] Task with [[wiki link]]
- [ ] Task with `inline code`
    - Indented non-task
    - [ ] Indented task
- Regular list item
- [x] Task with > blockquote
- [ ]Malformed task without space
-[ ] Another malformed task
- [ x] Malformed completed task"""
    )

    # Test with tasks excluded
    reader = ObsidianReader(str(temp_obsidian_vault), include_tasks=False)
    docs = reader.load_data()
    content = next(
        doc.text for doc in docs if doc.metadata["filename"] == "complex_tasks.md"
    )

    # Should remove properly formatted tasks only
    assert "Task with **bold**" not in content
    assert "Task with [[wiki link]]" not in content
    assert "Task with `inline code`" not in content
    assert "Regular list item" in content
    assert "- [ ]Malformed task without space" in content
    assert "-[ ] Another malformed task" in content


def test_mixed_content_task_handling(temp_obsidian_vault):
    """Test task handling in documents with mixed content types."""
    test_file = temp_obsidian_vault / "mixed_content.md"
    test_file.write_text(
        """# Mixed Content
Regular paragraph here.

- [ ] Task 1
Some content between tasks
- [x] Task 2

## Subheader
- [ ] Task in subheader
> Blockquote with task inside:
> - [ ] Task in blockquote

```markdown
- [ ] Task in code block
```

| Table Header |
|-------------|
| - [ ] Task in table |"""
    )

    # Test with tasks excluded
    reader = ObsidianReader(str(temp_obsidian_vault), include_tasks=False)
    docs = reader.load_data()

    content = next(
        doc.text for doc in docs if doc.metadata["filename"] == "mixed_content.md"
    )

    # Should keep structure while removing tasks
    assert "Regular paragraph here." in content
    assert "Some content between tasks" in content
    assert "Blockquote with task inside:" in content
    assert "- [ ] Task 1" not in content
    assert "- [x] Task 2" not in content
    # Tasks in code blocks and tables should be preserved
    assert "- [ ] Task in code block" in content
    assert "- [ ] Task in table" in content


def test_edge_case_task_patterns(temp_obsidian_vault):
    """Test handling of edge case task patterns."""
    test_file = temp_obsidian_vault / "edge_cases.md"
    test_file.write_text(
        """# Edge Cases
- [] Not a task (no space)
- [)] Invalid character
- [ ] Valid task
- [x] Valid completed task
- [X] Valid completed task (capital X)
-[ ] No space after dash
-    [ ] Extra spaces after dash
- [ ]No space after bracket
- [  ] Extra space in brackets
- [ x] Space before x
- [x ] Space after x"""
    )

    # Test task detection and filtering
    reader = ObsidianReader(str(temp_obsidian_vault), include_tasks=False)
    docs = reader.load_data()
    content = next(
        doc.text for doc in docs if doc.metadata["filename"] == "edge_cases.md"
    )

    # These should be removed (valid task syntax)
    assert "- [ ] Valid task" not in content
    assert "- [x] Valid completed task" not in content
    assert "- [X] Valid completed task (capital X)" not in content

    # These should remain (invalid task syntax)
    assert "- [] Not a task (no space)" in content
    assert "- [)] Invalid character" in content
    assert "-[ ] No space after dash" in content
    assert "-    [ ] Extra spaces after dash" in content
    assert "- [ ]No space after bracket" in content
    assert "- [  ] Extra space in brackets" in content


def test_header_based_splitting(temp_obsidian_vault):
    """Test that documents are correctly split based on headers."""
    # Create a test file with multiple headers
    test_file = temp_obsidian_vault / "headers.md"
    test_file.write_text(
        """# Main Header
This is the main content.
[[link1]]

## Section 1
This is section 1 content.
[[link2]]

### Subsection 1.1
Subsection content here.
[[link3]]

## Section 2
This is section 2 content.
[[link4]]

# Another Main Header
Final section content.
[[link5]]"""
    )

    reader = ObsidianReader(str(temp_obsidian_vault))
    docs = reader.load_data()

    # Get documents from our test file
    header_docs = [doc for doc in docs if doc.metadata["filename"] == "headers.md"]

    # We should have multiple documents due to header splitting
    assert len(header_docs) > 1, "Document should be split based on headers"

    # Each document should contain its header's wiki links
    for doc in header_docs:
        assert (
            "wiki_links" in doc.metadata
        ), "Each split document should have wiki links"

    # Verify some expected content splits
    content_snippets = [doc.text.strip() for doc in header_docs]

    # Find main sections
    main_sections = [
        text for text in content_snippets if "This is the main content" in text
    ]
    section_ones = [
        text for text in content_snippets if "This is section 1 content" in text
    ]
    section_twos = [
        text for text in content_snippets if "This is section 2 content" in text
    ]
    final_sections = [
        text for text in content_snippets if "Final section content" in text
    ]

    assert len(main_sections) > 0, "Should have main header section"
    assert len(section_ones) > 0, "Should have section 1"
    assert len(section_twos) > 0, "Should have section 2"
    assert len(final_sections) > 0, "Should have final section"

    # All split documents should maintain the same file metadata
    for doc in header_docs:
        assert doc.metadata["filename"] == "headers.md"
        assert doc.metadata["folder_name"] == "."
