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
