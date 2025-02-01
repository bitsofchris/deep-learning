from pathlib import Path
from typing import List, Dict
from llama_index.core.schema import Document


def build_notes_map(docs: List[Document]) -> Dict[str, List[Document]]:
    """
    Build a hash map keyed by note file name. Each key maps to the list of Document
    objects loaded for that note.

    Args:
        docs: List of Document objects (possibly from multiple files).

    Returns:
        Dictionary mapping file name (e.g., "Note.md") to list of Document objects.
    """
    notes_map: Dict[str, List[Document]] = {}
    for doc in docs:
        file_name = doc.metadata.get("file_name")
        if file_name:
            notes_map.setdefault(file_name, []).append(doc)
    return notes_map


def get_moc_and_linked_notes(
    moc_file: str, notes_map: Dict[str, List[Document]]
) -> List[Document]:
    """
    Given a path to a MOC file and a hash map of all notes (keyed by file name),
    return a combined list of Document objects for the MOC and its linked notes.

    The function does the following:
      1. Finds the documents for the MOC (by matching the file name).
      2. Collects all wikilinks from those MOC documents.
      3. For each wikilink (ensuring that it ends with ".md"), looks up the corresponding
         documents in the notes map.
      4. Returns the combined list of Document objects.

    Args:
        moc_file: The file path to the MOC note.
        notes_map: A dictionary mapping file names to Document objects.

    Returns:
        List of Document objects for the MOC and its linked notes.
    """
    # Get the MOC file name (assumed unique in the vault)
    moc_file_name = Path(moc_file).name
    if moc_file_name not in notes_map:
        raise ValueError(f"MOC file {moc_file_name} not found in the notes map")

    # Get the Document objects for the MOC note
    moc_docs = notes_map[moc_file_name]

    # Collect all unique wikilinks from the MOC's Document(s)
    linked_note_files = set()
    for doc in moc_docs:
        wikilinks = doc.metadata.get("wikilinks", [])
        for link in wikilinks:
            # Normalize the link to include the .md extension if missing.
            if not link.endswith(".md"):
                link = f"{link}.md"
            linked_note_files.add(link)

    # Build the final list of Document objects:
    # Start with the MOC documents and add the documents for each linked note (if found)
    combined_docs: List[Document] = []
    combined_docs.extend(moc_docs)

    for note_file in linked_note_files:
        if note_file in notes_map:
            combined_docs.extend(notes_map[note_file])
        else:
            print(f"Warning: Linked note '{note_file}' not found in notes map.")

    return combined_docs


# === Example usage ===
if __name__ == "__main__":
    # Suppose you already have an ObsidianReader that uses MarkdownReader.
    # For example:
    # from your_module import ObsidianReader
    #
    # reader = ObsidianReader("/path/to/obsidian/vault", extract_tasks=True, remove_tasks_from_text=True)
    # all_docs = reader.load_data()
    #
    # For demonstration, assume 'all_docs' is already loaded.
    #
    # Build the hash map keyed by file name.
    # notes_map = build_notes_map(all_docs)
    #
    # Given a MOC file path, build the combined list of documents.
    # moc_file_path = "/path/to/obsidian/vault/MOC.md"
    # combined_docs = get_moc_and_linked_notes(moc_file_path, notes_map)
    #
    # Now you can feed 'combined_docs' to your index or reasoning system.
    pass
