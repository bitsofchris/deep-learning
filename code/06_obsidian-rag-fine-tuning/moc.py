from pathlib import Path
from typing import List, Dict
from llama_index.core.schema import Document

from my_obsidian_reader import ObsidianReader


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
    moc_file: str, notes_map: Dict[str, List[Document]], depth: int = 1
) -> List[Document]:
    """
    Given a path to a MOC file and a hash map of all notes (keyed by file name),
    return a combined list of Document objects for the MOC and its linked notes,
    crawling N levels deep into the wikilinks.

    Depth levels:
      - depth == 0: Only the MOC note.
      - depth == 1: The MOC note and its directly linked notes.
      - depth == 2: The MOC note, its directly linked notes, and the notes linked from them.
      - etc.

    Args:
        moc_file: The file path to the MOC note.
        notes_map: A dictionary mapping file names (e.g., "Note.md") to a list of Document objects.
        depth: The number of link-hops to follow. Default is 1.

    Returns:
        A list of Document objects for the MOC and its linked notes up to the specified depth.

    Raises:
        ValueError: If the MOC file is not found in the notes_map.
    """
    # Get the MOC file name (assumed unique in the vault)
    moc_file_name = Path(moc_file).name
    if moc_file_name not in notes_map:
        raise ValueError(f"MOC file {moc_file_name} not found in the notes map")

    visited = set()  # To avoid processing a note more than once.
    combined_docs: List[Document] = []
    frontier = {moc_file_name}  # Start with the MOC note

    # Process up to 'depth' levels.
    for level in range(depth + 1):
        next_frontier = set()
        for note in frontier:
            if note in visited:
                continue
            visited.add(note)

            if note in notes_map:
                docs = notes_map[note]
                combined_docs.extend(docs)
                # For each document from this note, extract wikilinks and add them to the next frontier.
                for doc in docs:
                    wikilinks = doc.metadata.get("wikilinks", [])
                    for link in wikilinks:
                        # Normalize the link to include the ".md" extension if missing.
                        if not link.endswith(".md"):
                            link = f"{link}.md"
                        next_frontier.add(link)
            else:
                print(f"Warning: Linked note '{note}' not found in notes map.")
        frontier = next_frontier

    return combined_docs


def run():
    pass


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
