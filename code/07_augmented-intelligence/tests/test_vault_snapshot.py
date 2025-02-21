import os
import yaml
import pytest
from vault_snapshot.snapshot import load_config, file_is_ignored, create_snapshot


@pytest.fixture
def sample_config(tmp_path):
    """
    Create a sample YAML config file with:
      - vault_path: pointing to a temporary vault directory.
      - target_dir: pointing to a temporary snapshot directory.
      - exclude: a list of patterns for markdown files to ignore.
    """
    config_data = {
        "vault_path": str(tmp_path / "vault"),
        "target_dir": str(tmp_path / "snapshot"),
        "exclude": ["ignore*.md", "skip.md"],
    }
    config_file = tmp_path / "config.yaml"
    with open(config_file, "w") as f:
        yaml.dump(config_data, f)
    return config_file, config_data


def test_load_config(sample_config):
    config_file, expected_config = sample_config
    config = load_config(str(config_file))
    assert config["vault_path"] == expected_config["vault_path"]
    assert config["target_dir"] == expected_config["target_dir"]
    assert config["exclude"] == expected_config["exclude"]


def test_file_is_ignored():
    patterns = ["ignore*.md", "skip.md"]
    assert file_is_ignored("ignore_this.md", patterns) is True
    assert file_is_ignored("skip.md", patterns) is True
    assert file_is_ignored("keep.md", patterns) is False
    # Non-markdown file check
    assert file_is_ignored("notes.txt", patterns) is False


@pytest.fixture
def setup_vault(tmp_path):
    """
    Set up a fake vault directory structure:
      - Creates markdown and non-markdown files in the vault root and subdirectories.
      - Some markdown files match exclusion patterns.
    """
    vault_dir = tmp_path / "vault"
    vault_dir.mkdir()
    # In vault root:
    (vault_dir / "keep.md").write_text("Keep this markdown file")
    (vault_dir / "ignore_this.md").write_text("This markdown file should be ignored")
    (vault_dir / "not_markdown.txt").write_text(
        "This is not a markdown file and should not be copied"
    )

    # Create subdirectory with markdown files
    sub_dir = vault_dir / "subfolder"
    sub_dir.mkdir()
    (sub_dir / "sub_keep.md").write_text("Keep this markdown file in subfolder")
    (sub_dir / "skip.md").write_text("This markdown file should be skipped")
    (sub_dir / "note.txt").write_text("Not markdown file")

    return vault_dir


def test_create_snapshot_excludes_files(tmp_path, setup_vault):
    """
    Test that create_snapshot:
      - Copies only markdown files.
      - Excludes markdown files that match the ignore patterns.
    """
    vault_dir = setup_vault
    target_dir = tmp_path / "snapshot"
    config = {
        "vault_path": str(vault_dir),
        "target_dir": str(target_dir),
        "exclude": ["ignore*.md", "skip.md"],
    }
    create_snapshot(config)

    # In vault root: only "keep.md" should be copied.
    assert os.path.exists(target_dir / "keep.md")
    assert not os.path.exists(target_dir / "ignore_this.md")
    # Non-markdown file should not be copied
    assert not os.path.exists(target_dir / "not_markdown.txt")

    # In subfolder: only "sub_keep.md" should be copied.
    subfolder = target_dir / "subfolder"
    assert os.path.exists(subfolder / "sub_keep.md")
    assert not os.path.exists(subfolder / "skip.md")
    assert not os.path.exists(subfolder / "note.txt")


def test_create_snapshot_overwrites_target(tmp_path, setup_vault):
    """
    Test that if the target directory already exists, it is overwritten.
    """
    vault_dir = setup_vault
    target_dir = tmp_path / "snapshot"
    # Pre-create the target directory with a file that should be removed
    target_dir.mkdir()
    (target_dir / "old_file.md").write_text("old content")

    config = {
        "vault_path": str(vault_dir),
        "target_dir": str(target_dir),
        "exclude": [],  # No ignore; copy all markdown files
    }
    create_snapshot(config)

    # The old file should be removed
    assert not os.path.exists(target_dir / "old_file.md")
    # Since there is no exclusion, all markdown files should be copied
    assert os.path.exists(target_dir / "keep.md")
    assert os.path.exists(target_dir / "ignore_this.md")
    subfolder = target_dir / "subfolder"
    assert os.path.exists(subfolder / "sub_keep.md")
    assert os.path.exists(subfolder / "skip.md")


def test_target_directory_same_as_vault(tmp_path, setup_vault):
    """
    Ensure that using the vault path itself as the target directory raises a ValueError.
    """
    vault_dir = setup_vault
    config = {
        "vault_path": str(vault_dir),
        "target_dir": str(vault_dir),  # Same as vault
        "exclude": [],
    }
    with pytest.raises(ValueError, match="Target directory cannot be the same as"):
        create_snapshot(config)


def test_target_directory_inside_vault(tmp_path, setup_vault):
    """
    Ensure that using a subdirectory of the vault as the target directory raises a
    ValueError.
    """
    vault_dir = setup_vault
    target_dir_inside = vault_dir / "snapshot_inside"
    config = {
        "vault_path": str(vault_dir),
        "target_dir": str(target_dir_inside),  # Inside the vault
        "exclude": [],
    }
    with pytest.raises(ValueError, match="Target directory cannot be the same as"):
        create_snapshot(config)
