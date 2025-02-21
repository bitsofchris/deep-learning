import os
import shutil
import fnmatch
import yaml


def load_config(config_path: str) -> dict:
    """
    Load YAML configuration from the given path.

    Expected YAML keys:
      - vault_path: path to the source Obsidian vault.
      - exclude: a list of gitignore-style patterns for markdown files to exclude.
      - target_dir: (optional) default target directory for snapshot.
    """
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def file_is_ignored(filename: str, exclude_patterns) -> bool:
    """
    Check if a filename matches any of the exclusion patterns.

    :param filename: Name of the file (not full path)
    :param exclude_patterns: List of gitignore-style patterns.
    :return: True if the file should be ignored, False otherwise.
    """
    for pattern in exclude_patterns:
        if fnmatch.fnmatch(filename, pattern):
            return True
    return False


def create_snapshot(config: dict, target_dir: str = None):
    """
    Create a snapshot copy of the vault based on the configuration.

    This version:
      - Only copies markdown (.md) files.
      - Applies ignore patterns (if a markdown file matches one of the patterns, it is
       skipped).
      - **Safety check:** Raises an error if the target directory is the vault or is
      inside the vault.
      - Overwrites/deletes any existing content in the target folder.

    Raises:
      - ValueError if the source vault is not specified, does not exist, or if the
       target directory is invalid.
    """
    source_dir = config.get("vault_path")
    if source_dir is None:
        raise ValueError("vault_path not specified in configuration.")

    exclude_patterns = config.get("exclude", [])

    if target_dir is None:
        target_dir = config.get("target_dir")
        if target_dir is None:
            raise ValueError(
                "Target directory not specified in configuration or as argument."
            )

    abs_source = os.path.abspath(source_dir)
    abs_target = os.path.abspath(target_dir)

    # Safety check: Do not allow the target to be the same as or a subdirectory of the
    # vault.
    if abs_target == abs_source or abs_target.startswith(abs_source + os.sep):
        raise ValueError(
            "Target directory cannot be the same as, or inside, the vault directory."
        )

    if not os.path.exists(abs_source):
        raise ValueError(f"Source vault directory {abs_source} does not exist.")

    # Remove the target directory if it already exists
    if os.path.exists(abs_target):
        shutil.rmtree(abs_target)

    # Create the fresh target directory
    os.makedirs(abs_target, exist_ok=True)

    # Walk through the source vault and copy only markdown files
    for root, dirs, files in os.walk(abs_source):
        rel_dir = os.path.relpath(root, abs_source)
        target_subdir = (
            os.path.join(abs_target, rel_dir) if rel_dir != "." else abs_target
        )
        os.makedirs(target_subdir, exist_ok=True)

        for file in files:
            # Only process markdown files
            if not file.endswith(".md"):
                continue
            # Skip file if it matches an ignore pattern
            if file_is_ignored(file, exclude_patterns):
                continue
            src_file = os.path.join(root, file)
            dst_file = os.path.join(target_subdir, file)
            shutil.copy2(src_file, dst_file)
    print(f"Snapshot created successfully in {target_dir}")


def snapshot(target_dir: str = None, config_path: str = "config.yaml"):
    """
    Convenience function that loads the configuration from the YAML file and
    creates a snapshot.

    :param target_dir: Optional target directory (overrides the config file if provided)
    :param config_path: Path to the YAML configuration file.
    """
    config = load_config(config_path)
    create_snapshot(config, target_dir)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create a snapshot copy of an Obsidian vault."
    )
    parser.add_argument(
        "target", nargs="?", default=None, help="Target directory for snapshot copy"
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config YAML file"
    )
    args = parser.parse_args()
    snapshot(args.target, args.config)
