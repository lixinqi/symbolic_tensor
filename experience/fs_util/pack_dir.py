
import os
import sys
import seedir

# macOS has /var -> /private/var symlink that causes path resolution issues
_IS_MACOS = sys.platform == 'darwin'


def _resolve_symlink_safe(file_path: str, max_depth: int = 10) -> str:
    """Resolve symlink chain handling macOS /var -> /private/var symlink.

    This function recursively resolves symlinks, handling:
    1. Chains of symlinks (e.g., dump_view -> slice_view -> original storage)
    2. macOS /var -> /private/var mapping

    Args:
        file_path: Path that may be a symlink
        max_depth: Maximum symlink chain depth to prevent infinite loops

    Returns:
        The final resolved path
    """
    depth = 0
    current_path = file_path

    while depth < max_depth:
        # Fix /private/private/var -> /private/var (macOS artifact)
        if _IS_MACOS and '/private/private/' in current_path:
            current_path = current_path.replace('/private/private/', '/private/')

        # Check if current_path is a symlink (even if it "doesn't exist" due to /var vs /private/var)
        is_link = os.path.islink(current_path)
        if not is_link and _IS_MACOS and current_path.startswith('/var/'):
            alt_path = '/private' + current_path
            if os.path.islink(alt_path):
                current_path = alt_path
                is_link = True

        if not is_link:
            # Not a symlink, check if it exists
            if os.path.exists(current_path):
                return current_path
            # Try with /private prefix for macOS /var -> /private/var
            if _IS_MACOS and current_path.startswith('/var/'):
                alt_path = '/private' + current_path
                if os.path.exists(alt_path):
                    return alt_path
            return current_path

        # Read symlink target
        link_target = os.readlink(current_path)

        if os.path.isabs(link_target):
            # Absolute symlink
            current_path = link_target
        else:
            # Relative symlink: resolve relative to current_path's directory
            base_dir = os.path.dirname(current_path)
            current_path = os.path.normpath(os.path.join(base_dir, link_target))

        depth += 1

    # Max depth reached, return what we have
    return current_path


def pack_dir(root_dir: str) -> str:
    # Normalize root_dir to handle macOS /var -> /private/var
    # This ensures symlink resolution is consistent with dump_view
    root_dir = os.path.realpath(root_dir)

    # Get directory structure
    dir_struct = seedir.seedir(root_dir, printout=False)

    # Recursively find all files and their content, resolving symlinks
    file_sections = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            # Resolve symlinks safely to avoid macOS path doubling
            real_path = _resolve_symlink_safe(file_path)
            relative_file_path = os.path.relpath(file_path, root_dir)
            try:
                with open(real_path, "r", encoding="utf-8") as f:
                    file_content = f.read()
            except (UnicodeDecodeError, OSError):
                continue
            file_sections.append(
                f"## File: {relative_file_path}\n"
                f"```\n"
                f"{file_content}\n"
                f"```"
            )

    ret = (
        f"# Directories structure\n"
        f"```\n"
        f"{dir_struct}\n"
        f"```\n\n"
        f"# Files\n\n"
        + "\n\n".join(file_sections)
    )
    return ret


if __name__ == "__main__":
    import tempfile

    print("Running pack_dir tests...\n")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create test structure
        os.makedirs(os.path.join(tmpdir, "subdir"))
        with open(os.path.join(tmpdir, "hello.txt"), "w") as f:
            f.write("hello world")
        with open(os.path.join(tmpdir, "subdir", "nested.txt"), "w") as f:
            f.write("nested content")

        # Create a symlink
        target = os.path.join(tmpdir, "hello.txt")
        link = os.path.join(tmpdir, "link.txt")
        os.symlink(target, link)

        result = pack_dir(tmpdir)

        assert "# Directories structure" in result
        print("  ok: contains directory structure header")

        assert "# Files" in result
        print("  ok: contains files header")

        assert "hello world" in result
        print("  ok: contains file content")

        assert "nested content" in result
        print("  ok: contains nested file content")

        assert "## File: hello.txt" in result
        print("  ok: contains relative file path")

        assert "## File: link.txt" in result
        print("  ok: symlink file listed")

        # Symlink should resolve to same content
        assert result.count("hello world") >= 2  # original + symlink
        print("  ok: symlink resolves to original content")

    print("\nAll tests passed.")
