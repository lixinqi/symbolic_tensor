import os
import seedir
def pack_dir(root_dir: str) -> str:
    dir_struct = seedir.seedir(root_dir, printout=False)
    file_sections = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            file_path = os.path.join(dirpath, filename)
            real_path = os.path.realpath(file_path)
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
