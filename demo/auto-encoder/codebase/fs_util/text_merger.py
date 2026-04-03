"""TextMerger: pack/unpack a list of (index, coefficient, content) frames into/from a single string.
Used by merge_forward to merge multiple symbolic tensor elements along an axis
into a single text representation.
"""
from typing import List, Tuple
kFrameMarker = "===ST-MERGED-CONTENT===ST-MERGED-CONTENT===ST-MERGED-CONTENT==="
def frame_to_str(frame: Tuple[int, float, str]) -> str:
    """Convert a single (index, coefficient, content) frame to a string block."""
    index, coefficient, content = frame
    indented_content = "\n".join("  " + line for line in content.splitlines())
    return (
        f"{kFrameMarker}\n"
        f"index: {index}\n"
        f"coefficient: {coefficient}\n"
        f"content:\n\n"
        f"{indented_content}"
    )
def pack(frames: List[Tuple[int, float, str]]) -> str:
    """Pack a list of (index, coefficient, content) frames into a single merged string."""
    return "\n".join(frame_to_str(frame) for frame in frames)
def unpack(merged: str) -> List[Tuple[int, float, str]]:
    """Unpack a merged string back into a list of (index, coefficient, content) frames.
    This is the inverse of pack().
    """
    if not merged.strip():
        return []
    parts = merged.split(kFrameMarker)
    frames: List[Tuple[int, float, str]] = []
    for part in parts:
        part = part.strip()
        if not part:
            continue
        lines = part.splitlines()
        index = None
        coefficient = None
        content_lines: List[str] = []
        in_content = False
        content_started = False
        for line in lines:
            if in_content:
                if not content_started:
                    if line.strip() == "":
                        continue
                    content_started = True
                if line.startswith("  "):
                    content_lines.append(line[2:])
                else:
                    content_lines.append(line)
            elif line.startswith("index: "):
                index = int(line[len("index: "):])
            elif line.startswith("coefficient: "):
                coefficient = float(line[len("coefficient: "):])
            elif line.startswith("content:"):
                in_content = True
        if index is not None and coefficient is not None:
            frames.append((index, coefficient, "\n".join(content_lines)))
    return frames
class TextMerger:
    """Stateless class with pack/unpack as class methods."""
    pack = staticmethod(pack)
    unpack = staticmethod(unpack)
