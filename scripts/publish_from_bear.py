#!/usr/bin/env python3
"""
Publish the most recently exported Bear note to the Quarto blog.

Flow (no args required):
- Looks for the newest .md file in a default export folder
  (set by Shortcut), with optional attachments alongside.
- Optimizes images (via macOS `sips`) into posts/images/<slug>/
- Converts Obsidian-style callouts (> [!TYPE] Title) to Quarto callouts.
- Writes posts/<slug>.qmd with YAML front matter.
- Does not run Quarto render.

How to trigger from macOS Shortcuts:
1) Export the focused Bear note as Markdown with attachments into the repo’s
   "_exported_notes/" folder (or set BEAR_EXPORT_DIR env var to another path).
2) Run this script directly (python) and pass the chosen destination directory
   (e.g., "posts" or "tils") either as a positional arg or via --out-dir.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
import contextlib
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from urllib.parse import unquote


# ---- Configuration ----

REPO_ROOT = Path(__file__).resolve().parents[1]
# Destination directory is provided at runtime; default to posts/
# Export directory defaults to repo-local _exported_notes (user saves there via Cmd+Shift+S).
DEFAULT_EXPORT_DIR = Path(
    os.environ.get("BEAR_EXPORT_DIR", str(REPO_ROOT / "_exported_notes"))
).expanduser()
MAX_IMAGE_WIDTH = int(os.environ.get("BQ_MAX_WIDTH", "1600"))
JPEG_QUALITY = int(os.environ.get("BQ_JPEG_QUALITY", "85"))
# Default: do NOT run extra lossless compression; keep current behavior unless opted-in
LOSSLESS_OPTIM = os.environ.get("BQ_LOSSLESS_OPTIM", "0") not in {"0", "false", "False"}


# ---- Utilities ----


def slugify(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[\s/_]+", "-", text)
    text = re.sub(r"[^a-z0-9\-]", "", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text or "post"


def run(cmd: List[str], check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, check=check, text=True, capture_output=True)


def which(cmd: str) -> Optional[str]:
    return shutil.which(cmd)


# ---- Callout conversion ----

CALLTYPE_MAP = {
    "note": "note",
    "tip": "tip",
    "info": "note",
    "important": "important",
    "warning": "warning",
    "caution": "caution",
    "danger": "caution",
    "error": "caution",
}


def convert_callouts_to_quarto(md: str) -> str:
    lines: List[str] = md.splitlines()
    out: List[str] = []
    i = 0
    header_re = re.compile(r"^>\s*\[!([A-Za-z]+)\]([+\-])?\s*(.*)$")

    while i < len(lines):
        line = lines[i]
        m = header_re.match(line)
        if m:
            ctype_raw = m.group(1).lower()
            collapse_mark = m.group(2) or ""
            title = (m.group(3) or "").strip()
            ctype = CALLTYPE_MAP.get(ctype_raw, ctype_raw)
            attrs = [f".callout-{ctype}"]
            if collapse_mark == "+":
                attrs.append('collapse="true"')
            if title:
                attrs.append(f'title="{_escape_attr(title)}"')

            content: List[str] = []
            i += 1
            while i < len(lines):
                l2 = lines[i]
                if l2.startswith("> "):
                    content.append(l2[2:])
                    i += 1
                elif l2 == ">":
                    content.append("")
                    i += 1
                else:
                    break

            # Ensure a blank line before opening the fenced div
            if out and out[-1].strip() != "":
                out.append("")
            out.append(f"::: {{{' '.join(attrs)}}}")
            while content and content[0].strip() == "":
                content.pop(0)
            out.extend(content)
            out.append(":::")
            # And ensure a blank line after closing
            out.append("")
            continue

        out.append(line)
        i += 1

    return "\n".join(out) + ("\n" if md.endswith("\n") else "")


def _escape_attr(text: str) -> str:
    return text.replace("\\", "\\\\").replace('"', r"\"")


# ---- Image handling ----


def find_markdown_images(md: str) -> List[str]:
    paths: List[str] = []
    for m in re.finditer(r"!\[[^\]]*\]\(([^)]+)\)", md):
        paths.append(m.group(1).strip())
    for m in re.finditer(r"<img [^>]*src=\"([^\"]+)\"", md):
        paths.append(m.group(1).strip())
    return list(dict.fromkeys(paths))


def normalize_output_name(filename: str) -> Tuple[str, str]:
    stem = Path(filename).stem
    ext = Path(filename).suffix.lower()
    if ext == ".heic":
        ext = ".jpg"
    if ext == ".jpeg":
        ext = ".jpg"
    safe_stem = slugify(stem)
    return f"{safe_stem}{ext}", ext


def get_image_width(path: Path) -> Optional[int]:
    try:
        out = run(["sips", "-g", "pixelWidth", str(path)], check=False).stdout
    except Exception:
        return None
    for line in (out or "").splitlines():
        if "pixelWidth:" in line:
            try:
                return int(line.split(":", 1)[1].strip())
            except Exception:
                return None
    return None


def convert_image_with_sips(
    src: Path, dst: Path, *, max_width: int, quality: int
) -> None:
    dst_tmp = dst.with_suffix(dst.suffix + ".tmp")
    dst.parent.mkdir(parents=True, exist_ok=True)
    fmt = (
        "jpeg"
        if dst.suffix.lower() in {".jpg", ".jpeg"}
        else ("png" if dst.suffix.lower() == ".png" else None)
    )

    cmds: List[str] = ["sips", str(src)]
    width = get_image_width(src)
    if width and width > max_width:
        cmds += ["--resampleWidth", str(max_width)]
    if fmt == "jpeg":
        cmds += [
            "--setProperty",
            "format",
            "jpeg",
            "--setProperty",
            "formatOptions",
            str(quality),
        ]
    elif fmt == "png":
        cmds += ["--setProperty", "format", "png"]
    cmds += ["--out", str(dst_tmp)]

    try:
        # If max_width <= 0, skip sips and simply copy
        if max_width <= 0 and fmt is None:
            raise RuntimeError("skip-sips-copy")
        if max_width <= 0 and fmt is not None:
            # If format is known but no resize requested, still use sips to normalize format
            run(
                [
                    "sips",
                    str(src),
                    "--setProperty",
                    "format",
                    fmt,
                    "--out",
                    str(dst_tmp),
                ]
            )
            dst_tmp.rename(dst)
        else:
            run(cmds)
            dst_tmp.rename(dst)
    except Exception:
        if dst_tmp.exists():
            with contextlib.suppress(Exception):
                dst_tmp.unlink()
        shutil.copy2(src, dst)


def maybe_lossless_optimize(dst: Path) -> None:
    if not LOSSLESS_OPTIM:
        return
    ext = dst.suffix.lower()
    tmp = dst.with_suffix(dst.suffix + ".opt")
    # PNG: try zopflipng (lossless)
    if ext in {".png"} and which("zopflipng"):
        try:
            run(["zopflipng", "-y", str(dst), str(tmp)])
            tmp.replace(dst)
            return
        except Exception:
            with contextlib.suppress(Exception):
                tmp.unlink()
    # JPEG: try jpegtran (lossless)
    if ext in {".jpg", ".jpeg"} and which("jpegtran"):
        try:
            run(
                [
                    "jpegtran",
                    "-copy",
                    "none",
                    "-optimize",
                    "-progressive",
                    "-outfile",
                    str(tmp),
                    str(dst),
                ]
            )
            tmp.replace(dst)
            return
        except Exception:
            with contextlib.suppress(Exception):
                tmp.unlink()
    # WebP: strip metadata (lossless). Re-encoding is disabled by default to avoid quality changes.
    if ext == ".webp" and which("webpmux"):
        try:
            run(
                [
                    "webpmux",
                    "-strip",
                    "icc",
                    "-strip",
                    "exif",
                    "-strip",
                    "xmp",
                    "-o",
                    str(tmp),
                    str(dst),
                ]
            )
            tmp.replace(dst)
            return
        except Exception:
            with contextlib.suppress(Exception):
                tmp.unlink()
    # Other formats or tools unavailable: no-op


def process_images(
    md: str,
    *,
    assets_dir: Optional[Path],
    dest_dir: Path,
    rel_dir: Path,
    max_width: int,
    quality: int,
) -> Tuple[str, Dict[str, str]]:
    image_links = find_markdown_images(md)
    conversion_map: Dict[str, str] = {}

    for orig in image_links:
        # Work with both the literal path from markdown and a URL-decoded form
        orig_dec = unquote(orig)
        p = Path(orig_dec)
        src_path: Optional[Path] = None

        if p.is_absolute() and p.exists():
            src_path = p
        elif assets_dir is not None:
            # Prefer direct resolution first
            cand = assets_dir / p
            if cand.exists():
                src_path = cand
            else:
                matches = list(assets_dir.rglob(p.name))
                if matches:
                    src_path = matches[0]

        if src_path is None or not src_path.exists():
            print(f"[warn] image not found: {orig}")
            continue

        out_name, _ = normalize_output_name(src_path.name)
        dst_path = dest_dir / out_name
        convert_image_with_sips(
            src_path, dst_path, max_width=max_width, quality=quality
        )
        maybe_lossless_optimize(dst_path)
        # Map both encoded and decoded originals to the new relative path
        new_rel = str(rel_dir / dst_path.name)
        conversion_map[orig] = new_rel
        if orig != orig_dec:
            conversion_map[orig_dec] = new_rel

    new_md = md
    for orig, new in conversion_map.items():
        # Markdown image/link syntax
        new_md = new_md.replace(f"({orig})", f"({new})")
        # HTML <img src="...">
        new_md = re.sub(
            rf"(src=\")\s*{re.escape(orig)}(\")",
            rf"\\1{new}\\2",
            new_md,
        )
    return new_md, conversion_map


# ---- Content sanitation ----


def replace_hrule_triples(md: str) -> str:
    """Replace standalone '---' lines with '***' to avoid YAML misparse.

    Quarto/YAML can sometimes interpret a mid-document '---' as the start of a
    new YAML block. Using '***' preserves a horizontal rule without ambiguity.
    """
    lines = md.splitlines()
    for idx, line in enumerate(lines):
        if line.strip() == "---":
            lines[idx] = "***"
    return "\n".join(lines) + ("\n" if md.endswith("\n") else "")


def normalize_task_lists(md: str) -> str:
    """Ensure task lists are parsed as lists (blank line before first item).

    If a line beginning with "- [ ]" or "- [x]" follows a non-blank paragraph
    without an empty separator line, insert a blank line before it.
    """
    lines = md.splitlines()
    out: List[str] = []
    prev_blank = True
    in_code = False
    fence_re = re.compile(r"^\s*```")
    task_re = re.compile(r"^\s*-\s*\[( |x|X)\]\s+")
    for line in lines:
        if fence_re.match(line):
            in_code = not in_code
        if not in_code and task_re.match(line) and not prev_blank:
            # Insert a blank separator to start a list
            out.append("")
        out.append(line)
        prev_blank = line.strip() == ""
    return "\n".join(out) + ("\n" if md.endswith("\n") else "")


# ---- Cleanup exported notes (unconditional) ----


def clear_export_dir(path: Path) -> None:
    """Remove all contents of the export directory after a successful run.

    This is unconditional: everything inside `path` is deleted. The directory
    itself remains.
    """
    path = path.resolve()
    if not path.exists() or not path.is_dir():
        return
    for child in path.iterdir():
        try:
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink(missing_ok=True)
        except Exception as e:
            print(f"[warn] Failed to remove {child}: {e}")


# ---- Front matter ----


def extract_title(md: str, fallback: str) -> str:
    for line in md.splitlines():
        m = re.match(r"^#\s+(.+)$", line.strip())
        if m:
            return m.group(1).strip()
    return fallback


def extract_tags(md: str) -> List[str]:
    tags = re.findall(r"(?<!\w)#([A-Za-z0-9_\-]+)", md)
    seen = set()
    out: List[str] = []
    for t in tags:
        tl = t.lower()
        if tl not in seen:
            seen.add(tl)
            out.append(tl)
    return out


def build_front_matter(
    *,
    title: str,
    date: str,
    categories: List[str],
    tags: List[str],
    image: Optional[str],
) -> str:
    def esc(s: str) -> str:
        return s.replace("\\", r"\\").replace('"', r"\\\"")

    lines = [
        "---",
        f'title: "{esc(title)}"',
        f"date: {date}",
    ]
    if categories:
        lines.append("categories:")
        for c in categories:
            lines.append(f"  - {c}")
    if tags:
        lines.append("tags:")
        for t in tags:
            lines.append(f"  - {t}")
    if image:
        lines.append(f'image: "{esc(image)}"')
    lines += [
        "format:",
        "  html:",
        "    toc: true",
        "---",
        "",
    ]
    return "\n".join(lines)


# ---- Export discovery ----


@dataclass
class ExportedNote:
    md_path: Path
    assets_dir: Optional[Path]


def find_latest_export(export_root: Path) -> Optional[ExportedNote]:
    export_root.mkdir(parents=True, exist_ok=True)
    md_files = sorted(
        export_root.rglob("*.md"), key=lambda p: p.stat().st_mtime, reverse=True
    )
    if not md_files:
        return None
    md_path = md_files[0]

    # Guess attachments dir: sibling dir with images, or any images nearby
    candidates = []
    if md_path.parent != export_root:
        candidates.append(md_path.parent)
    candidates.append(export_root)

    assets_dir: Optional[Path] = None
    for base in candidates:
        for name in (
            "assets",
            "attachments",
            "images",
            "files",
            f"{md_path.stem} files",
            f"{md_path.stem} attachments",
            # Some exporters (Bear) use a folder with the exact note title
            f"{md_path.stem}",
        ):
            d = base / name
            if d.exists() and any(d.rglob("*.*")):
                assets_dir = d
                break
        if assets_dir:
            break

    return ExportedNote(md_path=md_path, assets_dir=assets_dir)


# ---- Main flow ----


def main() -> int:
    # Optional output directory relative to repo (e.g., posts, tils)
    # Accepted forms:
    #   positional:            publish_from_bear.py tils
    #   long with equals:      --out-dir=tils
    #   long with space:       --out-dir tils
    #   underscore alias:      --out_dir tils | --out_dir=tils
    #   short option:          -o tils | -o=tils
    out_dir_arg: Optional[str] = None
    argv = sys.argv[1:]
    i = 0
    while i < len(argv):
        a = argv[i]
        if a.startswith("--out-dir="):
            out_dir_arg = a.split("=", 1)[1]
        elif a == "--out-dir" and i + 1 < len(argv):
            out_dir_arg = argv[i + 1]
            i += 1
        elif a.startswith("--out_dir="):
            out_dir_arg = a.split("=", 1)[1]
        elif a == "--out_dir" and i + 1 < len(argv):
            out_dir_arg = argv[i + 1]
            i += 1
        elif a.startswith("-o="):
            out_dir_arg = a.split("=", 1)[1]
        elif a == "-o" and i + 1 < len(argv):
            out_dir_arg = argv[i + 1]
            i += 1
        elif not a.startswith("-") and out_dir_arg is None:
            out_dir_arg = a
        i += 1

    out_dir = (REPO_ROOT / (out_dir_arg or "posts")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    exported = find_latest_export(DEFAULT_EXPORT_DIR)
    if not exported:
        print(
            "No exported note found. Ensure your Shortcut exports the current Bear note to "
            f"{DEFAULT_EXPORT_DIR} before running this script.",
            file=sys.stderr,
        )
        return 2

    md_raw = exported.md_path.read_text(encoding="utf-8")
    title = extract_title(md_raw, fallback=exported.md_path.stem)
    date = datetime.now().strftime("%Y-%m-%d")
    slug = slugify(title)

    images_rel_dir = Path("images") / slug
    images_dst_dir = out_dir / images_rel_dir
    images_dst_dir.mkdir(parents=True, exist_ok=True)

    md_img, conv_map = process_images(
        md_raw,
        assets_dir=exported.assets_dir,
        dest_dir=images_dst_dir,
        rel_dir=images_rel_dir,
        max_width=MAX_IMAGE_WIDTH,
        quality=JPEG_QUALITY,
    )

    md_conv = convert_callouts_to_quarto(md_img)
    md_conv = replace_hrule_triples(md_conv)
    md_conv = normalize_task_lists(md_conv)

    all_tags = extract_tags(md_conv)
    hero_image = next(iter(conv_map.values()), None)
    fm = build_front_matter(
        title=title, date=date, categories=[], tags=all_tags, image=hero_image
    )
    qmd_text = fm + md_conv

    qmd_path = out_dir / f"{slug}.qmd"
    qmd_path.write_text(qmd_text, encoding="utf-8")
    print(f"Wrote {qmd_path}")
    if conv_map:
        print(f"Optimized {len(conv_map)} image(s) → {images_dst_dir}")
    # Clean up export directory to prepare for the next run
    clear_export_dir(DEFAULT_EXPORT_DIR)
    print(f"Cleaned export directory: {DEFAULT_EXPORT_DIR}")
    return 0


if __name__ == "__main__":
    main()
