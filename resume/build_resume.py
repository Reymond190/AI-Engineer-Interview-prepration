"""Build resume.pdf from resume.md.

Why this shape: the resume is edited often (new achievements, research), so the
source of truth is a plain Markdown file that diffs cleanly in git. This script
turns it into styled HTML and prints that to PDF with headless Microsoft Edge
(or Chrome), which every Windows machine already has, so no pip installs are
needed.

Markdown subset understood (kept small on purpose, see resume.md):
    # Name                      header block; the next lines are the tagline and contact line
    ## Section                  section heading with a rule underneath
    ### Left || Right           entry row (role | company || dates), right part right-aligned
    - bullet                    bullet point
    other line                  one paragraph per line (skills rows, summary)
    **bold**, [text](url)       inline formatting

Usage (from the project root, any Python 3.10+):
    python resume/build_resume.py                  # resume/resume.md -> resume/resume.pdf
    python resume/build_resume.py --html-only      # just write resume.html (to preview)
    python resume/build_resume.py -o out.pdf
"""

from __future__ import annotations

import argparse
import html
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent

CSS = """
@page { size: A4; margin: 12mm 14mm; }
* { box-sizing: border-box; }
body { font-family: Arial, Helvetica, sans-serif; font-size: 10.2pt; line-height: 1.32;
       color: #222; margin: 0; }
header { text-align: center; margin-bottom: 4pt; }
header h1 { font-size: 22pt; color: #1f3864; margin: 0; letter-spacing: .5pt; }
header .tagline { font-weight: bold; font-size: 10.5pt; margin-top: 1pt; }
header .contact { color: #555; margin-top: 1pt; }
header a { color: inherit; text-decoration: none; }
h2 { font-size: 11.5pt; text-transform: uppercase; color: #2f5496; margin: 9pt 0 4pt;
     padding-bottom: 1pt; border-bottom: 1.2px solid #1f3864; }
.entry { display: flex; justify-content: space-between; gap: 12pt; margin: 5pt 0 2pt; }
.entry .right { font-weight: bold; white-space: nowrap; }
p { margin: 0 0 1pt; }
ul { margin: 0 0 2pt; padding-left: 16pt; }
li { margin-bottom: 1pt; }
a { color: #2f5496; }
"""


def inline(text: str) -> str:
    out = html.escape(text, quote=False)
    out = re.sub(r"\*\*(.+?)\*\*", r"<strong>\1</strong>", out)
    out = re.sub(r"\[(.+?)\]\((\S+?)\)", r'<a href="\2">\1</a>', out)
    return out


def md_to_html(md: str) -> str:
    body: list[str] = []
    in_list = False
    header: list[str] | None = None  # lines collected after "# Name"

    def close_list() -> None:
        nonlocal in_list
        if in_list:
            body.append("</ul>")
            in_list = False

    def flush_header() -> None:
        nonlocal header
        if header is None:
            return
        name, *rest = header
        parts = [f"<h1>{inline(name)}</h1>"]
        classes = ["tagline", "contact"]
        for i, line in enumerate(rest):
            line = " &nbsp;|&nbsp; ".join(inline(p.strip()) for p in line.split("|"))
            parts.append(f'<div class="{classes[min(i, 1)]}">{line}</div>')
        body.append("<header>" + "".join(parts) + "</header>")
        header = None

    for raw in md.splitlines():
        line = raw.rstrip()
        if line.startswith("# "):
            close_list()
            header = [line[2:].strip()]
            continue
        if header is not None:
            if line and not line.startswith("#"):
                header.append(line)
                continue
            flush_header()
        if not line.strip():
            close_list()
        elif line.startswith("## "):
            close_list()
            body.append(f"<h2>{inline(line[3:].strip())}</h2>")
        elif line.startswith("### "):
            close_list()
            left, _, right = line[4:].partition("||")
            body.append(f'<div class="entry"><span>{inline(left.strip())}</span>'
                        f'<span class="right">{inline(right.strip())}</span></div>')
        elif line.lstrip().startswith(("- ", "* ")):
            if not in_list:
                body.append("<ul>")
                in_list = True
            body.append(f"<li>{inline(line.lstrip()[2:].strip())}</li>")
        else:
            close_list()
            body.append(f"<p>{inline(line.strip())}</p>")
    flush_header()
    close_list()

    title = html.escape(md.splitlines()[0].lstrip("# ").strip() if md.strip() else "Resume")
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f"<title>{title} - Resume</title><style>{CSS}</style></head>"
            f"<body>{''.join(body)}</body></html>")


def find_browser() -> str:
    candidates = [
        os.environ.get("RESUME_BROWSER", ""),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        shutil.which("msedge") or "", shutil.which("google-chrome") or "",
        shutil.which("chromium") or "",
    ]
    for c in candidates:
        if c and Path(c).exists():
            return c
    sys.exit("No Edge/Chrome found; set RESUME_BROWSER to a Chromium browser executable.")


def html_to_pdf(html_path: Path, pdf_path: Path) -> None:
    # A throwaway profile dir keeps headless Edge from attaching to a running Edge window.
    with tempfile.TemporaryDirectory() as profile:
        cmd = [find_browser(), "--headless=new", "--disable-gpu", "--no-pdf-header-footer",
               f"--user-data-dir={profile}", f"--print-to-pdf={pdf_path}",
               html_path.resolve().as_uri()]
        subprocess.run(cmd, check=True, capture_output=True, timeout=120)
    if not pdf_path.exists():
        sys.exit(f"Browser did not produce {pdf_path}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", nargs="?", default=HERE / "resume.md", type=Path)
    ap.add_argument("-o", "--output", type=Path, help="PDF path (default: next to source)")
    ap.add_argument("--html-only", action="store_true")
    args = ap.parse_args()

    html_path = args.source.with_suffix(".html")
    html_path.write_text(md_to_html(args.source.read_text(encoding="utf-8")), encoding="utf-8")
    print(f"wrote {html_path}")
    if args.html_only:
        return
    pdf_path = (args.output or args.source.with_suffix(".pdf")).resolve()
    html_to_pdf(html_path, pdf_path)
    print(f"wrote {pdf_path}")


if __name__ == "__main__":
    main()
