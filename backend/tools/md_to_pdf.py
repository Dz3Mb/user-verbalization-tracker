"""Convert one or more Markdown files to academic-style PDFs.

Usage:
    python tools/md_to_pdf.py <input.md> [<input2.md> ...] [--out <dir>]

Each <input.md> produces <input.pdf> next to it (or in --out <dir>).

Implementation: markdown -> styled HTML -> PDF via headless Chromium
(Playwright). No external tool required.
"""

from __future__ import annotations

import sys
import re
import asyncio
import argparse
from pathlib import Path

import markdown


# ---------------------------------------------------------------------------
# Academic CSS: serif body, sans-serif headings, monospace code, A4 margins,
# page numbers, table styling, blockquote, callout-friendly. Designed for a
# clean technical report.
# ---------------------------------------------------------------------------
CSS = r"""
:root {
    --color-text: #1a1a1a;
    --color-muted: #6b6b6b;
    --color-rule: #d4d4d4;
    --color-accent: #1b3a6b;
    --color-code-bg: #f5f5f7;
    --color-code-border: #e0e0e6;
    --color-table-stripe: #fafafa;
    --color-link: #1b3a6b;
}

@page {
    size: A4;
    margin: 18mm 18mm 20mm 18mm;
    @bottom-center {
        content: counter(page) " / " counter(pages);
        font-family: "Charter", "Georgia", serif;
        font-size: 9pt;
        color: #6b6b6b;
    }
    @top-right {
        content: string(doctitle);
        font-family: "Charter", "Georgia", serif;
        font-size: 8.5pt;
        color: #6b6b6b;
    }
}

@page :first {
    @top-right { content: ""; }
    @bottom-center { content: ""; }
}

html, body {
    font-family: "Charter", "Georgia", "Cambria", "Times New Roman", serif;
    font-size: 10.5pt;
    line-height: 1.5;
    color: var(--color-text);
    -webkit-print-color-adjust: exact;
    print-color-adjust: exact;
}

/* --- Cover page --- */

.cover {
    page-break-after: always;
    text-align: center;
    padding-top: 35mm;
}
.cover .eyebrow {
    text-transform: uppercase;
    letter-spacing: 0.2em;
    font-size: 9pt;
    color: var(--color-muted);
    margin-bottom: 1.5em;
}
.cover .title {
    font-family: "Helvetica Neue", "Helvetica", "Arial", sans-serif;
    font-size: 26pt;
    font-weight: 600;
    color: var(--color-accent);
    line-height: 1.2;
    margin: 0 0 0.7em 0;
}
.cover .subtitle {
    font-family: "Helvetica Neue", "Helvetica", "Arial", sans-serif;
    font-size: 13pt;
    color: #2a2a2a;
    font-weight: 400;
    max-width: 140mm;
    margin: 0 auto;
}
.cover .meta {
    margin-top: 30mm;
    font-size: 10pt;
    color: var(--color-muted);
}
.cover .meta strong {
    color: var(--color-text);
    font-weight: 600;
}

/* --- Headings --- */

h1, h2, h3, h4 {
    font-family: "Helvetica Neue", "Helvetica", "Arial", sans-serif;
    color: var(--color-accent);
    font-weight: 600;
    line-height: 1.25;
    page-break-after: avoid;
}
h1 {
    string-set: doctitle content();
    font-size: 19pt;
    margin: 0 0 0.8em 0;
    padding-bottom: 0.3em;
    border-bottom: 1px solid var(--color-rule);
}
h2 {
    font-size: 14.5pt;
    margin: 1.6em 0 0.6em 0;
}
h3 {
    font-size: 12pt;
    margin: 1.3em 0 0.4em 0;
}
h4 {
    font-size: 10.5pt;
    margin: 1.1em 0 0.3em 0;
    color: #2a3a55;
}

/* The very first h1 after the cover acts as the running header anchor. */
h1:first-of-type { string-set: doctitle content(); }

/* --- Paragraphs and lists --- */

p, ul, ol { margin: 0 0 0.8em 0; }
ul, ol { padding-left: 1.4em; }
li { margin-bottom: 0.25em; }
li > p { margin: 0 0 0.4em 0; }

a { color: var(--color-link); text-decoration: none; }
a:hover { text-decoration: underline; }

strong { font-weight: 600; }
em { font-style: italic; }

/* --- Blockquotes --- */

blockquote {
    margin: 0.8em 0;
    padding: 0.5em 1em;
    border-left: 3px solid var(--color-accent);
    background: #f7f9fc;
    color: #333;
    font-style: italic;
}
blockquote p:last-child { margin-bottom: 0; }

/* --- Tables --- */

table {
    border-collapse: collapse;
    margin: 0.8em 0;
    font-size: 9.5pt;
    width: 100%;
    page-break-inside: avoid;
}
th, td {
    border: 1px solid var(--color-rule);
    padding: 5px 9px;
    text-align: left;
    vertical-align: top;
}
th {
    background: var(--color-accent);
    color: #fff;
    font-family: "Helvetica Neue", "Helvetica", "Arial", sans-serif;
    font-weight: 600;
    font-size: 9pt;
}
tbody tr:nth-child(even) { background: var(--color-table-stripe); }

/* --- Code --- */

code, pre, kbd, samp {
    font-family: "JetBrains Mono", "Fira Code", "Consolas", "Menlo", monospace;
    font-size: 9pt;
}
p code, li code, td code, th code {
    background: var(--color-code-bg);
    border: 1px solid var(--color-code-border);
    border-radius: 3px;
    padding: 0.5px 4px;
    font-size: 0.92em;
}
pre {
    background: var(--color-code-bg);
    border: 1px solid var(--color-code-border);
    border-radius: 4px;
    padding: 10px 12px;
    overflow-x: auto;
    margin: 0.8em 0;
    line-height: 1.45;
    page-break-inside: avoid;
    white-space: pre-wrap;
    word-wrap: break-word;
}
pre code {
    background: transparent;
    border: 0;
    padding: 0;
    font-size: 9pt;
}

/* --- Horizontal rules --- */
hr {
    border: 0;
    border-top: 1px solid var(--color-rule);
    margin: 1.5em 0;
}

/* Avoid orphan headings */
h2 + h3, h3 + h4, h2 + p, h3 + p { page-break-before: avoid; }
"""


# ---------------------------------------------------------------------------
# Markdown -> HTML  (with code highlighting, tables, TOC, fenced code, etc.)
# ---------------------------------------------------------------------------

_MD_EXTENSIONS = [
    "extra",          # tables, fenced code, attr_list, footnotes...
    "codehilite",     # Pygments-based syntax highlighting
    "sane_lists",
    "smarty",         # smart quotes/dashes
    "toc",
]
_MD_EXT_CONFIGS = {
    "codehilite": {"guess_lang": False, "noclasses": True},
    "toc": {"title": "Table of contents"},
}


def _slug(text: str) -> str:
    """A simple slug used for file-friendly names."""
    s = re.sub(r"[^\w\s-]", "", text.lower()).strip()
    return re.sub(r"[\s_-]+", "-", s) or "doc"


def _extract_title(md_text: str, fallback: str) -> tuple[str, str]:
    """Find the first H1 and return (title, body_without_that_line).

    Markdown supports `# Title` and underlined Setext style; we only need to
    handle the common `# ` form here.
    """
    lines = md_text.splitlines()
    for i, line in enumerate(lines):
        m = re.match(r"^#\s+(.+?)\s*$", line)
        if m:
            title = m.group(1).strip()
            rest = "\n".join(lines[:i] + lines[i + 1 :])
            return title, rest
    return fallback, md_text


def _wrap_html(title: str, subtitle: str, body_html: str) -> str:
    subtitle_html = (
        f'<div class="subtitle">{subtitle}</div>' if subtitle else ""
    )
    return f"""<!doctype html>
<html lang="fr">
<head>
<meta charset="utf-8" />
<title>{title}</title>
<style>{CSS}</style>
</head>
<body>

<section class="cover">
  <h1 class="title">{title}</h1>
  {subtitle_html}
</section>

<article>
{body_html}
</article>

</body>
</html>
"""


def render_html(md_path: Path) -> tuple[str, str]:
    """Read a Markdown file and return (output filename stem, HTML)."""
    # Files can come in different encodings (UTF-8, UTF-8 BOM, UTF-16 LE/BE
    # when saved by an IDE). Try the common ones in order.
    raw = md_path.read_bytes()
    text: str | None = None
    for encoding in ("utf-8-sig", "utf-16", "utf-16-le", "utf-16-be", "utf-8", "latin-1"):
        try:
            text = raw.decode(encoding)
            break
        except UnicodeDecodeError:
            continue
    if text is None:
        text = raw.decode("utf-8", errors="replace")
    # Strip any leftover BOM
    if text.startswith("\ufeff"):
        text = text[1:]
    title, rest = _extract_title(text, md_path.stem)

    # First non-empty line after the title is treated as the subtitle. If
    # there is none (or only structural content), the cover has no subtitle.
    subtitle = ""
    for line in rest.splitlines():
        s = line.strip()
        if not s:
            continue
        if s.startswith(("#", "<", ">", "-", "|", "`")):
            break
        subtitle = s
        break

    html_body = markdown.markdown(rest, extensions=_MD_EXTENSIONS,
                                  extension_configs=_MD_EXT_CONFIGS)
    full_html = _wrap_html(title, subtitle, html_body)
    return _slug(md_path.stem), full_html


# ---------------------------------------------------------------------------
# HTML -> PDF via Playwright (Chromium)
# ---------------------------------------------------------------------------

async def html_to_pdf(html: str, output_pdf: Path) -> None:
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        browser = await p.chromium.launch()
        page = await browser.new_page()
        await page.set_content(html, wait_until="load")
        await page.emulate_media(media="print")
        await page.pdf(
            path=str(output_pdf),
            format="A4",
            print_background=True,
            margin={"top": "0", "right": "0", "bottom": "0", "left": "0"},
            prefer_css_page_size=True,
        )
        await browser.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", help="Markdown file(s) to convert")
    parser.add_argument("--out", default=None, help="Output directory (default: same as input)")
    args = parser.parse_args()

    out_dir = Path(args.out) if args.out else None
    if out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)

    tasks: list[tuple[Path, Path, str]] = []
    for inp in args.inputs:
        md_path = Path(inp)
        if not md_path.exists():
            print(f"skip (not found): {md_path}", file=sys.stderr)
            continue
        stem, html = render_html(md_path)
        out_path = (out_dir or md_path.parent) / f"{md_path.stem}.pdf"
        tasks.append((md_path, out_path, html))

    if not tasks:
        return 1

    async def run_all():
        for md_path, out_path, html in tasks:
            print(f"-> rendering {md_path}  ->  {out_path}")
            await html_to_pdf(html, out_path)
            print(f"   OK ({out_path.stat().st_size // 1024} KB)")

    asyncio.run(run_all())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
