from __future__ import annotations

import html
import base64
import mimetypes
import re
from pathlib import Path


HERE = Path(__file__).resolve().parent
SOURCE = HERE / "manuscript_full_draft_cn_v2_04.md"
OUTPUT = HERE / "manuscript_full_draft_cn_v2_04.html"


def inline_markup(text: str) -> str:
    placeholders: list[tuple[str, str]] = []

    def stash(pattern: str, repl: str, value: str) -> str:
        token = f"@@HTML{len(placeholders)}@@"
        placeholders.append((token, repl))
        return token

    image_pattern = re.compile(r"!\[([^\]]*)\]\(([^)]+)\)")

    def image_repl(match: re.Match[str]) -> str:
        alt = html.escape(match.group(1))
        src = html.escape(resolve_image_src(match.group(2)))
        return stash("", f'<figure><img src="{src}" alt="{alt}"><figcaption>{alt}</figcaption></figure>', "")

    text = image_pattern.sub(image_repl, text)
    escaped = html.escape(text)
    escaped = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", escaped)
    escaped = re.sub(r"`([^`]+)`", r"<code>\1</code>", escaped)
    for token, value in placeholders:
        escaped = escaped.replace(token, value)
    return escaped


def resolve_image_src(src: str) -> str:
    if re.match(r"^[a-z]+:", src, flags=re.IGNORECASE):
        path = Path(src)
    else:
        path = HERE / src
    if path.exists() and path.is_file():
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        encoded = base64.b64encode(path.read_bytes()).decode("ascii")
        return f"data:{mime};base64,{encoded}"
    return src.replace("\\", "/")


def is_table_separator(line: str) -> bool:
    cells = [cell.strip() for cell in line.strip().strip("|").split("|")]
    return bool(cells) and all(re.fullmatch(r":?-{3,}:?", cell or "") for cell in cells)


def split_table_row(line: str) -> list[str]:
    return [cell.strip() for cell in line.strip().strip("|").split("|")]


def render_table(lines: list[str]) -> str:
    header = split_table_row(lines[0])
    body_lines = lines[2:] if len(lines) > 1 and is_table_separator(lines[1]) else lines[1:]
    parts = ["<table>", "<thead><tr>"]
    parts.extend(f"<th>{inline_markup(cell)}</th>" for cell in header)
    parts.append("</tr></thead>")
    parts.append("<tbody>")
    for line in body_lines:
        cells = split_table_row(line)
        parts.append("<tr>")
        parts.extend(f"<td>{inline_markup(cell)}</td>" for cell in cells)
        parts.append("</tr>")
    parts.append("</tbody></table>")
    return "\n".join(parts)


def flush_paragraph(paragraph: list[str], out: list[str]) -> None:
    if not paragraph:
        return
    text = " ".join(line.strip() for line in paragraph).strip()
    if text:
        out.append(f"<p>{inline_markup(text)}</p>")
    paragraph.clear()


def markdown_to_html(markdown_text: str) -> str:
    lines = markdown_text.splitlines()
    out: list[str] = []
    paragraph: list[str] = []
    i = 0
    in_math = False
    math_lines: list[str] = []

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        if stripped == "$$":
            flush_paragraph(paragraph, out)
            if in_math:
                out.append('<div class="math-block">\\[')
                out.append("\n".join(math_lines))
                out.append("\\]</div>")
                math_lines.clear()
                in_math = False
            else:
                in_math = True
            i += 1
            continue

        if in_math:
            math_lines.append(line)
            i += 1
            continue

        if not stripped:
            flush_paragraph(paragraph, out)
            i += 1
            continue

        if stripped.startswith("|") and "|" in stripped[1:]:
            flush_paragraph(paragraph, out)
            table_lines = [line]
            i += 1
            while i < len(lines) and lines[i].strip().startswith("|"):
                table_lines.append(lines[i])
                i += 1
            out.append(render_table(table_lines))
            continue

        heading_match = re.match(r"^(#{1,6})\s+(.+)$", stripped)
        if heading_match:
            flush_paragraph(paragraph, out)
            level = len(heading_match.group(1))
            text = inline_markup(heading_match.group(2))
            out.append(f"<h{level}>{text}</h{level}>")
            i += 1
            continue

        if stripped.startswith("!["):
            flush_paragraph(paragraph, out)
            out.append(inline_markup(stripped))
            i += 1
            continue

        paragraph.append(line)
        i += 1

    flush_paragraph(paragraph, out)
    return "\n".join(out)


def build_page(body: str, title: str) -> str:
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)}</title>
  <script>
    window.MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
      }},
      svg: {{ fontCache: 'global' }}
    }};
  </script>
  <script async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-svg.js"></script>
  <style>
    :root {{
      color-scheme: light;
      --text: #202124;
      --muted: #5f6368;
      --line: #dadce0;
      --soft: #f7f8fa;
      --accent: #315f8c;
    }}
    body {{
      margin: 0;
      font-family: "Aptos", "Segoe UI", Arial, sans-serif;
      color: var(--text);
      background: #ffffff;
      line-height: 1.65;
    }}
    main {{
      max-width: 980px;
      margin: 0 auto;
      padding: 48px 28px 72px;
    }}
    h1, h2, h3, h4 {{
      line-height: 1.25;
      margin: 2.1em 0 0.65em;
      color: #17212b;
    }}
    h1 {{
      font-size: 2rem;
      margin-top: 0;
      padding-bottom: 0.55em;
      border-bottom: 2px solid var(--line);
    }}
    h2 {{
      font-size: 1.45rem;
      border-bottom: 1px solid var(--line);
      padding-bottom: 0.3em;
    }}
    h3 {{ font-size: 1.17rem; }}
    p {{ margin: 0.8em 0; }}
    strong {{ font-weight: 700; }}
    code {{
      font-family: "Cascadia Mono", Consolas, monospace;
      background: var(--soft);
      padding: 0.1em 0.3em;
      border-radius: 4px;
    }}
    figure {{
      margin: 2rem 0;
      padding: 0;
    }}
    img {{
      display: block;
      max-width: 100%;
      height: auto;
      border: 1px solid var(--line);
      background: white;
    }}
    figcaption {{
      margin-top: 0.5rem;
      color: var(--muted);
      font-size: 0.92rem;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      margin: 1.4rem 0 1.8rem;
      font-size: 0.92rem;
    }}
    th, td {{
      border: 1px solid var(--line);
      padding: 0.55rem 0.65rem;
      vertical-align: top;
    }}
    th {{
      background: #eef3f8;
      color: #1f3346;
      text-align: left;
      font-weight: 700;
    }}
    tr:nth-child(even) td {{ background: #fafbfc; }}
    .math-block {{
      overflow-x: auto;
      margin: 1.2rem 0 1.5rem;
      padding: 0.4rem 1rem;
      text-align: center;
      font-size: 1.08rem;
    }}
    @media print {{
      main {{ max-width: none; padding: 24px; }}
      img {{ page-break-inside: avoid; }}
      table {{ page-break-inside: auto; }}
      tr {{ page-break-inside: avoid; }}
    }}
  </style>
</head>
<body>
<main>
{body}
</main>
</body>
</html>
"""


def main() -> None:
    markdown_text = SOURCE.read_text(encoding="utf-8-sig")
    body = markdown_to_html(markdown_text)
    title = "Sedative-hypnotics and fall-related reporting signals in older adults"
    OUTPUT.write_text(build_page(body, title), encoding="utf-8")
    print(f"Wrote {OUTPUT}")


if __name__ == "__main__":
    main()
