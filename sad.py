#!/usr/bin/env python3
"""
Convert an Obsidian Markdown file (Untitled.md) to a self‑contained HTML file.
All image paths remain relative – run this script from inside your myvault/ directory.
"""

import re
from pathlib import Path
import markdown  # pip install markdown

# ----------------------------------------------------------------------
# 1. Read the original Markdown
# ----------------------------------------------------------------------
INPUT_FILE = "Untitled.md"
OUTPUT_FILE = "Untitled.html"

with open(INPUT_FILE, "r", encoding="utf-8") as f:
    md_content = f.read()

# ----------------------------------------------------------------------
# 2. Convert Obsidian image syntax → standard Markdown image syntax
#    ![[filename]]  →  ![alt](filename)
#    (the filename already contains the subfolder if needed, e.g. Excalidraw/…)
# ----------------------------------------------------------------------
def obsidian_image_to_md(match):
    filename = match.group(1).strip()
    # Remove any possible size suffix like |400, often used in Obsidian
    # e.g. ![[image.png|400]] → just image.png
    if "|" in filename:
        filename = filename.split("|")[0]
    # Return standard MD image, using the file name as alt text
    return f"![{filename}]({filename})"

md_content = re.sub(r"!\[\[(.*?)\]\]", obsidian_image_to_md, md_content)

# ----------------------------------------------------------------------
# 3. Convert (now standard) Markdown to HTML
#    extensions: extra (tables, fenced code), nl2br for line breaks
# ----------------------------------------------------------------------
html_body = markdown.markdown(
    md_content,
    extensions=["extra", "nl2br"],
    output_format="html5",
)

# ----------------------------------------------------------------------
# 4. Build a complete HTML page with MathJax for LaTeX rendering
# ----------------------------------------------------------------------
html_page = f"""<!DOCTYPE html>
<html lang="te">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>ఉన్నత పరిమాణంలో భయానక జ్ఞాపకాలు (సూపర్‌పొజిషన్)</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 800px; margin: auto; padding: 1em; line-height: 1.6; }}
  img {{ max-width: 100%; height: auto; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border: 1px solid #aaa; padding: 0.3em 0.6em; }}
  iframe {{ width: 100%; height: 600px; border: none; }}
  code {{ background: #f4f4f4; padding: 0.1em 0.3em; border-radius: 3px; }}
  pre {{ background: #f4f4f4; padding: 1em; overflow-x: auto; }}
</style>
<script id="MathJax-script" async src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
</head>
<body>
{html_body}
</body>
</html>"""

# ----------------------------------------------------------------------
# 5. Write the output
# ----------------------------------------------------------------------
with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    f.write(html_page)

print(f"✅ Successfully created '{OUTPUT_FILE}'")