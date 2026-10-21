"""Export our short notebook to a self-contained HTML preview, including math.

This small exporter supports the Markdown used by this notebook. It is not a
general replacement for nbconvert. No browser-side scripts or fonts are needed.
"""
import base64
import argparse
from html import escape
from io import BytesIO
from pathlib import Path
import re

import nbformat
from matplotlib import rc_context
from matplotlib.font_manager import FontProperties
from matplotlib.mathtext import math_to_image
from smallprob import __version__


def image(data, mime, css="output-image", alt="Notebook figure"):
    if isinstance(data, str):
        data = data.encode()
    encoded = base64.b64encode(data).decode()
    return f'<img class="{css}" src="data:{mime};base64,{encoded}" alt="{escape(alt, quote=True)}">'


def math_image(expression, inline=False):
    expression = expression.strip().strip("$")
    expression = expression.replace(r"\begin{gathered}", "").replace(r"\end{gathered}", "")
    lines = expression.split(r"\\")
    rendered = []
    for line in lines:
        buffer = BytesIO()
        with rc_context({"savefig.transparent": True}):
            math_to_image("$" + line.strip() + "$", buffer,
                          prop=FontProperties(size=14), format="svg", color="#173e43")
        rendered.append(image(buffer.getvalue(), "image/svg+xml",
                              "inline-math" if inline else "math-image", line.strip()))
    return "".join(rendered) if inline else '<div class="equation">' + "".join(rendered) + '</div>'


def inline_markdown(text):
    parts = re.split(r"(`[^`]+`|\$[^$]+\$|\*\*[^*]+\*\*|\[[^\]]+\]\(https?://[^)]+\))", text)
    result = []
    for part in parts:
        if part.startswith("`"):
            result.append("<code>" + escape(part[1:-1]) + "</code>")
        elif part.startswith("$"):
            result.append(math_image(part, inline=True))
        elif part.startswith("**"):
            result.append("<strong>" + escape(part[2:-2]) + "</strong>")
        elif part.startswith("[") and re.fullmatch(r"\[([^\]]+)\]\((https?://[^)]+)\)", part):
            match = re.fullmatch(r"\[([^\]]+)\]\((https?://[^)]+)\)", part)
            result.append('<a href="' + escape(match[2], quote=True) + '">' + escape(match[1]) + '</a>')
        else:
            result.append(escape(part))
    return "".join(result)


def markdown(source):
    result = []
    for chunk in re.split(r"(\$\$.*?\$\$)", source, flags=re.S):
        if chunk.startswith("$$"):
            result.append(math_image(chunk))
            continue
        for paragraph in chunk.strip().split("\n\n"):
            lines = paragraph.splitlines()
            if lines and lines[0].startswith("#"):
                heading = lines.pop(0)
                level = len(heading) - len(heading.lstrip("#"))
                result.append(f"<h{level}>" + inline_markdown(heading[level:].strip()) + f"</h{level}>")
            if lines:
                result.append("<p>" + inline_markdown(" ".join(lines)) + "</p>")
    return "".join(result)


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('notebook', nargs='?', default='notebooks/smallprob_examples.ipynb')
    parser.add_argument('--output', default=None)
    args = parser.parse_args()
    notebook = nbformat.read(root / args.notebook, as_version=4)
    title = next((c.source.splitlines()[0].lstrip('# ') for c in notebook.cells if c.cell_type == 'markdown'), 'Smallprob examples')
    body = []
    for cell in notebook.cells:
        if cell.cell_type == "markdown":
            body.append(markdown(cell.source))
        elif cell.cell_type == "code":
            body.append('<section class="cell"><div class="cell-label">PYTHON · ' + str(cell.execution_count) + '</div>')
            body.append('<pre class="source"><code>' + escape(cell.source) + '</code></pre>')
            for output in cell.outputs:
                data = output.get("data", {})
                if output.output_type == "error":
                    raise RuntimeError("Refusing to export a notebook with errors")
                if "image/svg+xml" in data:
                    body.append(image(data["image/svg+xml"], "image/svg+xml"))
                elif "image/png" in data:
                    body.append(image(base64.b64decode(data["image/png"]), "image/png"))
                elif "text/latex" in data:
                    body.append(math_image(data["text/latex"]))
                elif output.output_type == "stream":
                    body.append('<pre class="result">' + escape(output.text) + '</pre>')
                elif "text/plain" in data:
                    body.append('<pre class="result">' + escape(data["text/plain"]) + '</pre>')
            body.append('</section>')
    html = '''<!doctype html><html lang="en"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>NOTEBOOK_TITLE</title>
<style>
:root{color-scheme:light}*{box-sizing:border-box}body{margin:0;background:#f8f6f1;color:#243d41;font:17px/1.65 -apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
main{max-width:1030px;margin:auto;padding:56px 32px 90px}.eyebrow{font-size:12px;letter-spacing:.15em;font-weight:700;color:#697e7e}
h1{font-size:42px;line-height:1.15;letter-spacing:-1.2px;max-width:780px;margin:16px 0 24px}h2{font-size:28px;line-height:1.25;margin:58px 0 18px;border-top:1px solid #d7ddda;padding-top:26px}
p{max-width:850px;margin:15px 0}code{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.88em;background:#eaf0ec;padding:2px 5px;border-radius:4px}
a{color:#176b75;text-underline-offset:3px}
.cell{margin:22px 0 30px}.cell-label{font-size:10px;letter-spacing:.12em;color:#607c7c;font-weight:700;margin:0 0 6px 2px}
pre{white-space:pre-wrap;overflow-wrap:anywhere;margin:0;padding:18px 20px;font:13px/1.65 ui-monospace,SFMono-Regular,Consolas,monospace;border-radius:8px}
pre.source{background:#edf1ed;border:1px solid #dbe4de}pre code{background:none;font:inherit;padding:0}.result{margin-top:8px;background:#fffdfa;border-left:3px solid #6c9794;color:#34595b}
.output-image{display:block;max-width:100%;height:auto;margin:20px auto;border-radius:8px}.equation{overflow-x:auto;max-width:100%;padding:14px 0}.math-image{display:block;max-width:100%;height:auto;margin:8px auto}.inline-math{height:1.2em;width:auto;vertical-align:-.22em}
footer{border-top:1px solid #d7ddda;margin-top:50px;padding-top:18px;color:#6b7a79;font-size:13px}
@media(max-width:600px){main{padding:28px 16px 50px}h1{font-size:32px}h2{font-size:24px}body{font-size:16px}pre{font-size:11px;padding:12px}}
</style><main><div class="eyebrow">SMALLPROB_VERSION · EXECUTED NOTEBOOK</div>'''
    html = html.replace('NOTEBOOK_TITLE', escape(title)).replace('SMALLPROB_VERSION', 'SMALLPROB ' + __version__)
    html += "\n".join(body) + '<footer>Executed Python outputs. Data provenance and exact or approximate inference are identified in each example. This preview is self-contained; edit and run the accompanying .ipynb notebook.</footer></main></html>'
    target = root / (args.output or 'outputs/' + Path(args.notebook).stem.replace('_', '-') + '.html')
    target.parent.mkdir(exist_ok=True)
    target.write_text(html)
    print(target)


if __name__ == "__main__":
    main()
