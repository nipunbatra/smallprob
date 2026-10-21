"""Check showcase assets, links and downloadable Python files without a browser."""

from html.parser import HTMLParser
import json
from pathlib import Path
import re
from urllib.parse import unquote, urlsplit


class Links(HTMLParser):
    def __init__(self):
        super().__init__()
        self.urls = []

    def handle_starttag(self, tag, attrs):
        self.urls.extend(value for name, value in attrs
                         if name in {"href", "src"} and value)


def check():
    site = Path(__file__).resolve().parents[1] / "website"
    text = (site / "assets/cases.js").read_text()
    match = re.search(r"window\.SMALLPROB_CASES\s*=\s*(\[.*\]);?\s*$", text, re.S)
    assert match, "Missing static case data"
    cases = json.loads(match[1])
    assert len(cases) == 7 and len({case["id"] for case in cases}) == 7
    assets = []
    for case in cases:
        for name in ("id", "title", "description", "method", "smallprob", "pyro", "run", "download", "pyro_download", "diagram", "workflows", "smallprob_method", "pyro_method"):
            assert case[name], (case["id"], name)
        for name in ("diagram", "plot", "download", "pyro_download"):
            if case.get(name):
                assets.append((site / "index.html", case[name]))
        compile(case["smallprob"], f"{case['id']}:smallprob", "exec")
        compile(case["pyro"], f"{case['id']}:pyro", "exec")
        compile(case["run"], f"{case['id']}:run", "exec")
        assert [step["id"] for step in case["workflows"]] == ["generate", "fit", "summarize"]
        for step in case["workflows"]:
            for library in ("smallprob", "pyro"):
                compile(step[library], f"{case['id']}:{step['id']}:{library}", "exec")
    for page in site.rglob("*.html"):
        parser = Links()
        parser.feed(page.read_text())
        assets.extend((page, url) for url in parser.urls)
    vae = json.loads((site / "assets/vae/example.json").read_text())
    assert len(vae["steps"]) == 5
    for step in vae["steps"]:
        for library in ("smallprob", "pyro"):
            compile(step[library], f"vae:{step['id']}:{library}", "exec")
    for figure in vae["figures"]:
        assets.extend((site / "neural.html", figure[key]) for key in ("path", "png"))
    for download in vae["downloads"]:
        assets.append((site / "neural.html", download["path"]))
        compile((site / download["path"]).read_text(), download["path"], "exec")
    assert json.loads((site / "assets/meanfield/validation.json").read_text())["passed"]
    for page, url in assets:
        parsed = urlsplit(url)
        if parsed.scheme or parsed.netloc or not parsed.path:
            continue
        assert not parsed.path.startswith("/"), f"Root-relative link breaks project Pages: {url}"
        target = (page.parent / unquote(parsed.path)).resolve()
        assert target.is_relative_to(site.resolve()), (page, url)
        if target.is_dir():
            target = target / "index.html"
        assert target.is_file(), f"Missing asset: {page.relative_to(site)} -> {url}"
    scripts = list((site / "downloads/examples").glob("*.py"))
    assert len(scripts) >= 7
    for script in scripts:
        compile(script.read_text(), str(script), "exec")
    for notebook in (site / "downloads").glob("*.ipynb"):
        content = json.loads(notebook.read_text())
        cells = [cell for cell in content["cells"] if cell["cell_type"] == "code"]
        assert all(cell.get("execution_count") for cell in cells), notebook
        assert not any(out["output_type"] == "error" for cell in cells for out in cell["outputs"]), notebook
    print(f"Static showcase: {len(cases)} paired cases, native VI and VAE; {len(scripts)+len(vae['downloads'])} runnable downloads; all local assets resolve.")


if __name__ == "__main__":
    check()
