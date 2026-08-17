#!/usr/bin/env python3
"""Rebuild the GitHub Pages demo from the app's own UI.

The demo is not a copy that has to be kept in step by hand: it is octopus.html with a
banner and one extra <script> injected. The shim replays a recorded dispatch in place of
the API, so the page on Pages behaves like the real one without a Python backend.

    python3 docs/build.py

Re-record the underlying run with docs/record.py when the app's event stream changes.
"""

from __future__ import annotations

import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
SRC = ROOT / "nim-test-layer" / "static" / "octopus.html"
OUT = ROOT / "docs" / "index.html"

BANNER = """
<div class="demo-banner">
  <strong>Recorded demo.</strong> GitHub Pages serves static files, so there is no Python
  backend here — this replays one real dispatch against the NVIDIA NIM API, at 3&times;
  speed. Everything else on the page is the live app.
  <a href="https://github.com/{repo}">Source &amp; setup &rarr;</a>
</div>
"""

BANNER_CSS = """
  .demo-banner{max-width:1120px;margin:0 auto;padding:11px 20px;font-size:13px;
    color:var(--text);background:linear-gradient(90deg,#0d3242,#0a2733);
    border-bottom:1px solid var(--ridge);position:relative;z-index:2;text-align:center}
  .demo-banner strong{color:var(--glow);font-weight:500}
  .demo-banner a{color:var(--glow);text-decoration:none;border-bottom:1px solid #7ff2e055}
  .demo-banner a:hover{border-bottom-color:var(--glow)}
"""


def build(repo: str) -> None:
    if not SRC.is_file():
        sys.exit(f"missing source UI: {SRC}")
    html = SRC.read_text()

    if "</style>" not in html or "<body>" not in html:
        sys.exit("source UI is not shaped as expected; update docs/build.py")

    html = html.replace("</style>", BANNER_CSS + "</style>", 1)
    html = html.replace("<body>", "<body>\n" + BANNER.format(repo=repo).strip(), 1)

    # The shim must define its fetch override before the app script runs, and the app
    # script is the last thing in the body — so inject immediately after the banner.
    html = html.replace("</div>\n<div class=\"motes\"",
                        "</div>\n<script src=\"demo-shim.js\"></script>\n<div class=\"motes\"", 1)
    if "demo-shim.js" not in html:
        # Fall back to injecting before the first <script> tag in the document.
        i = html.index("<script")
        html = html[:i] + '<script src="demo-shim.js"></script>\n' + html[i:]

    html = html.replace("<title>Octopus · multi-agent NIM console</title>",
                        "<title>Octopus · multi-agent NIM console (demo)</title>", 1)

    OUT.write_text(html)
    print(f"wrote {OUT.relative_to(ROOT)}  ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    build(sys.argv[1] if len(sys.argv) > 1 else "Bablu-singh/octopus-nim")
