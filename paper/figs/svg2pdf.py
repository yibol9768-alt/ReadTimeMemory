"""Convert SVG figures to PDF (and PNG previews) with headless Chromium, keeping text as text.
    ~/.venvs/browser-qa/bin/python paper/figs/svg2pdf.py paper/figs/fig1_teaser.svg [...]"""
import re, sys, pathlib
from playwright.sync_api import sync_playwright
with sync_playwright() as p:
    exe = sorted(pathlib.Path.home().glob(".cache/ms-playwright/chromium_headless_shell-*/chrome-*/*headless*shell*"))[-1]
    b = p.chromium.launch(executable_path=str(exe), headless=True, args=["--disable-dev-shm-usage"])
    for f in sys.argv[1:]:
        src = pathlib.Path(f).resolve(); svg = src.read_text()
        w, h = map(float, re.search(r'viewBox="0 0 ([\d.]+) ([\d.]+)"', svg).groups())
        page = b.new_page(viewport={"width": int(w), "height": int(h)}, device_scale_factor=2)
        page.set_content(f'<html><body style="margin:0;background:white">{svg}</body></html>')
        page.pdf(path=str(src.with_suffix(".pdf")), width=f"{w}px", height=f"{h}px", print_background=True, page_ranges="1")
        page.screenshot(path=str(src.with_suffix(".png")))
        page.close(); print("wrote", src.with_suffix(".pdf").name)
    b.close()
