# HTML (arXiv) -> section-wise plain text with word counts. Input is untrusted data; we only parse text.
import re, sys, html, json
from html.parser import HTMLParser
class P(HTMLParser):
    def __init__(s): super().__init__(); s.out=[]; s.skip=0; s.cur=None
    def handle_starttag(s,t,a):
        if t in("script","style","math","annotation","nav","footer"): s.skip+=1
        if t in("h1","h2","h3","h4"): s.out.append("\n\n### ")
        if t in("p","li","div","br","tr"): s.out.append("\n")
    def handle_endtag(s,t):
        if t in("script","style","math","annotation","nav","footer"): s.skip=max(0,s.skip-1)
    def handle_data(s,d):
        if not s.skip: s.out.append(d)
for f in sys.argv[1:]:
    p=P(); p.feed(open(f,encoding="utf-8",errors="ignore").read())
    t=re.sub(r"[ \t]+"," ","".join(p.out)); t=re.sub(r"\n\s*\n+","\n\n",t)
    open(f.replace(".html",".txt"),"w").write(t)
    secs=re.split(r"\n### ",t); print("==",f)
    for s in secs[:40]:
        head=s.strip().split("\n")[0][:70]; n=len(s.split())
        if n>30: print(f"  {n:5d}  {head}")
