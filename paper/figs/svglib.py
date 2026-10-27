"""Shared drawing vocabulary for the paper's illustrated SVG figures: one palette, one set of pictograms."""
FONT = "'Nimbus Sans', Helvetica, Arial, sans-serif"; MONO = "'DejaVu Sans Mono', Menlo, monospace"
INK, SUB, ACC = "#2F3E4E", "#6B7785", "#4F7CAC"
TINT = {"blue": "#E3ECF6", "green": "#E3F0DF", "sand": "#FBF3DE", "grey": "#F1F3F6", "rose": "#F6E7E3", "lav": "#EAE6F3"}
OK, BAD, PAPER, PLINE, GOLD = "#5A9E6F", "#C46A5E", "#FFFDF6", "#CFC3A0", "#C9B37A"

class Canvas:
    def __init__(self, w, h):
        self.w, self.h, self.o = w, h, []
        cols = (SUB, ACC, "#B9C1CB", INK, OK, BAD)
        self.add(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {w} {h}" width="{w}" height="{h}">')
        self.add('<defs>' + "".join(f'<marker id="m{c[1:]}" viewBox="0 0 10 10" refX="7.5" refY="5" markerWidth="5.5" markerHeight="5.5" orient="auto"><path d="M0 0.8 L9.5 5 L0 9.2 z" fill="{c}"/></marker>' for c in cols) + '</defs>')
        self.add(f'<rect width="{w}" height="{h}" fill="#ffffff"/>')
    def add(self, s): self.o.append(s)
    def save(self, path): open(path, "w").write("\n".join(self.o + ["</svg>"])); print("wrote", path)
    def text(self, x, y, s, size=20, w="normal", fill=INK, anchor="start", font=FONT):
        self.add(f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" font-weight="{w}" fill="{fill}" text-anchor="{anchor}" xml:space="preserve">{s}</text>')
    def card(self, x, y, w, h, fill, r=18, stroke=None):
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}"' + (f' stroke="{stroke}" stroke-width="1.4"' if stroke else "") + '/>')
    def arr(self, d, col=SUB, w=2.2, dash=None):
        extra = f'stroke-dasharray="{dash}"' if dash else ""
        self.add(f'<path d="{d}" stroke="{col}" stroke-width="{w}" fill="none" stroke-linecap="round" stroke-linejoin="round" marker-end="url(#m{col[1:]})" {extra}/>')
    def robot(self, x, y, s=1.0, mood=None, col=ACC):
        mouth = {None: "M-6 6 L6 6", True: "M-7 4 Q0 11 7 4", False: "M-7 9 Q0 3 7 9"}[mood]
        self.add(f'<g transform="translate({x},{y}) scale({s})"><line x1="0" y1="-25" x2="0" y2="-33" stroke="{col}" stroke-width="3" stroke-linecap="round"/><circle cx="0" cy="-36" r="4.2" fill="{col}"/>'
                 f'<rect x="-22" y="-24" width="44" height="40" rx="12" fill="#ffffff" stroke="{col}" stroke-width="3"/><circle cx="-8.5" cy="-6" r="4.2" fill="{INK}"/><circle cx="8.5" cy="-6" r="4.2" fill="{INK}"/>'
                 f'<path d="{mouth}" stroke="{INK}" stroke-width="2.4" fill="none" stroke-linecap="round"/><path d="M-16 20 Q0 15 16 20 L19 32 L-19 32 Z" fill="{col}"/></g>')
    def key(self, x, y, s=1.0, col=OK):
        self.add(f'<g transform="translate({x},{y}) scale({s})" fill="none" stroke="{col}" stroke-width="3" stroke-linecap="round" stroke-linejoin="round"><circle cx="-7" cy="0" r="5.5"/><path d="M-1.5 0 L12 0 M7 0 L7 5 M11.5 0 L11.5 4"/></g>')
    def note(self, x, y, kind="lines", n=0, s=1.0):
        w, h, f = 34 * s, 42 * s, 9 * s
        self.add(f'<path d="M{x-w/2} {y-h/2} L{x+w/2-f} {y-h/2} L{x+w/2} {y-h/2+f} L{x+w/2} {y+h/2} L{x-w/2} {y+h/2} Z" fill="{PAPER}" stroke="{PLINE}" stroke-width="1.6" stroke-linejoin="round"/>')
        self.add(f'<path d="M{x+w/2-f} {y-h/2} L{x+w/2-f} {y-h/2+f} L{x+w/2} {y-h/2+f}" fill="#F2EAD0" stroke="{PLINE}" stroke-width="1.3" stroke-linejoin="round"/>')
        if kind == "key": self.key(x - 1, y + 3 * s, s)
        elif kind == "x":
            for i in range(n):
                cx = x - 7.5 * s + (i % 2) * 14 * s; cy = y - 8 * s + (i // 2) * 11 * s
                self.add(f'<path d="M{cx-3.2*s} {cy-3.2*s} L{cx+3.2*s} {cy+3.2*s} M{cx+3.2*s} {cy-3.2*s} L{cx-3.2*s} {cy+3.2*s}" stroke="{BAD}" stroke-width="{2.2*s}" stroke-linecap="round"/>')
        else:
            for k in (-9, -2, 5, 12): self.add(f'<line x1="{x-10*s}" y1="{y+k*s}" x2="{x+(10 if k < 12 else 3)*s}" y2="{y+k*s}" stroke="#B9AE8C" stroke-width="{1.6*s}" stroke-linecap="round"/>')
    def sheet(self, x, y, solid=True, s=1.0):
        st = f'stroke="{("#8794A2" if solid else "#B9C1CB")}" stroke-width="1.3" ' + ('' if solid else 'stroke-dasharray="3 3" ')
        self.add(f'<rect x="{x-13*s}" y="{y-9*s}" width="{26*s}" height="{18*s}" rx="2.5" fill="{"#ffffff" if solid else "none"}" {st}/>')
        for k in (-4, 0, 4): self.add(f'<line x1="{x-8*s}" y1="{y+k*s}" x2="{x+8*s}" y2="{y+k*s}" stroke="{"#A9B4C0" if solid else "#D3D9E0"}" stroke-width="1.2"/>')
    def stack(self, x, y, k=4, s=1.0):
        """A pile of record sheets; (x, y) is the centre of the top sheet."""
        for i in reversed(range(k)): self.sheet(x + i * 3.2 * s, y + i * 5.2 * s, True, s)
    def machine(self, x, y, lit, s=1.0):
        self.add(f'<g transform="translate({x},{y}) scale({s})"><rect x="-52" y="-44" width="104" height="80" rx="13" fill="{INK}"/><rect x="-44" y="-36" width="88" height="24" rx="7" fill="#243140"/>'
                 f'<circle cx="-24" cy="10" r="13" fill="#E9EEF3"/><path d="M-24 10 L-24 0" stroke="{INK}" stroke-width="3" stroke-linecap="round"/><circle cx="8" cy="10" r="13" fill="#E9EEF3"/><path d="M8 10 L16 5" stroke="{INK}" stroke-width="3" stroke-linecap="round"/>'
                 f'<rect x="29" y="-2" width="15" height="10" rx="3" fill="#E7C873"/><rect x="29" y="13" width="15" height="10" rx="3" fill="#9DBDDF"/><rect x="-40" y="36" width="80" height="7" rx="3" fill="#243140"/></g>')
        for i in range(5):
            on = i < lit; cx = x + (i - 2) * 15 * s; cy = y - 24 * s
            if on: self.add(f'<circle cx="{cx}" cy="{cy}" r="{8.5*s}" fill="{OK}" opacity="0.35"/>')
            self.add(f'<circle cx="{cx}" cy="{cy}" r="{5.2*s}" fill="{"#8FD3A0" if on else "#566473"}"/>')
    def badge(self, x, y, good, r=16):
        c = OK if good else BAD; d = "M-7.5 0 L-2.5 5.5 L8.5 -6.5" if good else "M-6.5 -6.5 L6.5 6.5 M6.5 -6.5 L-6.5 6.5"
        self.add(f'<circle cx="{x}" cy="{y}" r="{r}" fill="{c}" stroke="#ffffff" stroke-width="3"/><path d="{d}" transform="translate({x},{y}) scale({r/16})" stroke="#fff" stroke-width="3.6" fill="none" stroke-linecap="round" stroke-linejoin="round"/>')

# ---- additions for the dense versions --------------------------------------------------------------------------------
def _lines(self, x, y, ls, size=12, lh=None, **k):
    lh = lh or size * 1.32
    for i, s in enumerate(ls): self.text(x, y + i * lh, s, size, **k)
Canvas.lines = _lines
def _lamps(self, x, y, lit, r=3.6, gap=9.4, off="#ffffff", edge="#B9C1CB"):
    for i in range(5):
        on = i < lit
        self.add(f'<circle cx="{x + i * gap}" cy="{y}" r="{r}" fill="{OK if on else off}" stroke="{OK if on else edge}" stroke-width="1.2"/>')
Canvas.lamps = _lamps
def _head(self, x, y, s=1.0, col=ACC, mood=None):
    """Just the head of an agent, for tight layouts."""
    mouth = {None: "M-5 6 L5 6", True: "M-6 4 Q0 10 6 4", False: "M-6 8 Q0 3 6 8"}[mood]
    self.add(f'<g transform="translate({x},{y}) scale({s})"><line x1="0" y1="-19" x2="0" y2="-25" stroke="{col}" stroke-width="2.6" stroke-linecap="round"/><circle cx="0" cy="-27" r="3.4" fill="{col}"/>'
             f'<rect x="-18" y="-19" width="36" height="32" rx="10" fill="#ffffff" stroke="{col}" stroke-width="2.6"/><circle cx="-7" cy="-5" r="3.4" fill="{INK}"/><circle cx="7" cy="-5" r="3.4" fill="{INK}"/>'
             f'<path d="{mouth}" stroke="{INK}" stroke-width="2" fill="none" stroke-linecap="round"/></g>')
Canvas.head = _head
def _chart(self, x, y, w, h, series, ymax=5, xlabel="life", ylabel="marks", tick=11, label=12):
    """A small line chart. series: list of (label, values, colour, width, dash)."""
    n = len(series[0][1]); px = lambda i: x + i * w / (n - 1); py = lambda v: y + h - v * h / ymax
    for v in range(ymax + 1):
        self.add(f'<line x1="{x}" y1="{py(v)}" x2="{x+w}" y2="{py(v)}" stroke="#E3E7EC" stroke-width="1"/>'); self.text(x - 6, py(v) + tick * .36, str(v), tick, fill=SUB, anchor="end")
    for i in range(n): self.text(px(i), y + h + tick * 1.3, str(i + 1), tick, fill=SUB, anchor="middle")
    self.text(x + w / 2, y + h + tick * 1.3 + label * 1.25, xlabel, label, fill=SUB, anchor="middle")
    self.add(f'<text x="{x-tick*2}" y="{y+h/2}" font-family="{FONT}" font-size="{label}" fill="{SUB}" text-anchor="middle" transform="rotate(-90 {x-tick*2} {y+h/2})">{ylabel}</text>')
    for label, vals, col, lw, dash in series:
        d = " ".join(f'{"M" if i == 0 else "L"}{px(i):.1f} {py(v):.1f}' for i, v in enumerate(vals))
        self.add(f'<path d="{d}" stroke="{col}" stroke-width="{lw}" fill="none" stroke-linejoin="round" stroke-linecap="round" {("stroke-dasharray=" + chr(34) + dash + chr(34)) if dash else ""}/>')
        for i, v in enumerate(vals): self.add(f'<circle cx="{px(i):.1f}" cy="{py(v):.1f}" r="{lw+0.4}" fill="{col}"/>')
    return px, py
Canvas.chart = _chart
