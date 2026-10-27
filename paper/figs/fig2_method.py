"""Figure 2: the Ledger+Compile pipeline. Write side (top row) and read side (bottom row)."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from svglib import Canvas, INK, SUB, ACC, TINT, OK, BAD, FONT
W, H = 820, 330
c = Canvas(W, H)
def box(x, y, w, h, title, lines, fill, stroke=None, tcol=INK):
    c.card(x, y, w, h, fill, r=12, stroke=stroke)
    c.text(x + w / 2, y + 22, title, 14, "bold", tcol, "middle")
    for i, l in enumerate(lines): c.text(x + w / 2, y + 42 + i * 17, l, 12, "normal", SUB, "middle")
# row labels
c.text(14, 26, "WRITE TIME  (once per segment, as sessions arrive)", 12.5, "bold", SUB)
c.text(14, 182, "READ TIME  (once per question)", 12.5, "bold", ACC)
# write row
y1 = 38
box(14, y1, 120, 92, "Sessions", ["dated user–", "assistant turns"], TINT["grey"])
box(160, y1, 140, 92, "Segment", ["semantic boundaries", "from embeddings", "(no LLM)"], TINT["grey"])
box(326, y1, 170, 92, "Encode (1 call)", ["self-contained records:", "type, time, entities,", "source turns"], TINT["blue"])
box(522, y1, 150, 92, "Ledger", ["append-only;", "never merged, edited,", "or deleted"], TINT["blue"], stroke=ACC)
for x0, x1 in ((134, 160), (300, 326), (496, 522)): c.arr(f"M{x0+2} {y1+46} L{x1-3} {y1+46}", SUB)
# contrast: write-time consolidation (crossed out)
c.card(696, y1, 112, 92, TINT["rose"], r=12)
c.text(752, y1 + 22, "Others do", 13.5, "bold", BAD, "middle")
for i, l in enumerate(["merge / update /", "delete in place;", "summarize ahead"]): c.text(752, y1 + 42 + i * 17, l, 12, "normal", SUB, "middle")
c.arr(f"M674 {y1+46} L693 {y1+46}", BAD, dash="4 3")
# read row
y2 = 194
box(14, y2, 120, 112, "Question", ["with current", "date"], TINT["grey"])
box(160, y2, 132, 112, "Plan (1 call)", ["≤3 queries,", "entities,", "date range"], TINT["sand"])
box(318, y2, 152, 112, "Retrieve", ["dense + BM25 + date", "filter, fused by RRF;", "≈60K characters", "of records"], TINT["sand"])
c.card(496, y2, 182, 112, "#ffffff", r=12, stroke=ACC)
c.text(587, y2 + 22, "Compile (1 call)", 14, "bold", ACC, "middle")
for i, l in enumerate(["(1) evidence in time order", "(2) changes", "(3) conflicts", "(4) not found", "(5) standing instructions"]):
    c.text(510, y2 + 41 + i * 15.5, l, 12, "normal", INK)
box(704, y2, 104, 112, "Reader", ["reads notes", "+ the same", "records,", "answers"], TINT["green"])
for x0, x1 in ((134, 160), (292, 318), (470, 496), (678, 704)): c.arr(f"M{x0+2} {y2+56} L{x1-3} {y2+56}", ACC)
# ledger feeds retrieval
c.arr(f"M597 {y1+92} C597 160 394 150 394 {y2-3}", ACC, dash="5 4")
c.text(612, 166, "retrieved from the ledger", 12, "normal", ACC, "start")
c.save(__file__.replace(".py", ".svg"))
