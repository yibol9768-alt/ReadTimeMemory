"""Figure 1: one real BEAM-100K question (b2-knowledge_update-0). Left: Hindsight's write-time-organized context, read
directly (gpt-6-luna, tuned prompt): wrong in all three runs. Right: our ledger records -> compiled notes: right in all
three runs. All quoted text is excerpted verbatim from runs/beam/pro6 (ellipses mark cuts)."""
import sys; sys.path.insert(0, __file__.rsplit("/", 1)[0])
from svglib import Canvas, INK, SUB, ACC, TINT, OK, BAD, FONT
W, H = 820, 388
c = Canvas(W, H)
def rich(x, y, parts, size=13.5):
    s = "".join(f'<tspan fill="{col}" font-weight="{"bold" if b else "normal"}">{t}</tspan>' for t, col, b in parts)
    c.add(f'<text x="{x}" y="{y}" font-family="{FONT}" font-size="{size}" fill="{INK}" xml:space="preserve">{s}</text>')
# question bar
c.card(10, 8, W - 20, 50, TINT["grey"], r=12)
c.text(24, 30, "Question (BEAM-100K, knowledge update)", 13, "bold", SUB)
c.text(24, 49, "“What is the deadline for completing the first sprint focused on the basic layout and navigation?”", 14.5)
c.text(W - 24, 30, "Gold: April 5, 2024", 13.5, "bold", OK, "end")
# column titles
c.text(20, 86, "Organized at write time: Hindsight", 15.5, "bold")
c.text(20, 104, "maintained as sessions arrive; reader sees 124K characters", 12.5, "normal", SUB)
c.text(430, 86, "Organized at read time: Ledger+Compile", 15.5, "bold", ACC)
c.text(430, 104, "dated records never rewritten; organized per question", 12.5, "normal", SUB)
# left: Hindsight memories
c.card(10, 114, 395, 196, TINT["sand"], r=12)
L = [[("[world] … the first sprint deadline is ", INK, False), ("April 1, 2024", BAD, True), (".", INK, False)],
     [("[world] … with the first sprint deadline on ", INK, False), ("April 1, 2024", BAD, True), (".", INK, False)],
     [("[world] … The deadline is ", INK, False), ("April 1, 2024", BAD, True), (", for completing …", INK, False)],
     [("[experience] … For the ", INK, False), ("April 1, 2024", BAD, True), (" deadline for basic …", INK, False)],
     [("[world] … a new sprint deadline of ", INK, False), ("April 5, 2024", OK, True), (", and needs", INK, False)],
     [("        extra time for accessibility improvements …", INK, False)]]
for i, p in enumerate(L): rich(24, 140 + i * 22, p, 13)
c.text(24, 140 + 6 * 22 + 4, "… and 246 more of its 251 memories for this question", 12.5, "normal", SUB)
c.robot(64, 352, 0.82); c.arr("M200 312 L200 330", SUB)
c.card(100, 332, 305, 44, "#ffffff", r=12, stroke="#D6DCE3")
rich(114, 359, [("“The first sprint’s deadline … is ", INK, False), ("April 1, 2024", BAD, True), (".”", INK, False)], 13.5)
c.badge(392, 334, False, 13)
# right: ledger records -> notes
c.card(420, 114, 390, 92, TINT["blue"], r=12)
c.text(434, 133, "Ledger (atomic records, append-only)", 12.5, "bold", SUB)
rich(434, 154, [("2024-03-15  ", SUB, True), ("… Sprint 1, due ", INK, False), ("2024-04-01", BAD, True), (", … basic layout …", INK, False)], 12.5)
rich(434, 176, [("2024-03-15..04-05  ", SUB, True), ("Craig said he had changed his sprint", INK, False)], 12.5)
rich(434, 194, [("deadline to ", INK, False), ("April 5, 2024", OK, True), (" … extra time for accessibility …", INK, False)], 12.5)
c.arr("M615 207 L615 226", ACC); c.text(625, 222, "compile (one call)", 12.5, "bold", ACC)
c.card(420, 228, 390, 82, "#ffffff", r=12, stroke=ACC)
c.text(434, 247, "Evidence notes", 12.5, "bold", SUB)
rich(434, 268, [("Changes: ", ACC, True), ("The initial Sprint 1 deadline was ", INK, False), ("April 1, 2024", BAD, True), (". It was", INK, False)], 12.5)
rich(434, 286, [("later changed to ", INK, False), ("April 5, 2024", OK, True), (" to allow extra time for …", INK, False)], 12.5)
rich(434, 304, [("Conflicts: ", ACC, True), ("April 1, 2024 … conflicts with the later …", INK, False)], 12.5)
c.robot(474, 352, 0.82); c.arr("M615 312 L615 330", ACC)
c.card(510, 332, 300, 44, "#ffffff", r=12, stroke="#D6DCE3")
rich(524, 359, [("“… revised from April 1 to ", INK, False), ("April 5, 2024", OK, True), (".”", INK, False)], 13.5)
c.badge(797, 334, True, 13)
c.save(__file__.replace(".py", ".svg"))
