"""Draw the current decision-only pointer architecture as editable SVG and PNG.

Architecture: decision/modeling.py (PointerHead, prompt_masks, DecisionModel.forward),
decision/processing.py (pointer_ids), and decision/pipeline.py (probabilities).
Layout references only (the depicted architecture comes from the repository):
https://aclanthology.org/N19-1423.pdf, Fig. 1
https://aclanthology.org/P19-1441.pdf, Fig. 1

Run with a Python environment containing PyMuPDF and Pillow.
"""

from html import escape
from pathlib import Path
import json
import math
import re

W, H = 1800, 1080
ROOT = Path(__file__).resolve().parent
INK = "#253346"
MUTED = "#617084"
LINE = "#CBD4DE"
BLUE = "#356D9F"
TEAL = "#197C79"
PURPLE = "#80619E"
AMBER = "#996D2F"
PALE_BLUE = "#EDF4FB"
PALE_TEAL = "#ECF7F4"
PALE_PURPLE = "#F3EEF8"


class Figure:
    def __init__(self, zh=False):
        self.zh = zh
        self.parts = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" role="img" aria-labelledby="title desc">',
            '<title id="title">Chinese Decision: Joint Pointer Decision Model</title>',
            '<desc id="desc">Decision-only Qwen3.5 architecture. All questions share one prompt and one backbone forward pass. Full-attention layers allow bidirectional attention within the joint prompt; linear-attention layers remain causal. A shared LayerNorm and two projections produce candidate keys and question queries. Candidate logits combine a scaled dot product and an affine candidate prior. Valid candidate scores are temperature-scaled and normalized per question.</desc>',
            '<style>path,line,rect{vector-effect:non-scaling-stroke}</style>',
            '<rect width="1800" height="1080" fill="white"/>',
        ]

    def tr(self, en, zh):
        return zh if self.zh else en

    def rect(self, x, y, w, h, fill="white", stroke=LINE, r=7, dash=None, sw=1.2):
        extra = f' stroke-dasharray="{dash}"' if dash else ""
        self.parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{extra}/>')

    def text(self, x, y, value, size=21, color=INK, weight=400, anchor="start", cls=""):
        family = "Times New Roman" if cls == "math" else "Courier New" if cls == "mono" else "Helvetica"
        self.parts.append(f'<text data-label="true" x="{x}" y="{y}" font-size="{size}" fill="{color}" font-weight="{weight}" font-family="{family}" text-anchor="{anchor}">{escape(value)}</text>')

    def path(self, d, color="#647286", arrow=False, width=1.65, dash=None):
        extra = ""
        if dash:
            extra += f' stroke-dasharray="{dash}"'
        self.parts.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}" stroke-linejoin="round" stroke-linecap="round"{extra}/>')
        if arrow:
            # Explicit triangles render consistently in SVG viewers and MuPDF.
            points = re.findall(r"[-+]?\d+(?:\.\d+)?", d)
            px, py, x, y = map(float, points[-4:])
            angle = math.atan2(y - py, x - px)
            bx, by = x - 9 * math.cos(angle), y - 9 * math.sin(angle)
            dx, dy = 4 * math.sin(angle), -4 * math.cos(angle)
            self.parts.append(f'<polygon points="{x},{y} {bx+dx},{by+dy} {bx-dx},{by-dy}" fill="{color}"/>')

    def arrow(self, x1, y1, x2, y2):
        self.path(f'M{x1},{y1} L{x2},{y2}', arrow=True)

    def panel(self, x, y, w, h, title):
        self.rect(x, y, w, h, stroke="#D6DDE5", r=9)
        self.text(x + 22, y + 36, title, 24, weight=700)

    def pill(self, x, y, w, label, color, fill, size=18):
        self.rect(x, y, w, 30, fill=fill, stroke=color, r=5, sw=0.8)
        self.text(x + w / 2, y + 21, label, size, color, anchor="middle", weight=600)

    def vector(self, x, y, color, fill):
        for i in range(5):
            self.rect(x + i * 13, y, 11, 25, fill=fill, stroke=color, r=1, sw=0.7)

    def main(self):
        tr = self.tr
        self.text(36, 47, tr("Chinese Decision: Joint Pointer Decision Model", "Chinese Decision：多题联合指针决策模型"), 31, weight=700)
        self.text(37, 79, tr("One shared prompt  ·  One backbone forward pass  ·  Candidate-level probability distributions", "共享上下文与提示序列  ·  一次主干前向计算  ·  逐题输出候选概率分布"), 20, MUTED)
        self.panel(35, 108, 1730, 437, tr("(a)  Joint decision architecture", "(a)  多题联合决策主流程"))

        # One joint sequence. [C] and [D] are display aliases for the actual markers.
        self.text(67, 183, tr("Joint request", "联合请求"), 22, weight=700)
        self.rect(67, 202, 370, 276, fill="#FAFCFE")
        self.rect(81, 215, 342, 46, fill=PALE_BLUE, stroke="#B6CBE0", r=4)
        self.text(97, 244, tr("Shared context x (written once)", "共享上下文 x（只写入一次）"), 19, BLUE, weight=600)
        self.text(97, 292, tr("Question 1 + candidate descriptions", "问题 1 + 各候选项的描述"), 18, weight=600)
        self.text(99, 325, "c₁₁", 22, cls="math")
        self.pill(139, 303, 45, "[C]", TEAL, PALE_TEAL)
        self.text(201, 325, "c₁₂", 22, cls="math")
        self.pill(240, 303, 45, "[C]", TEAL, PALE_TEAL)
        self.text(302, 324, "…", 24, MUTED)
        self.pill(348, 303, 56, "[D₁]", PURPLE, PALE_PURPLE)
        self.path("M83,350 L422,350", color="#E0E6ED", width=1)
        self.text(97, 380, tr("Question 2 + candidate descriptions", "问题 2 + 各候选项的描述"), 18, weight=600)
        self.text(99, 414, "c₂₁", 22, cls="math")
        self.pill(139, 392, 45, "[C]", TEAL, PALE_TEAL)
        self.text(201, 414, "c₂₂", 22, cls="math")
        self.pill(240, 392, 45, "[C]", TEAL, PALE_TEAL)
        self.text(302, 413, "…", 24, MUTED)
        self.pill(348, 392, 56, "[D₂]", PURPLE, PALE_PURPLE)
        self.text(251, 457, tr("…  Q questions in one token sequence", "…  一条 token 序列容纳 Q 道题"), 17, MUTED, anchor="middle")
        self.arrow(437, 336, 486, 336)

        # The causal LM decoder, with layer-specific masks, is the representation backbone.
        self.text(511, 183, tr("Shared representation", "共享表示"), 22, weight=700)
        self.rect(486, 238, 316, 201, fill=PALE_BLUE, stroke=BLUE)
        self.rect(496, 228, 296, 201, fill=PALE_BLUE, stroke=BLUE)
        self.rect(506, 218, 276, 201, fill="#F6FAFE", stroke=BLUE)
        self.text(644, 258, "Qwen3.5", 26, BLUE, 700, "middle")
        self.text(644, 289, tr("Text backbone", "文本主干"), 21, BLUE, 600, "middle")
        self.path("M526,308 L762,308", color="#C9D9E8", width=1)
        self.text(644, 338, tr("Token embeddings", "Token 嵌入"), 19, anchor="middle")
        self.text(644, 370, tr("Hybrid decoder stack × L", "混合注意力解码层 × L"), 19, anchor="middle")
        self.pill(556, 390, 176, tr("LoRA (training)", "LoRA（训练时）"), AMBER, "#FFF7E9", 17)
        self.text(644, 472, tr("Layer-specific masks → (b)", "分层注意力掩码 → (b)"), 18, MUTED, anchor="middle")
        self.arrow(803, 336, 850, 336)

        # Read exactly the candidate and decision marker positions from H.
        self.text(855, 183, tr("Marker states", "标记位置的隐状态"), 22, weight=700)
        self.rect(850, 218, 230, 240, fill="#FCFDFE")
        self.text(965, 252, "H ∈ ℝᴮ×ᵀ×ᴰ", 24, MUTED, anchor="middle", cls="math")
        self.vector(872, 284, TEAL, "#D7EEE6")
        self.text(957, 305, "hⱼ₁, …, hⱼKⱼ", 22, TEAL, cls="math")
        self.text(872, 337, tr("Candidate [C] positions", "候选标记 [C] 位置"), 17, TEAL)
        self.vector(872, 371, PURPLE, "#E5DAF1")
        self.text(957, 392, "hⱼᴰ", 24, PURPLE, cls="math")
        self.text(872, 426, tr("Decision [Dⱼ] position", "决策标记 [Dⱼ] 位置"), 17, PURPLE)
        self.arrow(1080, 336, 1129, 336)

        self.text(1135, 183, tr("Shared readout", "共享决策头"), 22, weight=700)
        self.rect(1129, 275, 228, 122, fill=PALE_TEAL, stroke=TEAL)
        self.text(1243, 316, "PointerHead", 24, TEAL, 700, "middle")
        self.text(1243, 350, tr("Candidate scoring", "逐候选打分"), 20, anchor="middle")
        self.text(1243, 379, tr("d = 256  ·  detail → (c)", "d = 256  ·  细节 → (c)"), 17, MUTED, anchor="middle")
        self.text(1243, 437, "zⱼ = [zⱼ₁, …, zⱼKⱼ]", 23, anchor="middle", cls="math")
        self.arrow(1357, 336, 1406, 336)

        self.text(1411, 183, tr("Per-question probabilities", "逐题候选概率"), 22, weight=700)
        self.rect(1406, 218, 326, 240, fill="#FCFDFE")
        self.text(1569, 256, tr("Valid candidates + softmax", "有效候选掩码 + Softmax"), 20, BLUE, 600, "middle")
        self.text(1569, 292, "pⱼ = softmax(zⱼ / τ)", 25, anchor="middle", cls="math")
        for idx, (yy, length) in enumerate([(317, 48), (352, 162), (387, 48)], 1):
            self.text(1429, yy + 19, f"pⱼ{idx}", 22, cls="math")
            self.rect(1480, yy, 205, 21, fill="#EFF3F7", stroke="none", r=3)
            self.rect(1480, yy, length, 21, fill=BLUE if idx == 2 else "#A8BFD3", stroke="none", r=3)
        self.text(1569, 438, tr("Schematic distribution; j = 1, …, Q", "示意分布；j = 1, …, Q"), 16, MUTED, anchor="middle")
        self.text(67, 519, tr("The same head handles choice, yes/no, and ordered score candidates; each question has its own candidate set Kⱼ.", "同一个指针头处理 choice、是 / 否和有序评分候选；每道题使用各自的候选集合 Kⱼ。"), 19, MUTED)

    def mask(self, x, y, full):
        n, cell = 8, 20
        # Abstract token order: prefix, six prompt tokens, suffix. No padding shown.
        for row in range(n):
            for col in range(n):
                causal = row >= col
                added = full and 1 <= row <= 6 and 1 <= col <= 6 and not causal
                color = "#BAD8EA" if added else "#D9E1EA" if causal else "#FFFFFF"
                self.rect(x + col * cell, y + row * cell, cell - 1, cell - 1,
                          fill=color, stroke="#E2E7ED", r=0, sw=0.5)
        self.rect(x - 1, y - 1, 161, 161, fill="none", stroke="#93A3B5", r=0, sw=0.8)
        if full:
            self.rect(x + 19, y + 19, 121, 121, fill="none", stroke=BLUE, r=0, dash="4 3", sw=1)
        self.text(x + 80, y - 9, self.tr("Keys →", "键 →"), 15, MUTED, anchor="middle")
        self.text(x - 15, y + 90, "Q", 17, MUTED, anchor="middle", cls="math")

    def details(self):
        tr = self.tr
        self.panel(35, 575, 650, 415, tr("(b)  Layer-specific attention masks", "(b)  按层类型设置注意力掩码"))
        self.text(193, 656, tr("Full attention", "全注意力层"), 22, BLUE, 600, "middle")
        self.text(498, 656, tr("Linear attention", "线性注意力层"), 22, weight=600, anchor="middle")
        self.mask(113, 688, True)
        self.mask(418, 688, False)
        self.text(193, 878, tr("Bidirectional joint prompt", "联合提示区间内双向可见"), 18, BLUE, anchor="middle")
        self.text(498, 878, tr("Causal Gated DeltaNet", "Gated DeltaNet 保持因果"), 18, MUTED, anchor="middle")
        self.rect(80, 905, 16, 16, fill="#D9E1EA", stroke="none", r=1)
        self.text(104, 919, tr("Causal links", "原有因果连接"), 17, MUTED)
        self.rect(284, 905, 16, 16, fill="#BAD8EA", stroke="none", r=1)
        self.text(308, 919, tr("Added prompt links", "新增提示区间连接"), 17, MUTED)
        self.rect(526, 905, 16, 16, fill="white", stroke="#B5C0CC", r=1)
        self.text(550, 919, tr("Blocked", "不可见"), 17, MUTED)
        self.text(67, 952, tr("Prompt = context + all questions + candidates + markers.", "提示区间 = 上下文 + 全部问题、候选项及标记。"), 18, MUTED)
        self.text(67, 977, tr("Rows: queries; columns: keys. Chat prefix/suffix stay causal.", "行：查询；列：键。区间外的聊天前后缀仍遵循因果规则。"), 16, MUTED)

        self.panel(715, 575, 1050, 415, tr("(c)  PointerHead: shared scoring for every question", "(c)  PointerHead：所有题目共享同一套打分参数"))
        self.text(747, 653, tr("Gather marker states → shared LayerNorm → separate affine projections", "读取标记隐状态 → 共享 LayerNorm → 两路独立线性投影"), 19, MUTED)

        self.rect(748, 685, 168, 66, fill=PALE_TEAL, stroke=TEAL)
        self.text(832, 713, tr("Candidate state", "候选隐状态"), 18, TEAL, 600, "middle")
        self.text(832, 740, "hⱼₖ ∈ ℝᴰ", 22, TEAL, anchor="middle", cls="math")
        self.arrow(916, 718, 957, 718)
        self.rect(957, 685, 216, 66, fill=PALE_TEAL, stroke=TEAL)
        self.text(1065, 714, tr("LayerNorm + Linear", "LayerNorm + 线性层"), 19, TEAL, 600, "middle")
        self.text(1065, 741, "kⱼₖ = Wc LN(hⱼₖ)", 22, anchor="middle", cls="math")
        self.arrow(1173, 718, 1211, 718)
        self.vector(1219, 706, TEAL, "#D7EEE6")
        self.text(1250, 762, "kⱼₖ ∈ ℝᵈ", 22, TEAL, anchor="middle", cls="math")

        self.rect(748, 798, 168, 66, fill=PALE_PURPLE, stroke=PURPLE)
        self.text(832, 826, tr("Decision state", "决策隐状态"), 18, PURPLE, 600, "middle")
        self.text(832, 853, "hⱼᴰ ∈ ℝᴰ", 22, PURPLE, anchor="middle", cls="math")
        self.arrow(916, 831, 957, 831)
        self.rect(957, 798, 216, 66, fill=PALE_PURPLE, stroke=PURPLE)
        self.text(1065, 827, tr("LayerNorm + Linear", "LayerNorm + 线性层"), 19, PURPLE, 600, "middle")
        self.text(1065, 854, "qⱼ = Wq LN(hⱼᴰ)", 22, anchor="middle", cls="math")
        self.arrow(1173, 831, 1211, 831)
        self.vector(1219, 819, PURPLE, "#E5DAF1")
        self.text(1250, 875, "qⱼ ∈ ℝᵈ", 22, PURPLE, anchor="middle", cls="math")

        self.path("M1286,718 L1321,718 L1321,761 L1361,761", arrow=True)
        self.path("M1286,831 L1321,831 L1321,798 L1361,798", arrow=True)
        self.rect(1361, 714, 369, 127, fill="#F6F9FC", stroke="#8A9BAE")
        self.text(1545, 747, tr("Scaled dot product", "缩放点积"), 22, weight=600, anchor="middle")
        self.text(1545, 780, "+", 24, MUTED, anchor="middle")
        self.text(1545, 810, tr("Affine candidate prior", "候选线性先验项"), 21, weight=600, anchor="middle")
        self.text(1545, 875, tr("Float32 scoring; d = 256", "Float32 打分；d = 256"), 18, MUTED, anchor="middle")
        self.path("M747,899 L1732,899", color="#DFE5EC", width=1)
        self.text(747, 938, "zⱼₖ = kⱼₖᵀ qⱼ / √d  +  wᵀ kⱼₖ + b", 29, cls="math")
        self.text(1729, 936, tr("k = 1, …, Kⱼ", "k = 1, …, Kⱼ"), 21, MUTED, anchor="end", cls="math")
        self.text(747, 975, tr("Affine projections include biases. Invalid padded candidates receive −10⁴ logits.", "两路投影均含偏置；候选补齐位置的无效 logit 设为 −10⁴。"), 17, MUTED)

        self.text(37, 1024, "[C] = <|object_ref_end|>", 17, TEAL, cls="mono")
        self.text(400, 1024, "[Dⱼ] = <|object_ref_start|>", 17, PURPLE, cls="mono")
        self.text(833, 1024, tr("τ: configured temperature  ·  d: pointer dimension  ·  D: backbone hidden size", "τ：配置的温度参数  ·  d：指针投影维度  ·  D：主干隐状态维度"), 17, MUTED)
        self.text(37, 1056, tr("Implementation: DecisionProcessor.pointer_ids → DecisionModel.forward → PointerHead → candidate_probabilities", "实现对应：DecisionProcessor.pointer_ids → DecisionModel.forward → PointerHead → candidate_probabilities"), 16, MUTED)

    def svg(self):
        self.main()
        self.details()
        return "\n".join(self.parts) + "\n</svg>\n"


def render():
    import pymupdf
    from PIL import Image

    outputs = []
    for zh in (False, True):
        stem = "chinese-decision-pointer-acl" + ("-zh" if zh else "")
        svg_path = ROOT / f"{stem}.svg"
        png_path = ROOT / f"{stem}.png"
        svg_path.write_text(Figure(zh).svg(), encoding="utf-8")
        with pymupdf.open(svg_path) as doc:
            doc[0].get_pixmap(matrix=pymupdf.Matrix(2, 2), alpha=False).save(png_path)
            # Inspect vector text bounds in memory, without exporting a temporary PDF.
            with pymupdf.open(stream=doc.convert_to_pdf(), filetype="pdf") as vector:
                labels = [span for block in vector[0].get_text("dict")["blocks"]
                          for line in block.get("lines", []) for span in line["spans"]]
                problems = [s["text"] for s in labels if s["bbox"][0] < 0 or s["bbox"][1] < 0
                            or s["bbox"][2] > W or s["bbox"][3] > H]
                if problems:
                    raise RuntimeError(f"Text outside figure: {problems}")
        with Image.open(png_path) as bitmap:
            assert bitmap.size == (W * 2, H * 2)
            bitmap.save(png_path, dpi=(300, 300))
        outputs.append({"svg": str(svg_path), "png": str(png_path),
                        "pixel_size": [W * 2, H * 2], "overflow_labels": problems})
    print(json.dumps(outputs, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    render()
