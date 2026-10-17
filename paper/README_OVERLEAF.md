# Chinese Decision 论文工程

题目：Chinese Decision: A Family of Language Models for Single-Pass Probabilistic Decision-Making

作者：yanqiang；Institute of Computing Technology, Chinese Academy of Sciences；yanqiang@ict.ac.cn。

## 上传 Overleaf

1. 在 Overleaf 选择 New Project → Upload Project，上传交付的 ZIP。
2. 主文档设为 `main.tex`，编译器选择 **pdfLaTeX**，使用常见的 TeX Live 环境。
3. 点击 Recompile；引用使用 BibTeX，Overleaf 会自动处理。也兼容 XeLaTeX。

不需要 shell escape、私有模板、系统字体或 Python。所有图和表已打包，上传后可直接编译。未替你创建 Overleaf 账号内项目或提交 arXiv。

## 文件结构

- `main.tex`：主文件、标题、作者、摘要及版式。
- `sections/`：引言、相关工作、方法、实验、结果、部署、讨论及附录。
- `references.bib`：已核对的文献元数据与来源链接；`main.bbl` 为编译生成的参考文献，便于 arXiv 源码处理。
- `figures/`：两张矢量 PDF 图。
- `tables/`：从本地评测记录自动生成的表格。
- `data/evidence.json`：论文使用的精简评测快照。
- `data/source_manifest.json`：原始证据路径和 SHA256。
- `generate_artifacts.py`：可选的图表重建工具。

## 证据与写作口径

正文按 **2026-10-16** 的本地资料写成。0.8B 已完成一轮训练，主结果为验证集选定的 step 10000、T=1。2B 使用 CE+RLCD，已存在 step 1000 和 step 2000 的全量评测，但训练尚未完成。2B 与 0.8B 在目标函数、批量和训练进度上不同，不将其写成控制变量的 scaling 实验。

正文另外收录了合并后 T=1.108133 的模型与 Q8_0 的全量部署评测。该结果比根目录 README 更完整；不把校准后 ECE 与未校准主结果混成同一配置。

基线数字来自 Chinese-Jev 论文 Tables 1/2，已在线核对，但没有重新运行基线。本次编写不重新运行训练或 GPU 回测。CPU 性能来自现有测量；不同输入、硬件和服务开销不混用。

## 投稿前需要补充的研究工作

这是一份内容完整、证据边界明确的技术报告初稿。若按完整模型系列论文投稿，建议完成 2B 训练并更新结果，同时补齐同规模 CE/RLCD 对照、相同数据预算的规模对照、多个随机种子或 parent-group bootstrap、候选顺序和分布外鲁棒性评估。不能把尚未执行的实验写成已有结论。

请作者最终核定署名大小写、单位信息、数据使用条件，以及所有学术陈述。正文没有虚构基金、合作作者、2B 最终结果或额外消融。

## 可选本地重建

安装 `requirements.txt` 后，在本目录运行 `python generate_artifacts.py`，从随包快照重新生成图表。若原项目评测目录仍在，可运行 `python generate_artifacts.py --refresh-snapshot` 更新证据；更新后也需要同步检查正文中的文字数值和结论。

常见本地编译方式：`latexmk -pdf main.tex`；或依次运行 `pdflatex main.tex`、`bibtex main`、`pdflatex main.tex`、`pdflatex main.tex`。
