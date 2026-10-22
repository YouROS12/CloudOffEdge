"""Word (.docx) version of the manuscript, built from the same LaTeX source.

Steps
 1. compile the manuscript in a scratch copy, keeping main.aux, so that every
    \\ref / \\eqref / caption gets the number LaTeX assigned;
 2. render the three TikZ diagrams (Figs. with a tikzpicture) from main.pdf
    to PNG, and use the PNG twins of the matplotlib figures;
 3. flatten main.tex for pandoc: measured values from numbers.tex substituted,
    references resolved, numbered captions, elsarticle front matter rewritten
    as plain title/authors/highlights/abstract/keywords;
 4. LaTeX -> Markdown with pandoc, place the reference list before the
    appendix, Markdown -> .docx with citeproc (numbered, paper/manuscript/elsevier-numeric.csl).

  python experiments/scripts/make_docx.py  ->  paper/manuscript/main.docx
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path

import pymupdf
import pypandoc

PAPER2 = Path(__file__).resolve().parents[2]  # repository root
MS = PAPER2 / "paper"
TECTONIC = Path(shutil.which("tectonic") or "tectonic")


def balanced(s: str, i: int) -> int:
    """Index just past the brace group starting at s[i] == '{'."""
    assert s[i] == "{", s[i:i + 20]
    d = 0
    for j in range(i, len(s)):
        if s[j] == "{" and s[j - 1] != "\\":
            d += 1
        elif s[j] == "}" and s[j - 1] != "\\":
            d -= 1
            if d == 0:
                return j + 1
    raise ValueError("unbalanced")


def compile_scratch(work: Path) -> dict[str, str]:
    if work.exists():
        shutil.rmtree(work)
    shutil.copytree(MS, work, ignore=shutil.ignore_patterns("*.docx", "_docx*"))
    subprocess.run([str(TECTONIC), "-X", "compile", "--keep-intermediates", "main.tex"], cwd=work,
                   capture_output=True, check=True)
    labels = {}
    for line in (work / "main.aux").read_text(encoding="utf-8", errors="replace").splitlines():
        m = re.match(r"\\newlabel\{([^}]*)\}\{\{(.*?)\}\{", line)
        if m and not m.group(1).endswith("@cref"):
            labels[m.group(1)] = m.group(2)
    return labels


def render_tikz(work: Path, out: Path, numbers: list[str]) -> dict[str, Path]:
    """Crop each 'Figure N' TikZ drawing from the compiled PDF (vector -> 300 dpi PNG)."""
    doc = pymupdf.open(work / "main.pdf")
    res = {}
    for n in numbers:
        for page in doc:
            hits = page.search_for(f"Figure {n}:")
            if not hits:
                continue
            cap = hits[0]
            items = [d["rect"] for d in page.get_drawings() if d["rect"].y1 <= cap.y0 + 1]
            items += [pymupdf.Rect(b[:4]) for b in page.get_text("blocks")
                      if b[3] <= cap.y0 + 1 and b[1] >= (min(r.y0 for r in items) - 30 if items else 0)]
            box = items[0]
            for r in items[1:]:
                box |= r
            box = pymupdf.Rect(box.x0 - 4, box.y0 - 4, box.x1 + 4, cap.y0 - 2) & page.rect
            p = out / f"_docx_fig{n}.png"
            page.get_pixmap(dpi=300, clip=box).save(p)
            res[n] = p
            break
    return res


def main() -> None:
    work = Path(__import__("tempfile").gettempdir()) / "cloudoffedge_docx_build"
    labels = compile_scratch(work)
    src = (MS / "main.tex").read_text(encoding="utf-8")

    # ---- measured values
    nums = {}
    for line in (MS / "numbers.tex").read_text(encoding="utf-8").splitlines():
        m = re.match(r"\\newcommand\{\\(\w+)\}\{(.*)\\xspace\}$", line)
        if m:
            nums[m.group(1)] = m.group(2)
    for name in sorted(nums, key=len, reverse=True):
        src = re.sub(r"\\" + name + r"(?![A-Za-z])(\{\})?", lambda _m, v=nums[name]: v, src)
    assert not re.search(r"\\R[A-Z]\w+", src), re.findall(r"\\R[A-Z]\w+", src)[:5]

    # ---- body only; front matter rebuilt below
    body = src[src.index("\\linenumbers") + len("\\linenumbers"):src.index("\\end{document}")]

    # ---- figures and tables: numbered captions, images
    fig_n = {k: v for k, v in labels.items() if k.startswith("fig:")}
    tikz_figs = []

    def float_env(m: re.Match) -> str:
        env, star, inner = m.group(1), m.group(2) or "", m.group(3)
        lab = re.search(r"\\label\{([^}]*)\}", inner)
        num = labels.get(lab.group(1), "?") if lab else "?"
        kind = "Figure" if env == "figure" else "Table"
        ci = inner.find("\\caption{")
        if ci >= 0:
            j = balanced(inner, ci + len("\\caption"))
            cap = inner[ci + len("\\caption{"):j - 1]
            inner = inner[:ci] + "\\caption{" + f"{kind} {num}. " + cap + "}" + inner[j:]
        if "tikzpicture" in inner:
            tikz_figs.append(num)
            a = inner.index("\\resizebox")
            b = inner.index("\\end{tikzpicture}}") + len("\\end{tikzpicture}}")
            inner = inner[:a] + f"\\includegraphics[width=\\linewidth]{{_docx_fig{num}.png}}" + inner[b:]
        # Word has no two-column floats; pandoc drops the caption of table*
        return f"\\begin{{{env}}}{inner}\\end{{{env}}}"

    body = re.sub(r"\\begin\{(figure|table)(\*?)\}(?:\[[^\]]*\])?(.*?)\\end\{\1\2\}", float_env, body, flags=re.S)
    body = re.sub(r"figures/(\w+)\.pdf", r"figures/\1.png", body)
    # longtable caption (appendix)
    body = re.sub(r"\\caption\{Software adaptations, by node.\}",
                  "\\\\caption{Table " + labels.get("tab:adaptations", "A.1") + ". Software adaptations, by node.}", body)
    # photo appendix only if the photo exists (pandoc cannot evaluate \IfFileExists)
    has_photo = (MS / "figures" / "setup_photo.jpg").exists()

    # remove \IfFileExists{...}{A}{B} keeping A or B
    out, i = [], 0
    while True:
        p = body.find("\\IfFileExists{", i)
        if p < 0:
            out.append(body[i:])
            break
        out.append(body[i:p])
        a = balanced(body, p + len("\\IfFileExists"))
        b = balanced(body, a)
        c = balanced(body, b)
        out.append(body[a + 1:b - 1] if has_photo else body[b + 1:c - 1])
        i = c
    body = "".join(out)

    # ---- numbered display equations: Word gets the number as text at the end of the line
    def eqn(m: re.Match) -> str:
        inner = m.group(1)
        lab = re.search(r"\\label\{([^}]*)\}", inner)
        inner = re.sub(r"\\label\{[^}]*\}", "", inner).rstrip()
        num = labels.get(lab.group(1), "") if lab else ""
        return "\\[" + inner + (f"\\qquad\\qquad ({num})" if num else "") + "\\]"

    body = re.sub(r"\\begin\{equation\}(.*?)\\end\{equation\}", eqn, body, flags=re.S)

    # ---- references
    body = re.sub(r"\\eqref\{([^}]*)\}", lambda m: f"({labels.get(m.group(1), '?')})", body)
    body = re.sub(r"\\mbox\{\\ref\{([^}]*)\}\}", lambda m: labels.get(m.group(1), "?"), body)
    body = re.sub(r"\\ref\{([^}]*)\}", lambda m: labels.get(m.group(1), "?"), body)
    body = body.replace("\\appendix", "")
    body = re.sub(r"\\section\{Software adaptations for embedded deployment\}",
                  r"\\section*{Appendix A. Software adaptations for embedded deployment}", body)
    body = re.sub(r"\\section\{Embedded testbed\}", r"\\section*{Appendix B. Embedded testbed}", body)
    body = re.sub(r"\\bibliographystyle\{[^}]*\}\s*\\bibliography\{[^}]*\}", "", body)
    body = body.replace("\\linenumbers", "")
    body = re.sub(r"L\{(0\.\d+\\linewidth)\}", r"p{\1}", body)
    body = body.replace("@{}", "")  # pandoc's LaTeX reader drops tables whose \multicolumn specs use @{}
    # Table 1: Word's larger body font needs a wider label column ("Processor", "Memory")
    body = body.replace("{p{0.15\\linewidth}p{0.36\\linewidth}p{0.42\\linewidth}}",
                        "{p{0.24\\linewidth}p{0.31\\linewidth}p{0.38\\linewidth}}")
    body = body.replace("\\xspace", "")

    # ---- front matter
    fm = src[src.index("\\begin{frontmatter}"):src.index("\\end{frontmatter}")]
    title = re.search(r"\\title\{(.*?)\}\n", fm, re.S).group(1).replace("\\mbox{hardware}", "hardware")
    title = " ".join(title.split())
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", fm, re.S).group(1)
    highlights = re.search(r"\\begin\{highlights\}(.*?)\\end\{highlights\}", fm, re.S).group(1)
    keywords = re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", fm, re.S).group(1)
    keywords = "; ".join(k.strip() for k in keywords.replace("\n", " ").split("\\sep"))
    front = (
        f"\\title{{{title}}}\n\\author{{Youssef Natij\\textsuperscript{{a,*}} \\and Hajar El Karch\\textsuperscript{{b}}"
        f" \\and Abdelkader Mezouari\\textsuperscript{{a}}}}\n\\date{{}}\n\\begin{{document}}\n\\maketitle\n"
        "\\noindent\\textsuperscript{a}Scientific Research and Innovation Laboratory, Higher School of Technology, "
        "Ibn Tofail University, Campus Universitaire, B.P. 242, 14000 Kenitra, Morocco\n\n"
        "\\noindent\\textsuperscript{b}Laboratory of Advanced Systems Engineering, National School of Applied "
        "Engineering, Ibn Tofail University, Campus Universitaire, B.P. 241, 14000 Kenitra, Morocco\n\n"
        "\\noindent\\textsuperscript{*}Corresponding author: youssef.natij@uit.ac.ma\n\n"
        "\\noindent E-mail addresses: youssef.natij@uit.ac.ma (Y. Natij), hajar.elkarch@uit.ac.ma (H. El Karch), "
        "abdelkader.mezouari1@uit.ac.ma (A. Mezouari)\n\n"
        f"\\section*{{Highlights}}\n\\begin{{itemize}}{highlights}\\end{{itemize}}\n\n"
        f"\\section*{{Abstract}}\n{abstract}\n\n\\noindent\\textbf{{Keywords:}} {keywords}\n\n"
    )
    for name in sorted(nums, key=len, reverse=True):
        front = re.sub(r"\\" + name + r"(?![A-Za-z])(\{\})?", lambda _m, v=nums[name]: v, front)
    front = front.replace("\\xspace", "")
    tex = "\\documentclass{article}\n\\usepackage{amsmath}\n" + front + body + "\n\\end{document}\n"

    build = work.parent / "paper2_docx_flat"
    if build.exists():
        shutil.rmtree(build)
    build.mkdir()
    (build / "figures").mkdir()
    for f in (MS / "figures").glob("*"):
        shutil.copy(f, build / "figures" / f.name)
    for n, p in render_tikz(work, build, tikz_figs).items():
        pass
    (build / "flat.tex").write_text(tex, encoding="utf-8")

    md = pypandoc.convert_file(str(build / "flat.tex"), "markdown", format="latex",
                               extra_args=["--wrap=none", "--standalone"])  # keep title/author metadata
    # reference list before the appendix
    md = md.replace("# Appendix A. Software adaptations", "# References {-}\n\n::: {#refs}\n:::\n\n# Appendix A. Software adaptations", 1)
    # \paragraph{} run-in headings (level 4) are unnumbered in the PDF
    md = re.sub(r"^(####+ .*?)\s*(\{[^}]*\})?$", lambda m: m.group(1) + " {-}", md, flags=re.M)
    (build / "flat.md").write_text(md, encoding="utf-8")
    out = MS / "main.docx"
    pypandoc.convert_file(str(build / "flat.md"), "docx", format="markdown", outputfile=str(out),
                          extra_args=["--citeproc", f"--bibliography={MS / 'refs_paper1.bib'}",
                                      f"--bibliography={MS / 'refs_edge.bib'}", f"--csl={MS / 'elsevier-numeric.csl'}", "--number-sections",
                                      f"--resource-path={build}", "--metadata=link-citations:true"])
    print("wrote", out, f"({out.stat().st_size / 1e6:.1f} MB); TikZ figures rendered:", tikz_figs)


if __name__ == "__main__":
    main()
