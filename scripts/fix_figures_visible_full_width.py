from copy import deepcopy
from pathlib import Path

from docx import Document
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt


SRC = Path(
    r"D:\program_FAERS\docs_or_reports\zolpidem_faers_manuscript_draft_cn_journal_20260604_two_column_layout_title_author_abstract_fixed.docx"
)
OUT = Path(
    r"D:\program_FAERS\docs_or_reports\zolpidem_faers_manuscript_draft_cn_journal_20260604_two_column_layout_title_author_abstract_figures_fixed.docx"
)

CONTENT_WIDTH_CM = 16.7
GUTTER_CM = 0.8


def make_sect_pr(cols_num):
    sect = OxmlElement("w:sectPr")

    sect_type = OxmlElement("w:type")
    sect_type.set(qn("w:val"), "continuous")
    sect.append(sect_type)

    pg_sz = OxmlElement("w:pgSz")
    pg_sz.set(qn("w:w"), str(int(Cm(21).twips)))
    pg_sz.set(qn("w:h"), str(int(Cm(29.7).twips)))
    sect.append(pg_sz)

    pg_mar = OxmlElement("w:pgMar")
    pg_mar.set(qn("w:top"), str(int(Cm(2.54).twips)))
    pg_mar.set(qn("w:right"), str(int(Cm(2.15).twips)))
    pg_mar.set(qn("w:bottom"), str(int(Cm(2.2).twips)))
    pg_mar.set(qn("w:left"), str(int(Cm(2.15).twips)))
    pg_mar.set(qn("w:header"), str(int(Cm(1.27).twips)))
    pg_mar.set(qn("w:footer"), str(int(Cm(1.27).twips)))
    pg_mar.set(qn("w:gutter"), "0")
    sect.append(pg_mar)

    cols = OxmlElement("w:cols")
    cols.set(qn("w:num"), str(cols_num))
    cols.set(qn("w:space"), str(int(Cm(GUTTER_CM).twips)))
    sect.append(cols)

    return sect


def set_paragraph_section(paragraph, cols_num):
    ppr = paragraph._p.get_or_add_pPr()
    old = ppr.xpath("./w:sectPr")
    for node in old:
        ppr.remove(node)
    ppr.append(make_sect_pr(cols_num))


def has_drawing(paragraph):
    return bool(paragraph._p.xpath(".//w:drawing"))


def is_figure_caption(text):
    text = text.strip()
    return text.startswith("图 ") or text.startswith("图\t") or text.startswith("图")


def main():
    doc = Document(SRC)
    paragraphs = doc.paragraphs

    drawing_indices = [i for i, p in enumerate(paragraphs) if has_drawing(p)]

    for idx, shape in enumerate(doc.inline_shapes, start=1):
        if shape.width.cm:
            ratio = CONTENT_WIDTH_CM / shape.width.cm
            shape.width = Cm(CONTENT_WIDTH_CM)
            shape.height = Cm(shape.height.cm * ratio)

    for image_idx in drawing_indices:
        image_p = paragraphs[image_idx]
        image_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        image_p.paragraph_format.first_line_indent = None
        image_p.paragraph_format.space_before = Pt(4)
        image_p.paragraph_format.space_after = Pt(2)

        before_idx = image_idx - 1
        while before_idx >= 0 and not paragraphs[before_idx].text.strip() and not has_drawing(paragraphs[before_idx]):
            before_idx -= 1
        if before_idx >= 0:
            set_paragraph_section(paragraphs[before_idx], 2)

        caption_idx = image_idx + 1
        while caption_idx < len(paragraphs) and not paragraphs[caption_idx].text.strip():
            caption_idx += 1
        if caption_idx < len(paragraphs) and is_figure_caption(paragraphs[caption_idx].text):
            cap = paragraphs[caption_idx]
            cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
            cap.paragraph_format.first_line_indent = None
            cap.paragraph_format.space_before = Pt(2)
            cap.paragraph_format.space_after = Pt(6)
            set_paragraph_section(cap, 1)

    # Keep the final trailing section as the normal two-column article body.
    final_sect = doc.sections[-1]._sectPr
    cols = final_sect.xpath("./w:cols")
    if cols:
        cols[0].set(qn("w:num"), "2")
        cols[0].set(qn("w:space"), str(int(Cm(GUTTER_CM).twips)))

    doc.save(OUT)
    print(OUT)


if __name__ == "__main__":
    main()
