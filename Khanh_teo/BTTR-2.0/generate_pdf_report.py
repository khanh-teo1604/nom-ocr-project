import os
import shutil
from pathlib import Path
from PIL import Image as PILImage

from reportlab.lib.pagesizes import letter, A4
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    Table,
    TableStyle,
    Image as RLImage,
    KeepTogether,
    HRFlowable,
)
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont


def create_pdf(
    output_pdf_path: str,
    results_data: list = None,
    ckpt_name: str = "epoch=11-step=504660-val_ExpRate=0.9120.ckpt",
    dataset_name: str = "MTH1200 (Cropped Nom Characters)",
    images_dir: Path = None,
):
    font_path = "/home/projectg7/.local/share/fonts/BabelStoneHan.ttf"
    if not os.path.exists(font_path):
        raise FileNotFoundError(f"Font not found at {font_path}")

    pdfmetrics.registerFont(TTFont("BabelStoneHan", font_path))

    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=A4,
        leftMargin=36,
        rightMargin=36,
        topMargin=36,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()

    title_style = ParagraphStyle(
        "DocTitle",
        fontName="BabelStoneHan",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#1e3a8a"),
        alignment=1,  # Center
        spaceAfter=6,
    )
    subtitle_style = ParagraphStyle(
        "DocSubtitle",
        fontName="BabelStoneHan",
        fontSize=10,
        leading=15,
        textColor=colors.HexColor("#334155"),
        spaceAfter=3,
    )
    meta_box_label = ParagraphStyle(
        "MetaBoxLabel",
        fontName="BabelStoneHan",
        fontSize=10,
        leading=14,
        textColor=colors.HexColor("#1e293b"),
    )
    table_header_style = ParagraphStyle(
        "TableHeader",
        fontName="BabelStoneHan",
        fontSize=10,
        leading=12,
        textColor=colors.white,
        alignment=1,
    )
    cell_center = ParagraphStyle(
        "CellCenter",
        fontName="BabelStoneHan",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#1e293b"),
        alignment=1,
    )
    cell_fname = ParagraphStyle(
        "CellFname",
        fontName="BabelStoneHan",
        fontSize=8,
        leading=10,
        textColor=colors.HexColor("#475569"),
        alignment=0,
    )
    cell_ids_gt = ParagraphStyle(
        "CellIdsGt",
        fontName="BabelStoneHan",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
        alignment=1,
    )
    cell_ids_pred = ParagraphStyle(
        "CellIdsPred",
        fontName="BabelStoneHan",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#0f172a"),
        alignment=1,
    )
    match_badge_style = ParagraphStyle(
        "MatchBadge",
        fontName="BabelStoneHan",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#166534"),
        alignment=1,
    )
    diff_badge_style = ParagraphStyle(
        "DiffBadge",
        fontName="BabelStoneHan",
        fontSize=9,
        leading=12,
        textColor=colors.HexColor("#991b1b"),
        alignment=1,
    )

    story = []

    # Title
    story.append(Paragraph("BTTR Model Evaluation Report", title_style))
    story.append(Paragraph(f"Test on Sample from {dataset_name}", ParagraphStyle(
        "SubHeader",
        fontName="BabelStoneHan",
        fontSize=11,
        leading=15,
        textColor=colors.HexColor("#475569"),
        alignment=1,
        spaceAfter=12,
    )))

    story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#cbd5e1"), spaceAfter=12))

    if results_data is None:
        results_data = [
            {"idx": 1, "filename": "GL_924_2_23_0441_某.png", "gt_ids": "⿱甘木", "pred_ids": "⿳匕⿺𠃊目八", "match": False, "dist": 6},
            {"idx": 2, "filename": "QL_7_513_0212_果.png", "gt_ids": "⿻日木", "pred_ids": "⿻日木", "match": True, "dist": 0},
            {"idx": 3, "filename": "QL_14_17_0188_諸.png", "gt_ids": "⿰言者", "pred_ids": "⿰言者", "match": True, "dist": 0},
            {"idx": 4, "filename": "QL_13_559_0294_據.png", "gt_ids": "⿰扌豦", "pred_ids": "⿰扌𡊮", "match": False, "dist": 1},
            {"idx": 5, "filename": "GL_1056_2_24_0245_丘.png", "gt_ids": "丘", "pred_ids": "丘", "match": True, "dist": 0},
            {"idx": 6, "filename": "QL_12_577_0078_三.png", "gt_ids": "⿱一二", "pred_ids": "⿱一二", "match": True, "dist": 0},
            {"idx": 7, "filename": "QL_10_145_0072_亦.png", "gt_ids": "⿱亠④", "pred_ids": "⿱亠④", "match": True, "dist": 0},
            {"idx": 8, "filename": "YB_28_79_0210_口.png", "gt_ids": "口", "pred_ids": "口", "match": True, "dist": 0},
            {"idx": 9, "filename": "QL_3_340_0307_羅.png", "gt_ids": "⿱罒維", "pred_ids": "⿱罒維", "match": True, "dist": 0},
            {"idx": 10, "filename": "GL_62_1_15_0207_反.png", "gt_ids": "⿸𠂆又", "pred_ids": "⿸𠂆又", "match": True, "dist": 0},
        ]

    total = len(results_data)
    matches = sum(1 for r in results_data if r["match"])
    accuracy = (matches / total) * 100 if total > 0 else 0
    mean_dist = sum(r.get("dist", r.get("edit_dist", 0)) for r in results_data) / total if total > 0 else 0

    # Metadata card / summary table
    meta_data = [
        [
            Paragraph(f"<b>Model Checkpoint:</b> {ckpt_name}", subtitle_style),
            Paragraph(f"<b>Exact Match (ExpRate):</b> <font color='#166534'><b>{matches} / {total} ({accuracy:.1f}%)</b></font>", subtitle_style),
        ],
        [
            Paragraph(f"<b>Dataset:</b> {dataset_name}", subtitle_style),
            Paragraph(f"<b>Avg Levenshtein Distance:</b> <b>{mean_dist:.2f}</b>", subtitle_style),
        ],
        [
            Paragraph("<b>Task:</b> Ideographic Description Sequence (IDS) Recognition", subtitle_style),
            Paragraph("<b>Decoding Method:</b> Bi-directional Beam Search (beam_size=10)", subtitle_style),
        ],
    ]
    meta_table = Table(meta_data, colWidths=[280, 240])
    meta_table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ("BOX", (0, 0), (-1, -1), 1, colors.HexColor("#e2e8f0")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 6),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
        ("LEFTPADDING", (0, 0), (-1, -1), 10),
        ("RIGHTPADDING", (0, 0), (-1, -1), 10),
    ]))
    story.append(meta_table)
    story.append(Spacer(1, 14))

    # Main comparison table
    story.append(Paragraph("<b>Side-by-Side Sample Comparison</b>", ParagraphStyle(
        "SectionHeading",
        fontName="BabelStoneHan",
        fontSize=12,
        leading=16,
        textColor=colors.HexColor("#1e293b"),
        spaceAfter=8,
    )))

    headers = [
        Paragraph("<b>#</b>", table_header_style),
        Paragraph("<b>Crop</b>", table_header_style),
        Paragraph("<b>Sample Filename</b>", table_header_style),
        Paragraph("<b>Ground Truth IDS</b>", table_header_style),
        Paragraph("<b>Predicted IDS</b>", table_header_style),
        Paragraph("<b>Status</b>", table_header_style),
        Paragraph("<b>Edit Dist</b>", table_header_style),
    ]

    table_rows = [headers]

    if images_dir is None:
        images_dir = Path("/home/projectg7/.gemini/antigravity/brain/6e28f53b-d95c-454a-9d5c-0e0fa9164fe2/sample_images")

    table_styles = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1e3a8a")),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("LEFTPADDING", (0, 0), (-1, -1), 4),
        ("RIGHTPADDING", (0, 0), (-1, -1), 4),
    ]

    for row_idx, item in enumerate(results_data, start=1):
        # Image
        img_file = images_dir / item["filename"]
        if img_file.exists():
            with PILImage.open(img_file) as im:
                w, h = im.size
                aspect = w / h
                max_box = 32
                if aspect >= 1:
                    draw_w = max_box
                    draw_h = max_box / aspect
                else:
                    draw_h = max_box
                    draw_w = max_box * aspect
            rl_img = RLImage(str(img_file), width=draw_w, height=draw_h)
        else:
            rl_img = Paragraph("-", cell_center)

        status_text = "MATCH" if item["match"] else "DIFF"
        status_para = Paragraph(
            f"<b>{status_text}</b>",
            match_badge_style if item["match"] else diff_badge_style,
        )

        dist_val = item.get("dist", item.get("edit_dist", 0))
        row = [
            Paragraph(str(item["idx"]), cell_center),
            rl_img,
            Paragraph(item["filename"], cell_fname),
            Paragraph(item["gt_ids"], cell_ids_gt),
            Paragraph(item["pred_ids"], cell_ids_pred),
            status_para,
            Paragraph(str(dist_val), cell_center),
        ]
        table_rows.append(row)

        bg_color = colors.HexColor("#ffffff") if row_idx % 2 == 1 else colors.HexColor("#f8fafc")
        table_styles.append(("BACKGROUND", (0, row_idx), (-1, row_idx), bg_color))

        if not item["match"]:
            table_styles.append(("BACKGROUND", (5, row_idx), (5, row_idx), colors.HexColor("#fef2f2")))
        else:
            table_styles.append(("BACKGROUND", (5, row_idx), (5, row_idx), colors.HexColor("#f0fdf4")))

    # Col widths: total 520 pt
    col_widths = [24, 46, 125, 105, 120, 58, 42]
    main_table = Table(table_rows, colWidths=col_widths, repeatRows=1)
    main_table.setStyle(TableStyle(table_styles))
    story.append(main_table)

    story.append(Spacer(1, 14))
    story.append(Paragraph(
        "<i>Note: Model outputs Ideographic Description Sequences (IDS) representing the compositional components of Nom characters.</i>",
        ParagraphStyle(
            "FooterNote",
            fontName="BabelStoneHan",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#64748b"),
            alignment=0,
        )
    ))

    doc.build(story)
    print(f"PDF successfully created at: {output_pdf_path}")


if __name__ == "__main__":
    out_path_1 = "/home/projectg7/workspace/Week6_Recognition/Project Dr.Cuong/BTTR-2.0/mth1200_evaluation_report.pdf"
    out_path_2 = "/home/projectg7/.gemini/antigravity/brain/6e28f53b-d95c-454a-9d5c-0e0fa9164fe2/mth1200_evaluation_report.pdf"

    create_pdf(out_path_1)
    shutil.copy2(out_path_1, out_path_2)
    print(f"Copied to artifact directory: {out_path_2}")

