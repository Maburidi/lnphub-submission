#!/usr/bin/env python3
"""
Assemble Figure 1 panels for journal submission.

Outputs:
  - Figure1_final.pdf
  - Figure1_final_600dpi.png
  - Figure1_final_preview.png
"""

from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.utils import ImageReader
from reportlab.pdfgen import canvas


BASE = Path(__file__).resolve().parent

PANEL_FILES = {
    "a": BASE / "Fig1a.png",
    "b": BASE / "Fig1b.png",
    "c": BASE / "Fig1c.png",
    "d": BASE / "Fig1d.png",
    "e": BASE / "Fig1e.png",
    "f": BASE / "Fig1f.png",
    "g": BASE / "Fig1g.png",
}

OUT_PDF = BASE / "Figure1_final.pdf"
OUT_PNG = BASE / "Figure1_final_600dpi.png"
OUT_PREVIEW = BASE / "Figure1_final_preview.png"


# Nature double-column width is commonly 180 mm.
PAGE_WIDTH_MM = 180.0
DPI = 600

# The important spacing controls.
OUTER_MARGIN_MM = 3.0
TOP_MARGIN_MM = 7.0
BOTTOM_MARGIN_MM = 3.0
PANEL_GAP_MM = 4.0
ROW_GAP_MM = 5.5

# Keep a little white border inside each cropped panel so neighboring panels
# do not visually touch after cropping.
CROP_PADDING_PX = 16
WHITE_THRESHOLD = 250

# Scale an entire row while preserving aspect ratios and alignment.
ROW_SCALE = {
    1: 1.00,  # a + b
    2: 1.00,  # c + d
    3: 0.98,  # e
    4: 0.94,  # f + g; slightly reduced because these panels are dense
}

DRAW_PANEL_LABELS = True
LABEL_FONT_SIZE_PT = 12
LABEL_X_OFFSET_MM = -1.6
LABEL_Y_OFFSET_MM = -4.4


MM_TO_PT = 72.0 / 25.4
MM_TO_PX = DPI / 25.4


def mm_to_px(value):
    return int(round(value * MM_TO_PX))


def crop_to_content(image_path):
    """Crop exterior white margins while preserving a small safety padding."""
    img = Image.open(image_path).convert("RGB")
    arr = np.asarray(img)
    mask = np.any(arr < WHITE_THRESHOLD, axis=2)
    ys, xs = np.where(mask)

    if len(xs) == 0:
        return img

    x0 = max(0, int(xs.min()) - CROP_PADDING_PX)
    y0 = max(0, int(ys.min()) - CROP_PADDING_PX)
    x1 = min(img.width, int(xs.max()) + 1 + CROP_PADDING_PX)
    y1 = min(img.height, int(ys.max()) + 1 + CROP_PADDING_PX)
    return img.crop((x0, y0, x1, y1))


def load_panels():
    missing = [str(path) for path in PANEL_FILES.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing panel file(s):\n" + "\n".join(missing))

    panels = {}
    for label, path in PANEL_FILES.items():
        cropped = crop_to_content(path)
        panels[label] = cropped
        print(
            f"{label}: original={Image.open(path).size}, "
            f"cropped={cropped.size}, aspect={cropped.width / cropped.height:.3f}"
        )
    return panels


def row_geometry(labels, panels, available_width_mm, gap_mm, scale=1.0):
    total_aspect = sum(panels[label].width / panels[label].height for label in labels)
    usable_width_mm = available_width_mm - gap_mm * (len(labels) - 1)
    height_mm = usable_width_mm / total_aspect
    widths_mm = {
        label: height_mm * panels[label].width / panels[label].height
        for label in labels
    }

    height_mm *= scale
    widths_mm = {label: width * scale for label, width in widths_mm.items()}
    return height_mm, widths_mm


def get_label_font():
    size_px = max(1, int(round(LABEL_FONT_SIZE_PT * DPI / 72.0)))
    candidates = [
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/System/Library/Fonts/Supplemental/Helvetica Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
    ]
    for font_path in candidates:
        if Path(font_path).exists():
            return ImageFont.truetype(font_path, size_px)
    return ImageFont.load_default()


def paste_panel(canvas_img, panel, box_mm):
    x_mm, y_mm, w_mm, h_mm = box_mm
    x = mm_to_px(x_mm)
    y = mm_to_px(y_mm)
    w = mm_to_px(w_mm)
    h = mm_to_px(h_mm)
    resized = panel.resize((w, h), Image.Resampling.LANCZOS)
    canvas_img.paste(resized, (x, y))


def draw_panel_label(draw, label, box_mm, font):
    if not DRAW_PANEL_LABELS:
        return

    x_mm, y_mm, _, _ = box_mm
    x = mm_to_px(x_mm + LABEL_X_OFFSET_MM)
    y = mm_to_px(y_mm + LABEL_Y_OFFSET_MM)
    draw.text((x, y), label, fill=(0, 0, 0), font=font)


def save_pdf_from_png(png_path, pdf_path, page_width_mm, page_height_mm):
    page_width_pt = page_width_mm * MM_TO_PT
    page_height_pt = page_height_mm * MM_TO_PT

    c = canvas.Canvas(str(pdf_path), pagesize=(page_width_pt, page_height_pt))
    c.drawImage(
        ImageReader(str(png_path)),
        0,
        0,
        width=page_width_pt,
        height=page_height_pt,
        preserveAspectRatio=False,
        mask="auto",
    )
    c.showPage()
    c.save()


def main():
    panels = load_panels()
    inner_width_mm = PAGE_WIDTH_MM - 2 * OUTER_MARGIN_MM

    row1 = ["a", "b"]
    row2 = ["c", "d"]
    row4 = ["f", "g"]

    row1_h, row1_w = row_geometry(row1, panels, inner_width_mm, PANEL_GAP_MM, ROW_SCALE[1])
    row2_h, row2_w = row_geometry(row2, panels, inner_width_mm, PANEL_GAP_MM, ROW_SCALE[2])
    row3_w = inner_width_mm * ROW_SCALE[3]
    row3_h = row3_w / (panels["e"].width / panels["e"].height)
    row4_h, row4_w = row_geometry(row4, panels, inner_width_mm, PANEL_GAP_MM, ROW_SCALE[4])

    page_height_mm = (
        TOP_MARGIN_MM
        + row1_h
        + ROW_GAP_MM
        + row2_h
        + ROW_GAP_MM
        + row3_h
        + ROW_GAP_MM
        + row4_h
        + BOTTOM_MARGIN_MM
    )

    print(f"\nFinal figure size: {PAGE_WIDTH_MM:.1f} mm x {page_height_mm:.1f} mm")
    print(f"Panel gap: {PANEL_GAP_MM:.1f} mm; row gap: {ROW_GAP_MM:.1f} mm")

    canvas_img = Image.new(
        "RGB",
        (mm_to_px(PAGE_WIDTH_MM), mm_to_px(page_height_mm)),
        "white",
    )
    draw = ImageDraw.Draw(canvas_img)
    label_font = get_label_font()

    placements = []
    y = TOP_MARGIN_MM

    def add_two_panel_row(labels, widths_mm, height_mm):
        nonlocal y
        total_width_mm = sum(widths_mm[label] for label in labels) + PANEL_GAP_MM
        x = OUTER_MARGIN_MM + (inner_width_mm - total_width_mm) / 2
        for label in labels:
            box = (x, y, widths_mm[label], height_mm)
            placements.append((label, box))
            x += widths_mm[label] + PANEL_GAP_MM
        y += height_mm + ROW_GAP_MM

    add_two_panel_row(row1, row1_w, row1_h)
    add_two_panel_row(row2, row2_w, row2_h)

    x_e = OUTER_MARGIN_MM + (inner_width_mm - row3_w) / 2
    placements.append(("e", (x_e, y, row3_w, row3_h)))
    y += row3_h + ROW_GAP_MM

    add_two_panel_row(row4, row4_w, row4_h)

    for label, box in placements:
        paste_panel(canvas_img, panels[label], box)
        draw_panel_label(draw, label, box, label_font)

    canvas_img.save(OUT_PNG, dpi=(DPI, DPI))
    save_pdf_from_png(OUT_PNG, OUT_PDF, PAGE_WIDTH_MM, page_height_mm)

    preview_width_px = 1800
    preview_height_px = int(round(preview_width_px * canvas_img.height / canvas_img.width))
    preview = canvas_img.resize((preview_width_px, preview_height_px), Image.Resampling.LANCZOS)
    preview.save(OUT_PREVIEW)

    print(f"\nSaved PDF: {OUT_PDF}")
    print(f"Saved 600-dpi PNG: {OUT_PNG}")
    print(f"Saved preview: {OUT_PREVIEW}")


if __name__ == "__main__":
    main()
