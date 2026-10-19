#!/usr/bin/env python3
"""Print-to-scale PDF of the Conv-ChArT board on one A4 page, drawn as vector squares.

    python3 tools/board_pdf.py --square-mm 40 --out docs/board_5x5_200mm_A4.pdf

The pattern is cv2.aruco.CharucoBoard((5, 5), 1.0, 0.7, DICT_5X5_50), Conv-ChArT's dcc.board.get_board() and the
layout the deployed detector was trained on. OpenCV renders it and each 1/20-square cell of that render is traced
into the PDF, so the print is exactly that board at the requested size. Needs cv2 and numpy (the p4p-gpu container).
"""
import argparse
import zlib

import cv2
import numpy as np

A4_MM = (210.0, 297.0)
PT_PER_MM = 72.0 / 25.4
SIDE_MARGIN_MM = 5.0
TOP_MM = 10.0
CELLS_PER_SQUARE = 20


def board_cells() -> np.ndarray:
    """Return the board as a (100, 100) uint8 grid, 0 black and 255 white.

    The grid has one entry per 1/20 of a square.
    """
    board = cv2.aruco.CharucoBoard((5, 5), 1.0, 0.7, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_50))
    px = 5
    img = board.generateImage((5 * CELLS_PER_SQUARE * px,) * 2, marginSize=0, borderBits=1)
    assert set(np.unique(img).tolist()) <= {0, 255}, "the render is not binary"
    cells = img[px // 2::px, px // 2::px]
    assert np.array_equal(np.repeat(np.repeat(cells, px, 0), px, 1), img), "the render does not sit on the cell grid"
    return cells


def _text(s: str) -> str:
    return s.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")


def page_content(cells: np.ndarray, square_mm: float) -> bytes:
    W, H = A4_MM
    board_mm = 5 * square_mm
    x0 = (W - board_mm) / 2
    cell = square_mm / CELLS_PER_SQUARE

    def X(mm):
        return mm * PT_PER_MM

    def Y(mm):
        return (H - mm) * PT_PER_MM

    rects, open_runs = [], {}
    for r, row in enumerate(np.vstack((cells, np.full((1, cells.shape[1]), 255, np.uint8)))):
        runs, c = set(), 0
        while c < len(row):
            if row[c]:
                c += 1
                continue
            start = c
            while c < len(row) and not row[c]:
                c += 1
            runs.add((start, c))
        for run in [k for k in open_runs if k not in runs]:
            rects.append((*run, open_runs.pop(run), r))
        for run in runs - open_runs.keys():
            open_runs[run] = r
    ops = ["0 g"]
    for c0, c1, r0, r1 in sorted(rects, key=lambda t: (t[2], t[0])):
        ops.append(f"{X(x0 + c0 * cell):.3f} {Y(TOP_MM + r1 * cell):.3f} "
                   f"{(c1 - c0) * cell * PT_PER_MM:.3f} {(r1 - r0) * cell * PT_PER_MM:.3f} re")
    ops.append("f")

    ruler_y = TOP_MM + board_mm + 9
    ops.append("0 G 0.5 w")
    ops.append(f"{X(x0):.3f} {Y(ruler_y):.3f} m {X(x0 + 100):.3f} {Y(ruler_y):.3f} l S")
    ops.append("0.35 w")
    for mm in range(101):
        tick = 3.5 if mm % 10 == 0 else 2.2 if mm % 5 == 0 else 1.2
        ops.append(f"{X(x0 + mm):.3f} {Y(ruler_y):.3f} m {X(x0 + mm):.3f} {Y(ruler_y - tick):.3f} l S")
    for mm in range(0, 101, 10):
        ops.append(f"BT /F1 6 Tf {X(x0 + mm) - (1.7 if mm < 10 else 3.3 if mm < 100 else 5.0):.3f} "
                   f"{Y(ruler_y + 3.2):.3f} Td ({mm}) Tj ET")
    ops.append(f"BT /F1 6 Tf {X(x0 + 103):.3f} {Y(ruler_y + 0.8):.3f} Td (mm) Tj ET")

    sq = f"{square_mm:.1f}"
    m = f"{square_mm / 1000:.3f}"
    lines = [
        ("F2", 12, f"Conv-ChArT docking board: 5 x 5 ChArUco, {board_mm:.0f} x {board_mm:.0f} mm"),
        ("F1", 8.5, f"DICT_5X5_50, markers 0-11, square {sq} mm, marker {0.7 * square_mm:.1f} mm (ratio 0.7), "
                    "16 inner corners."),
        ("F1", 8.5, f"Config: cfg.yaml BOARD {{squares: [5, 5], square_length_m: {m}}}; controller dock_demo.yaml "
                    f"board_square_m: {m}."),
        ("F2", 8.5, "Print on A4 at 100% / Actual size, with 'Fit to page' and 'Shrink oversized pages' turned off."),
        ("F1", 8.5, f"Then check with a ruler: the bar above must read 100 mm and the board {board_mm:.0f} mm "
                    f"(each square {sq} mm). If a square"),
        ("F1", 8.5, "measures anything else, put the measured size (in metres) into both config entries: "
                    "metric pose scales with it."),
        ("F1", 8.5, f"The board runs to {x0:.0f} mm from the paper's sides. A printer that clips a millimetre "
                    "of the outer black squares"),
        ("F1", 8.5, f"does no harm: the 16 corners are {sq} mm in from the board's edge."),
        ("F1", 8.5, "Mount flat on a rigid backing (foam board), board centre at the camera lens height. "
                    "Matte paper, no gloss or"),
        ("F1", 8.5, "laminate. If it is inkjet-printed, check the black squares still look black in the camera "
                    "image."),
        ("F1", 8.5, "As printed, corner 0 is the top-left inner corner (board x to the right, y down)."),
    ]
    y = ruler_y + 14
    for font, size, s in lines:
        ops.append(f"BT /{font} {size} Tf {X(x0):.3f} {Y(y):.3f} Td ({_text(s)}) Tj ET")
        y += 6.5 if font == "F2" and size > 10 else 4.4
    ops.append("0.45 g")
    ops.append(f"BT /F1 6.5 Tf {X(x0):.3f} {Y(H - 9):.3f} Td ("
               + _text("Generated by tools/board_pdf.py from cv2.aruco.CharucoBoard((5, 5), 1.0, 0.7, DICT_5X5_50) "
                       "(Conv-ChArT dcc.board.get_board()), the layout the deployed models were trained on.")
               + ") Tj ET")
    return "\n".join(ops).encode("ascii")


def write_pdf(path: str, content: bytes, title: str) -> None:
    stream = zlib.compress(content, 9)
    w, h = (v * PT_PER_MM for v in A4_MM)
    objects = [
        b"<< /Type /Catalog /Pages 2 0 R >>",
        b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>",
        (f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w:.4f} {h:.4f}] /Contents 4 0 R "
         f"/Resources << /Font << /F1 5 0 R /F2 6 0 R >> >> >>").encode(),
        b"<< /Length %d /Filter /FlateDecode >>\nstream\n" % len(stream) + stream + b"\nendstream",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica /Encoding /WinAnsiEncoding >>",
        b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica-Bold /Encoding /WinAnsiEncoding >>",
        f"<< /Title ({_text(title)}) /Author (Conv-ChArT) /Producer (tools/board_pdf.py) >>".encode(),
    ]
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, body in enumerate(objects, 1):
        offsets.append(len(out))
        out += b"%d 0 obj\n" % i + body + b"\nendobj\n"
    xref = len(out)
    out += b"xref\n0 %d\n0000000000 65535 f \n" % (len(objects) + 1)
    out += b"".join(b"%010d 00000 n \n" % o for o in offsets)
    out += b"trailer\n<< /Size %d /Root 1 0 R /Info 7 0 R >>\nstartxref\n%d\n%%%%EOF\n" % (len(objects) + 1, xref)
    with open(path, "wb") as f:
        f.write(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--square-mm", type=float, default=40.0, help="printed square edge in mm (board = 5 squares)")
    ap.add_argument("--out", required=True, help="output PDF path")
    a = ap.parse_args()
    board_mm = 5 * a.square_mm
    if board_mm > A4_MM[0] - 2 * SIDE_MARGIN_MM + 1e-9:
        ap.error(f"a {board_mm:g} mm board leaves under {SIDE_MARGIN_MM:g} mm at the sides of A4")
    write_pdf(a.out, page_content(board_cells(), a.square_mm),
              f"Conv-ChArT board 5x5 DICT_5X5_50, {board_mm:g} mm")
    print(f"wrote {a.out}: {board_mm:g} mm board, {a.square_mm:g} mm squares, A4")


if __name__ == "__main__":
    main()
