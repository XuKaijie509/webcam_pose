"""生成 A4 整页 PDF 标靶：ChArUco 占页面约 1/4（105x147 mm），1:1 打印。

用法:
    python webcam_pose/make_board.py

输出:
    webcam_pose/board/charuco_a4.pdf   打印用：A4 整页，CCITT G4 无损，1:1
    webcam_pose/board/charuco_a4.png   预览用（同样的版面）
    webcam_pose/board/charuco_a4.json  标定/位姿程序读取的参数

打印要求:
    A4 纸张；缩放选「实际大小 / 100%」，不要「适合页面」。
    打印后用尺量 100 mm 标尺：误差 >1 mm 说明被缩放，需重打，
    或把实测方格边长填进 json 的 actual_square_mm。
"""
import json
import os

import cv2
import numpy as np

try:
    from PIL import Image
except ImportError:
    Image = None

A4_MM = (210.0, 297.0)
DPI = 300
PPM = DPI / 25.4
SQUARES = (5, 7)
SQUARE_MM = 21.0
MARKER_MM = 15.75
DICT_NAME = "DICT_5X5_100"
BOARD_SIZE_MM = (SQUARES[0] * SQUARE_MM, SQUARES[1] * SQUARE_MM)  # 105 x 147
BOARD_ORIGIN_MM = ((A4_MM[0] - BOARD_SIZE_MM[0]) / 2.0,
                   (A4_MM[1] - BOARD_SIZE_MM[1]) / 2.0)          # 52.5, 75
RULER_LEN_MM = 100.0
RULER_X_MM = (A4_MM[0] - RULER_LEN_MM) / 2.0                         # 55
RULER_Y_MM = BOARD_ORIGIN_MM[1] + BOARD_SIZE_MM[1] + 12.0            # 234
OUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "board")


def mm2px(mm):
    return int(round(mm * PPM))


def build_board_image():
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, DICT_NAME))
    board = cv2.aruco.CharucoBoard(SQUARES, SQUARE_MM, MARKER_MM, dictionary)
    board_w = mm2px(BOARD_SIZE_MM[0])
    board_h = mm2px(BOARD_SIZE_MM[1])
    board_img = board.generateImage((board_w, board_h), 0, 1)
    if board_img.shape[:2] != (board_h, board_w):
        raise RuntimeError("unexpected board image shape %r" % (board_img.shape,))
    return board, board_img


def write_png(path, image):
    """用 imencode + 普通文件写盘，避开 cv2.imwrite 在中文路径下的失败。"""
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("imencode failed for %s" % path)
    with open(path, "wb") as f:
        f.write(buf.tobytes())


def draw_ruler(canvas):
    x0 = mm2px(RULER_X_MM)
    y0 = mm2px(RULER_Y_MM)
    x1 = x0 + mm2px(RULER_LEN_MM)
    cv2.line(canvas, (x0, y0), (x1, y0), 0, 3, cv2.LINE_AA)
    for mm in range(0, int(RULER_LEN_MM) + 1, 5):
        x = x0 + mm2px(mm)
        major = (mm % 10 == 0)
        h = mm2px(5.0) if major else mm2px(2.5)
        cv2.line(canvas, (x, y0), (x, y0 - h), 0, 3 if major else 2, cv2.LINE_AA)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canvas, "0", (x0 - mm2px(3), y0 + mm2px(15)), font, 1.0, 0, 2, cv2.LINE_AA)
    label = "100 mm"
    (tw, _), _ = cv2.getTextSize(label, font, 1.0, 2)
    cv2.putText(canvas, label, ((x0 + x1) // 2 - tw // 2, y0 + mm2px(15)),
                font, 1.0, 0, 2, cv2.LINE_AA)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    board, board_img = build_board_image()

    canvas_w, canvas_h = mm2px(A4_MM[0]), mm2px(A4_MM[1])
    canvas = np.full((canvas_h, canvas_w), 255, np.uint8)
    ox, oy = mm2px(BOARD_ORIGIN_MM[0]), mm2px(BOARD_ORIGIN_MM[1])
    bh, bw = board_img.shape[:2]
    canvas[oy:oy + bh, ox:ox + bw] = board_img

    draw_ruler(canvas)

    png = os.path.join(OUT_DIR, "charuco_a4.png")
    pdf = os.path.join(OUT_DIR, "charuco_a4.pdf")
    write_png(png, canvas)
    if Image is not None:
        # mode "1" 让 Pillow 用 CCITT G4 编码，打印无损；300 dpi -> 整页 A4
        pdf_img = Image.fromarray(canvas).convert("1", dither=Image.Dither.NONE)
        pdf_img.save(pdf, "PDF", resolution=DPI)
    else:
        raise RuntimeError("需要 Pillow 才能写 PDF：python -m pip install pillow")

    meta = {
        "dictionary": DICT_NAME,
        "squares_x": SQUARES[0],
        "squares_y": SQUARES[1],
        "square_mm": SQUARE_MM,
        "marker_mm": MARKER_MM,
        "actual_square_mm": SQUARE_MM,
        "dpi": DPI,
        "page_mm": list(A4_MM),
        "image_px": [canvas_w, canvas_h],
        "board_origin_mm": [round(BOARD_ORIGIN_MM[0], 3), round(BOARD_ORIGIN_MM[1], 3)],
        "board_size_mm": list(BOARD_SIZE_MM),
        "ruler": {"x_mm": RULER_X_MM, "y_mm": RULER_Y_MM, "length_mm": RULER_LEN_MM},
        "pdf": "charuco_a4.pdf",
        "png": "charuco_a4.png",
        "note": "A4 page, 1:1. Board is about 1/4 of the page (105x147 mm). "
                "Print at 100%/actual size, then measure the 100 mm bar; if off by "
                ">1 mm, reprint or put the measured square edge into actual_square_mm.",
    }
    json_path = os.path.join(OUT_DIR, "charuco_a4.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, ensure_ascii=False)

    print("pdf          : %s (A4, %d dpi, 1:1)" % (pdf, DPI))
    print("png preview  : %s" % png)
    print("board meta   : %s" % json_path)
    print("board        : %dx%d squares of %.1f mm = %.0fx%.0f mm (~1/4 A4)"
          % (SQUARES[0], SQUARES[1], SQUARE_MM, BOARD_SIZE_MM[0], BOARD_SIZE_MM[1]))
    print("rule bar     : 100 mm = %d px" % mm2px(RULER_LEN_MM))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
