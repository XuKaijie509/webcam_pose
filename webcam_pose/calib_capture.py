"""标定图像采集：只负责把内参标定需要的帧存到 calib_frames/，不做任何求解。

标定分两步：
    python webcam_pose/calib_capture.py    # 采集 >= 15 个不同姿态的帧
    python webcam_pose/calib_solve.py      # 用这些帧求解内参 -> calib.yaml

采集要点（直接决定 calib_solve 的 RMS）:
    - 板占画面宽度的 60-80%，别只占 1/4；
    - 板要平整（贴在硬板上），纸面翘曲是最常见的 1 px 级误差来源；
    - 倾角在 ±30 度内分散，并覆盖画面四角；
    - 停稳再按 SPACE，避免运动模糊。

用法:
    python webcam_pose/calib_capture.py
    python webcam_pose/calib_capture.py --index 1 --frames-dir my_frames

按键:
    SPACE  采集当前姿态（至少 8 个角点）
    Q/ESC  退出

输出:
    calib_frames/view_001.png ...（原始彩色帧，供 calib_solve.py 求解与排查）
"""
import argparse
import glob
import os
import sys

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import calib_common as common
from capture import open_camera, set_resolution
from make_board import write_png

WINDOW = "webcam_pose - calib capture"
TARGET_W, TARGET_H = 1280, 720   # 内参按这个分辨率标定，pose_estimate 也只接受这个分辨率
FALLBACK_W, FALLBACK_H = 640, 480


def next_index(frames_dir):
    """返回下一个可用序号，避免覆盖已有帧。"""
    indices = []
    for path in glob.glob(os.path.join(frames_dir, "view_*.png")):
        stem = os.path.splitext(os.path.basename(path))[0]
        tail = stem.split("_")[-1]
        if tail.isdigit():
            indices.append(int(tail))
    return (max(indices) + 1) if indices else 1


def main(argv=None):
    parser = argparse.ArgumentParser(description="Capture frames for ChArUco calibration")
    parser.add_argument("--index", type=int, default=0, help="摄像头索引，默认 0")
    parser.add_argument("--frames-dir", default=common.FRAMES_DIR,
                        help="帧输出目录，默认 webcam_pose/calib_frames")
    parser.add_argument("--meta", default=common.BOARD_JSON, help="标定板描述 json")
    parser.add_argument("--min-corners", type=int, default=common.MIN_CORNERS,
                        help="单帧最少角点数，默认 %d" % common.MIN_CORNERS)
    args = parser.parse_args(argv)

    board, meta = common.load_board(args.meta)
    detector = cv2.aruco.CharucoDetector(board)
    print("board: %s  %dx%d squares, square %.1f mm"
          % (meta["dictionary"], meta["squares_x"], meta["squares_y"],
             float(meta.get("actual_square_mm", meta["square_mm"]))))

    cap = open_camera(args.index)
    if cap is None:
        print("ERROR: cannot open camera index %d" % args.index)
        return 2
    width, height = set_resolution(cap, TARGET_W, TARGET_H)
    if width < 1000 or height < 700:
        width, height = set_resolution(cap, FALLBACK_W, FALLBACK_H)
        print("WARNING: 相机只给 %dx%d；pose_estimate.py 要求 %dx%d，"
              "此分辨率下标定的内参无法用于实时位姿。" % (width, height, TARGET_W, TARGET_H))
    print("resolution: %dx%d" % (width, height))

    os.makedirs(args.frames_dir, exist_ok=True)
    existing = len(glob.glob(os.path.join(args.frames_dir, "view_*.png")))
    index = next_index(args.frames_dir)
    if existing:
        print("note: %s 已有 %d 张帧，本批从 view_%03d.png 续写（不覆盖旧帧）。"
              % (args.frames_dir, existing, index))
        print("      如需全新一批，请先清空该目录或换 --frames-dir。")

    n_saved = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("ERROR: frame grab failed")
                break

            corners, ids, marker_corners, marker_ids = common.detect_board(detector, frame)
            vis = frame.copy()
            if marker_corners is not None and marker_ids is not None:
                cv2.aruco.drawDetectedMarkers(vis, marker_corners, marker_ids)
            if corners is not None:
                for p in corners:
                    cv2.circle(vis, (int(p[0]), int(p[1])), 3, (0, 0, 255), -1)
            n_corners = 0 if corners is None else len(corners)
            color = (0, 255, 0) if n_corners >= args.min_corners else (0, 165, 255)
            text = "captured %d  corners %d/%d   [SPACE] capture   [Q] quit" % (
                n_saved, n_corners, len(board.getChessboardCorners()))
            cv2.putText(vis, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, color, 1, cv2.LINE_AA)
            cv2.imshow(WINDOW, vis)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord(" "):
                if n_corners < args.min_corners:
                    print("not captured: need >= %d corners, got %d"
                          % (args.min_corners, n_corners))
                    continue
                path = os.path.join(args.frames_dir, "view_%03d.png" % index)
                write_png(path, frame)
                print("saved %s (%d corners)  total %d"
                      % (os.path.basename(path), n_corners, n_saved + 1))
                index += 1
                n_saved += 1
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("captured %d frame(s) this run, need >= %d in %s"
          % (n_saved, common.MIN_VIEWS, args.frames_dir))
    if n_saved < common.MIN_VIEWS:
        print("HINT: 帧数不足 %d，标定结果不可靠；继续采集。" % common.MIN_VIEWS)
    else:
        print("next: python webcam_pose/calib_solve.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
