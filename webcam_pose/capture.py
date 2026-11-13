"""最小摄像头闭环：取流 -> 显示 -> 实测 FPS -> 按 S 存图。

用法:
    python webcam_pose/capture.py
    python webcam_pose/capture.py --index 1 --out snap.png

按键:
    S  保存当前帧到 --out（默认 snap.png）
    Q / ESC  退出
"""
import argparse
import os
import sys
import time
from collections import deque

import cv2

BACKENDS = (("DSHOW", cv2.CAP_DSHOW), ("ANY", cv2.CAP_ANY), ("MSMF", cv2.CAP_MSMF))
WINDOW = "webcam_pose - capture"
REQUESTED = (1280, 720)
FALLBACK = (640, 480)


def open_camera(index):
    """依次尝试可用的后端，返回第一个能打开的 VideoCapture。"""
    for name, backend in BACKENDS:
        cap = cv2.VideoCapture(index, backend)
        if cap.isOpened():
            print("camera opened: index=%d backend=%s" % (index, name))
            return cap
        cap.release()
    return None


def set_resolution(cap, width, height):
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))


def save_frame(frame, path):
    """存图并打印文件大小、灰度均值，便于判断是否全黑。"""
    if not cv2.imwrite(path, frame):
        print("ERROR: failed to write %s" % path)
        return False
    size_kb = os.path.getsize(path) / 1024.0
    gray_mean = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
    print("saved %s (%.1f KB, gray mean %.1f)" % (path, size_kb, gray_mean))
    return True


def main(argv=None):
    parser = argparse.ArgumentParser(description="webcam minimal closed loop")
    parser.add_argument("--index", type=int, default=0, help="摄像头索引，默认 0")
    parser.add_argument("--out", default="snap.png", help="S 键保存的文件名")
    parser.add_argument("--frames", type=int, default=0,
                        help="跑满 N 帧后自动存图并退出（0=正常交互模式）")
    args = parser.parse_args(argv)
                    
    cap = open_camera(args.index)
    if cap is None:
        print("ERROR: cannot open camera index %d" % args.index)
        print("HINT: 检查 USB 摄像头是否插好、设备管理器是否有 Camera；")
        print("      或在有摄像头的机器上运行本脚本。")
        return 2

    width, height = set_resolution(cap, *REQUESTED)
    if width < 1000 or height < 700:
        width, height = set_resolution(cap, *FALLBACK)
    print("resolution: %dx%d" % (width, height))

    fps_window = deque(maxlen=61)
    saved = None
    nframes = 0
    last_print = 0.0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("ERROR: frame grab failed after %d frames" % nframes)
                break
            nframes += 1
            now = time.perf_counter()
            fps_window.append(now)
            fps = 0.0
            if len(fps_window) >= 2:
                dt = fps_window[-1] - fps_window[0]
                if dt > 0:
                    fps = (len(fps_window) - 1) / dt

            text = "%dx%d  %.1f FPS  [S save / Q quit]" % (width, height, fps)
            cv2.putText(frame, text, (8, 24), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 1, cv2.LINE_AA)
            cv2.imshow(WINDOW, frame)

            if now - last_print >= 2.0:
                print("%dx%d  %.1f FPS" % (width, height, fps))
                last_print = now

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            if key == ord("s"):
                saved = save_frame(frame, args.out) and args.out
            if args.frames and nframes >= args.frames:
                saved = save_frame(frame, args.out) and args.out
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("frames processed: %d" % nframes)
    if saved:
        print("saved snapshot: %s" % saved)
    return 0


if __name__ == "__main__":
    sys.exit(main())
