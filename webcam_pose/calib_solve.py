"""从已采集的帧求解相机内参：读 calib_frames/ 里的 PNG，检测角点并标定。

标定分两步（本文件是第二步）:
    python webcam_pose/calib_capture.py    # 采集 >= 15 个不同姿态的帧
    python webcam_pose/calib_solve.py      # 用这些帧求解内参 -> calib.yaml

适用场景:
    - 采集完帧后单独求解，采集过程不阻塞；
    - 换板参数 / 换 RMS 门槛后复标定，不必重拍；
    - 排查是哪一张图把整体 RMS 拉高（--drop-worst 逐张剔除）。

用法:
    python webcam_pose/calib_solve.py
    python webcam_pose/calib_solve.py --drop-worst 2
    python webcam_pose/calib_solve.py --gate 1.0

输入:
    --frames-dir 下的 PNG 帧（默认 webcam_pose/calib_frames/view_*.png）
    --meta 标定板描述（默认 board/charuco_a4.json）

输出:
    calib.yaml         相机内参 + 畸变（cv2.FileStorage 可读，pose_estimate.py 直接使用）
    calib_report.json  总体 RMS、每帧 RMS、passed（质检用）

退出码:
    0  RMS <  --gate
    1  RMS >= --gate
    2  帧目录不存在 / 可读帧不足 / 读图或写盘失败
"""
import argparse
import glob
import json
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import calib_common as common


def read_image(path):
    """读图并返回 BGR 帧。

    用 np.fromfile + imdecode 而不是 cv2.imread：本仓库路径含中文，
    imread 在 Windows 上会静默返回 None（make_board.write_png 有同类规避）。
    """
    try:
        data = np.fromfile(path, dtype=np.uint8)
    except OSError as exc:
        raise RuntimeError("cannot read %s: %s" % (path, exc))
    if data.size == 0:
        raise RuntimeError("empty file: %s" % path)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError("not a decodable image: %s" % path)
    return image


def list_frames(frames_dir, pattern="*.png"):
    """按文件名排序返回帧路径列表（view_001.png ... view_015.png 顺序稳定）。"""
    if not os.path.isdir(frames_dir):
        raise RuntimeError("frames dir not found: %s" % frames_dir)
    return sorted(glob.glob(os.path.join(frames_dir, pattern)))


def collect_entries(detector, paths, min_corners):
    """逐帧检测 ChArUco 角点。

    返回 (entries, skipped, image_size)：
        entries    [(path, frame, corners, ids), ...] 参加标定的帧
        skipped    [(path, corner_count), ...] 角点太少被丢弃的帧
        image_size (w, h)，取自帧的真实尺寸而不是相机的 CAP_PROP
    """
    entries, skipped = [], []
    image_size = None
    for path in paths:
        name = os.path.basename(path)
        frame = read_image(path)
        h, w = frame.shape[:2]
        if image_size is None:
            image_size = (w, h)
        elif (w, h) != image_size:
            raise RuntimeError("frame size mismatch: %s is %dx%d, expected %dx%d"
                               % (name, w, h, image_size[0], image_size[1]))

        corners, ids, _, _ = common.detect_board(detector, frame)
        got = 0 if corners is None else len(corners)
        if ids is None or got < min_corners:
            skipped.append((path, got))
            print("skip %-15s corners %2d < %d" % (name, got, min_corners))
            continue
        entries.append((path, frame, corners, ids))
        print("load %-15s corners %2d" % (name, got))
    return entries, skipped, image_size


def to_views(entries):
    """转成 calibrate_from_views 需要的 (frame, corners, ids) 列表。"""
    return [(frame, corners, ids) for _, frame, corners, ids in entries]


def calibrate_from_views(board, views, image_size):
    """纯函数：由若干视图求内参，不依赖相机，便于离线/合成数据自检。"""
    chess = board.getChessboardCorners()
    obj_points, img_points = [], []
    for _, corners, ids in views:
        obj = chess[ids].astype(np.float32).reshape(-1, 1, 3)
        img = corners.astype(np.float32).reshape(-1, 1, 2)
        obj_points.append(obj)
        img_points.append(img)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 200, 1e-6)
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        obj_points, img_points, image_size, None, None, criteria=criteria)

    per_view = []
    for i, (obj, img) in enumerate(zip(obj_points, img_points)):
        proj, _ = cv2.projectPoints(obj, rvecs[i], tvecs[i], K, dist)
        err = np.linalg.norm(proj.reshape(-1, 2) - img.reshape(-1, 2), axis=1)
        per_view.append(float(np.sqrt(np.mean(err ** 2))))

    return {
        "rms": float(rms),
        "K": K,
        "dist": dist.reshape(-1),
        "per_view_rms": per_view,
        "image_size": list(image_size),
        "n_views": len(views),
    }


def save_report(path, result, meta):
    """写 JSON 质检报告（含 rms_gate/passed，让报告自带判定依据）。"""
    report = {
        "rms": result["rms"],
        "rms_gate": common.RMS_GATE,
        "passed": result["rms"] < common.RMS_GATE,
        "n_views": result["n_views"],
        "image_size": result["image_size"],
        "per_view_rms": result["per_view_rms"],
        "camera_matrix": result["K"].tolist(),
        "distortion": result["dist"].tolist(),
        "square_mm": float(meta.get("actual_square_mm", meta["square_mm"])),
        "squares": [int(meta["squares_x"]), int(meta["squares_y"])],
        "dictionary": meta["dictionary"],
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def print_table(entries, per_view_rms):
    """按单图 RMS 从大到小打印，最差的一行在最上面，便于定位坏帧。"""
    print("%-16s %6s %9s" % ("frame", "corner", "rms(px)"))
    order = sorted(range(len(entries)), key=lambda i: per_view_rms[i], reverse=True)
    for i in order:
        print("%-16s %6d %9.3f"
              % (os.path.basename(entries[i][0]), len(entries[i][2]), per_view_rms[i]))


def drop_worst(entries, board, image_size, n_drop, min_keep=3):
    """剔除 per-view RMS 最大的帧，每剔一张重标一次，返回 (entries, result, dropped)。"""
    dropped = []
    result = calibrate_from_views(board, to_views(entries), image_size)
    while len(dropped) < n_drop and len(entries) - 1 >= min_keep:
        worst = max(range(len(entries)), key=lambda i: result["per_view_rms"][i])
        before = result["rms"]
        path, _, _, _ = entries.pop(worst)
        dropped.append((path, result["per_view_rms"][worst]))
        result = calibrate_from_views(board, to_views(entries), image_size)
        print("drop %-15s per-view %.3f: total RMS %.3f -> %.3f (%d views left)"
              % (os.path.basename(path), dropped[-1][1], before, result["rms"], len(entries)))
    return entries, result, dropped


def patch_gate(report_path, gate, rms):
    """把报告里的 rms_gate/passed 改成命令行实际使用的门槛（save_report 用模块默认）。"""
    if float(gate) == float(common.RMS_GATE):
        return
    with open(report_path, encoding="utf-8") as f:
        report = json.load(f)
    report["rms_gate"] = float(gate)
    report["passed"] = rms < gate
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Solve camera intrinsics from captured ChArUco frames",
        epilog="exit: 0 = RMS < gate, 1 = RMS >= gate, 2 = setup/read error")
    parser.add_argument("--frames-dir", default=common.FRAMES_DIR,
                        help="帧目录，默认 webcam_pose/calib_frames")
    parser.add_argument("--pattern", default="*.png", help="帧文件名通配，默认 *.png")
    parser.add_argument("--meta", default=common.BOARD_JSON, help="标定板描述 json")
    parser.add_argument("--out", default=common.OUT_YAML, help="内参输出 yaml")
    parser.add_argument("--report", default=common.OUT_REPORT, help="报告输出 json")
    parser.add_argument("--min-corners", type=int, default=common.MIN_CORNERS,
                        help="单帧最少角点数，默认 %d" % common.MIN_CORNERS)
    parser.add_argument("--min-views", type=int, default=common.MIN_VIEWS,
                        help="最少可用帧数，默认 %d" % common.MIN_VIEWS)
    parser.add_argument("--gate", type=float, default=common.RMS_GATE,
                        help="RMS 合格阈值，默认 %.2f px" % common.RMS_GATE)
    parser.add_argument("--drop-worst", type=int, default=0,
                        help="剔除 per-view RMS 最大的 N 张后重标（逐张剔除，默认 0=不剔）")
    args = parser.parse_args(argv)

    # --- 1. 板 + 帧清单 ---
    try:
        board, meta = common.load_board(args.meta)
        paths = list_frames(args.frames_dir, args.pattern)
    except RuntimeError as exc:
        print("ERROR: %s" % exc)
        return 2

    if not paths:
        print("ERROR: no frames matching %r in %s" % (args.pattern, args.frames_dir))
        print("       先运行: python webcam_pose/calib_capture.py")
        return 2

    print("board: %s  %dx%d squares, square %.1f mm (actual %.1f mm)"
          % (meta["dictionary"], meta["squares_x"], meta["squares_y"],
             float(meta["square_mm"]),
             float(meta.get("actual_square_mm", meta["square_mm"]))))
    print("frames: %d found in %s" % (len(paths), args.frames_dir))
    if len(paths) < args.min_views:
        print("WARN: only %d frames, expected >= %d" % (len(paths), args.min_views))

    # --- 2. 逐帧检测 ---
    detector = cv2.aruco.CharucoDetector(board)
    try:
        entries, skipped, image_size = collect_entries(detector, paths, args.min_corners)
    except RuntimeError as exc:
        print("ERROR: %s" % exc)
        return 2

    if len(entries) < args.min_views:
        print("ERROR: usable views %d < %d (skipped %d: too few corners)"
              % (len(entries), args.min_views, len(skipped)))
        return 2
    if skipped:
        print("WARN: %d frame(s) skipped for too few corners" % len(skipped))

    # --- 3. 标定（可选剔除最差帧）---
    result = calibrate_from_views(board, to_views(entries), image_size)
    if args.drop_worst > 0:
        entries, result, _ = drop_worst(entries, board, image_size, args.drop_worst)
        if len(entries) < args.min_views:
            print("WARN: %d views left after dropping, below %d"
                  % (len(entries), args.min_views))

    print("")
    print_table(entries, result["per_view_rms"])
    print("calibration: %d views, RMS %.3f px (gate %.2f)"
          % (result["n_views"], result["rms"], args.gate))
    print("per-view RMS: min %.2f  max %.2f"
          % (min(result["per_view_rms"]), max(result["per_view_rms"])))
    print("K =\n%s" % np.array2string(result["K"], precision=2, suppress_small=True))
    print("dist = %s" % np.array2string(result["dist"], precision=4))

    # --- 4. 落盘 + 回读校验 ---
    try:
        common.save_calib(args.out, result, meta)
        save_report(args.report, result, meta)
        patch_gate(args.report, args.gate, result["rms"])
    except RuntimeError as exc:
        print("ERROR: %s" % exc)
        return 2
    print("wrote %s" % args.out)
    print("wrote %s" % args.report)

    K2, dist2, rms2, size2 = common.read_calib(args.out)
    ok = (K2 is not None and dist2 is not None
          and np.allclose(K2, result["K"], atol=1e-6))
    print("yaml readback: %s  %dx%d  rms %.3f"
          % ("OK" if ok else "MISMATCH", size2[0], size2[1], rms2))

    if result["rms"] < args.gate:
        print("PASS: RMS < %.2f px" % args.gate)
        return 0
    print("FAIL: RMS >= %.2f px; 可试 --drop-worst 1~3 剔除坏帧，"
          "或检查打印是否为 100%%（实测方格边长填入 actual_square_mm）" % args.gate)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
