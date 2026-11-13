"""标定链路共用部分：路径常量、板加载、角点检测、内参文件读写。

被三个脚本共用，避免重复实现，也避免"采集脚本反向依赖求解脚本"：
    calib_capture.py   采集标定图像      -> load_board / detect_board
    calib_solve.py     从图像求解内参    -> load_board / detect_board / save_calib / read_calib
    pose_estimate.py   实时位姿估计      -> load_board / read_calib
"""
import json
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
BOARD_JSON = os.path.join(HERE, "board", "charuco_a4.json")
OUT_YAML = os.path.join(HERE, "calib.yaml")
OUT_REPORT = os.path.join(HERE, "calib_report.json")
FRAMES_DIR = os.path.join(HERE, "calib_frames")

MIN_VIEWS = 15          # 触发标定的最少姿态数
MIN_CORNERS = 8         # 单帧至少检出多少个 ChArUco 角点才算有效
RMS_GATE = 0.5          # 重投影 RMS 合格阈值（px）


def load_board(meta_path=BOARD_JSON):
    """读板参数并构造 CharucoBoard。

    actual_square_mm 是打印后实测值，码块长度按同比例缩放，
    这样即使打印机有 1-2% 缩放，板上的"方格/码块"比例仍与设计一致。
    """
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    nominal = float(meta["square_mm"])
    actual = float(meta.get("actual_square_mm", nominal))
    scale = actual / nominal
    dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, meta["dictionary"]))
    board = cv2.aruco.CharucoBoard(
        (int(meta["squares_x"]), int(meta["squares_y"])),
        actual, float(meta["marker_mm"]) * scale, dictionary)
    return board, meta


def detect_board(detector, frame):
    """返回 (corners(N,2) float32, ids(N,) int32, marker_corners, marker_ids)。

    corners/ids 统一成扁平形状，未检出时为 None。
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)
    if corners is not None:
        corners = corners.reshape(-1, 2).astype(np.float32)
    if ids is not None:
        ids = ids.reshape(-1).astype(np.int32)
    return corners, ids, marker_corners, marker_ids


def _ascii_path(path):
    return all(ord(ch) < 128 for ch in os.path.abspath(path))


def _call_with_ascii_name(func, path, *args):
    """在纯 ASCII 路径下直接调用，否则临时切到目标目录只传文件名。

    cv2.FileStorage 在 Windows 上用 ANSI fopen，路径含中文时直接打不开
    （本仓库位于 E:\\A_资料库\\...，属于必踩）；Python 自带的 open/json 没有
    这个限制，所以只有 FileStorage 相关的读写需要绕。
    """
    path = os.path.abspath(path)
    if _ascii_path(path):
        return func(path, *args)
    folder, name = os.path.split(path)
    cwd = os.getcwd()
    os.chdir(folder)
    try:
        return func(name, *args)
    finally:
        os.chdir(cwd)


def _save_calib_file(yaml_path, result, meta):
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_WRITE)
    if not fs.isOpened():
        raise RuntimeError("cannot write %s" % yaml_path)
    fs.write("format", "webcam_pose/calib-v1")
    fs.write("image_width", int(result["image_size"][0]))
    fs.write("image_height", int(result["image_size"][1]))
    fs.write("camera_matrix", result["K"])
    fs.write("distortion_coefficients", result["dist"].reshape(1, -1))
    fs.write("rms", result["rms"])
    fs.write("n_views", result["n_views"])
    fs.write("square_mm", float(meta.get("actual_square_mm", meta["square_mm"])))
    fs.write("squares_x", int(meta["squares_x"]))
    fs.write("squares_y", int(meta["squares_y"]))
    fs.write("dictionary", meta["dictionary"])
    fs.write("board_meta", "board/charuco_a4.json")
    fs.release()


def _read_calib_file(yaml_path):
    fs = cv2.FileStorage(yaml_path, cv2.FILE_STORAGE_READ)
    if not fs.isOpened():
        raise RuntimeError("cannot read %s" % yaml_path)
    K = fs.getNode("camera_matrix").mat()
    dist = fs.getNode("distortion_coefficients").mat()
    rms = fs.getNode("rms").real()
    width = int(fs.getNode("image_width").real())
    height = int(fs.getNode("image_height").real())
    fs.release()
    return K, dist, rms, (width, height)


def save_calib(yaml_path, result, meta):
    """写内参 yaml（中文路径安全）。"""
    return _call_with_ascii_name(_save_calib_file, yaml_path, result, meta)


def read_calib(yaml_path):
    """读内参 yaml，返回 (K, dist, rms, (width, height))（中文路径安全）。"""
    return _call_with_ascii_name(_read_calib_file, yaml_path)
