"""实时位姿估计：ChArUco 板坐标系在相机坐标系下的位姿。

功能：实时链路 + 每 2 s 打印齐次变换矩阵 + 画面叠加物体坐标系三轴、板面轮廓
与数值面板。平滑与质量门留到后续步骤。

物体坐标系定义（与板刚性固连，单位 mm）:
    原点  = 板几何中心 (squares_x*square/2, squares_y*square/2, 0)
    x 轴  = 板宽边方向（向右）
    y 轴  = 板高边方向（向上）
    z 轴  = 垂直板面、指向相机（观察者）

用法:
    python webcam_pose/pose_estimate.py
    python webcam_pose/pose_estimate.py --index 1 --calib webcam_pose/calib.yaml

按键:
    Q / ESC  退出

输出:
    终端每 2 s 打印一次 T_cam<-obj（4x4 齐次变换矩阵，平移单位 mm），
    附角点数、重投影 RMS、RPY(ZYX) 与板面法向。

退出码:
    0  正常退出
    2  calib.yaml 读不到 / 相机打不开
    3  相机分辨率与标定不一致（内参不可跨分辨率复用，必须报错而不是硬算）
"""
import argparse
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import calib_common as common
from capture import open_camera, set_resolution

WINDOW = "webcam_pose - pose"
PREVIEW_SIZE = (960, 540)
PRINT_INTERVAL = 2.0
MIN_PNP_CORNERS = 4      # 平面 PnP 的数学下限；真实帧通常 22-24
RMS_WARN = 3.0           # 仅提示，不做过滤（质量门属于后续步骤）
AXIS_LEN_MM = 40         # 三轴长度：约 2 个方格，300 mm 处约 107 px
AXIS_THICKNESS = 2
OUTLINE_COLOR = (0, 255, 255)   # 板面轮廓（BGR 黄）
OUTLINE_THICKNESS = 4
LOST_GRACE = 0.5         # 板短暂检出失败时，打印仍可沿用最近一次成功结果
OUT_CALIB = os.path.join(HERE, "calib.yaml")

# 屏幕文字：在显示分辨率上绘制（1280x720 画完再缩小会让字发虚），
# 用 DUPLEX（比 SIMPLEX 笔画重，缩小后更抗糊）+ 半透明底板提高对比度。
FONT = cv2.FONT_HERSHEY_DUPLEX
HUD_SCALE = 0.75
HUD_THICKNESS = 1
HUD_MARGIN = 12
HUD_PAD_X = 14
HUD_PAD_Y = 10
HUD_LINE_GAP = 9
HUD_SEG_GAP = 16
PANEL_ALPHA = 0.62
PANEL_BG = (22, 22, 22)
LABEL_COLOR = (150, 190, 210)   # 淡蓝灰，弱化标签
VALUE_COLOR = (245, 245, 245)   # 白，突出数值
WARN_COLOR = (90, 220, 255)     # 橙黄，reproj 超阈值
MISS_COLOR = (90, 90, 255)      # 红，未检出


def read_calib(path):
    """读内参；中文路径的 FileStorage 绕行由 calib_common 统一处理。"""
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise RuntimeError("calib not found: %s" % path)
    return common.read_calib(path)


def make_object_frame(board):
    """返回物体系相对板系的固定变换 (R_fix, c)：p_obj = R_fix @ (p_board - c)。"""
    rb = board.getRightBottomCorner()
    c = np.array([rb[0] / 2.0, rb[1] / 2.0, 0.0], dtype=np.float64)
    return np.diag([1.0, -1.0, -1.0]), c


def solve_pose(gray, detector, board, K, dist, R_fix, c):
    """纯函数：一帧灰度图 -> 位姿结果 dict，无板返回 None。

    dict: T(4x4), R(3x3), t(3,), rvec, tvec, rvec_obj, outline(4x3),
          reproj(px), n(角点数)
    """
    corners, ids, _, _ = detector.detectBoard(gray)
    if ids is None or len(ids) < MIN_PNP_CORNERS:
        return None
    objp, imgp = board.matchImagePoints(corners, ids)
    if objp is None or imgp is None or len(objp) < MIN_PNP_CORNERS:
        return None

    try:
        ok, rvec, tvec = cv2.solvePnP(objp, imgp, K, dist, flags=cv2.SOLVEPNP_IPPE)
        if not ok:
            return None
        rvec, tvec = cv2.solvePnPRefineLM(objp, imgp, K, dist, rvec, tvec)
    except cv2.error:
        return None

    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, dist)
    err = np.linalg.norm(proj.reshape(-1, 2) - imgp.reshape(-1, 2), axis=1)

    R_board, _ = cv2.Rodrigues(rvec)
    R_obj = R_board @ R_fix
    t_obj = (R_board @ c.reshape(3, 1) + tvec).reshape(3)

    T = np.eye(4)
    T[:3, :3] = R_obj
    T[:3, 3] = t_obj
    half_w, half_h = float(c[0]), float(c[1])
    outline = np.array([[-half_w, half_h, 0.0],
                        [half_w, half_h, 0.0],
                        [half_w, -half_h, 0.0],
                        [-half_w, -half_h, 0.0]], dtype=np.float64)
    return {
        "T": T,
        "R": R_obj,
        "t": t_obj,
        "rvec": rvec,
        "tvec": tvec,
        "rvec_obj": cv2.Rodrigues(R_obj)[0],
        "outline": outline,
        "reproj": float(np.sqrt(np.mean(err ** 2))),
        "n": int(len(ids)),
    }


def euler_zyx(R):
    """ZYX 欧拉角（yaw-pitch-roll），单位度。"""
    yaw = np.arctan2(R[1, 0], R[0, 0])
    pitch = np.arcsin(np.clip(-R[2, 0], -1.0, 1.0))
    roll = np.arctan2(R[2, 1], R[2, 2])
    return np.degrees([yaw, pitch, roll])


def print_pose(result, elapsed, age=0.0):
    yaw, pitch, roll = euler_zyx(result["R"])
    normal = result["R"][:, 2]
    head = "[t=%6.1fs] corners=%2d  reproj=%.2f px" % (elapsed, result["n"], result["reproj"])
    if age > 0.02:
        head += "  (stale: last ok %.0f ms ago)" % (age * 1e3)
    if result["reproj"] > RMS_WARN:
        head += "  (WARN: reproj > %.1f px)" % RMS_WARN
    print(head)
    print("T_cam<-obj (mm):")
    print(np.array2string(result["T"], precision=3, suppress_small=False))
    print("  RPY(ZYX) = yaw %7.1f  pitch %6.1f  roll %7.1f deg" % (yaw, pitch, roll))
    print("  z axis(board normal) in cam = (%+.3f, %+.3f, %+.3f)"
          % (normal[0], normal[1], normal[2]))
    sys.stdout.flush()


def scale_intrinsics(K, sx, sy):
    """把内参缩放到另一分辨率（仅用于在缩小后的显示图上画叠加）。"""
    S = np.diag([sx, sy, 1.0])
    return S @ K


def _panel(img, x0, y0, x1, y1):
    """半透明深色底板，让文字在任何背景上都清晰。"""
    h, w = img.shape[:2]
    x0, y0 = max(0, int(x0)), max(0, int(y0))
    x1, y1 = min(w, int(x1)), min(h, int(y1))
    if x1 <= x0 or y1 <= y0:
        return
    roi = img[y0:y1, x0:x1]
    patch = np.empty_like(roi)
    patch[:] = PANEL_BG
    cv2.addWeighted(patch, PANEL_ALPHA, roi, 1.0 - PANEL_ALPHA, 0, roi)
    cv2.rectangle(img, (x0, y0), (x1 - 1, y1 - 1), (70, 70, 70), 1, cv2.LINE_AA)


def _table_metrics(rows):
    """返回 (标签列宽, 数值列宽, 行高)，用于把数值对齐成一列。"""
    label_col = max(cv2.getTextSize(r[0], FONT, HUD_SCALE, HUD_THICKNESS)[0][0] for r in rows)
    value_col = max(cv2.getTextSize(r[1], FONT, HUD_SCALE, HUD_THICKNESS)[0][0] for r in rows)
    (_, line_h), _ = cv2.getTextSize("Ag", FONT, HUD_SCALE, HUD_THICKNESS)
    return label_col, value_col, line_h


def _draw_table(img, rows, x, y_top, label_col):
    """两列表格：标签固定列宽，数值从同一 x 起画，天然对齐。"""
    (_, line_h), _ = cv2.getTextSize("Ag", FONT, HUD_SCALE, HUD_THICKNESS)
    step = line_h + HUD_LINE_GAP
    value_x = x + label_col + HUD_SEG_GAP
    y = y_top
    for label, value, color in rows:
        cv2.putText(img, label, (x, y), FONT, HUD_SCALE, LABEL_COLOR,
                    HUD_THICKNESS, cv2.LINE_AA)
        cv2.putText(img, value, (value_x, y), FONT, HUD_SCALE, color,
                    HUD_THICKNESS, cv2.LINE_AA)
        y += step


def draw_geometry(img, result, K, dist):
    """叠加板面轮廓与物体坐标系三轴（K/dist 需与当前图像分辨率匹配）。"""
    if result is None:
        return
    rvec = result["rvec_obj"]
    tvec = result["t"].reshape(3, 1)

    # 1) 板面轮廓：物体系 4 个角投影后连成闭合多边形
    outline = result["outline"]
    cam_z = (result["R"] @ outline.T + result["t"].reshape(3, 1))[2]
    if np.all(cam_z > 1.0):
        pts = cv2.projectPoints(outline, rvec, tvec, K, dist)[0].reshape(-1, 2)
        ip = np.round(pts).astype(np.int32)
        cv2.polylines(img, [ip], True, OUTLINE_COLOR, OUTLINE_THICKNESS, cv2.LINE_AA)
        for p in ip:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, OUTLINE_COLOR, -1, cv2.LINE_AA)

    # 2) 物体系三轴 + 原点
    cv2.drawFrameAxes(img, K, dist, rvec, tvec, AXIS_LEN_MM, AXIS_THICKNESS)
    px = cv2.projectPoints(np.zeros((1, 3)), rvec, tvec, K, dist)[0].reshape(-1, 2)[0]
    cv2.circle(img, (int(round(px[0])), int(round(px[1]))), 4, (0, 255, 255), -1)


def draw_hud(img, result, fps):
    """左上角数值面板 + 左下角帧率；result 为 None 时只提示，不画陈旧位姿。"""
    if result is None:
        rows = [("status", "board not found", MISS_COLOR)]
    else:
        yaw, pitch, roll = euler_zyx(result["R"])
        reproj_color = WARN_COLOR if result["reproj"] > RMS_WARN else VALUE_COLOR
        rows = [
            ("n", "%2d" % result["n"], VALUE_COLOR),
            ("reproj", "%.2f px" % result["reproj"], reproj_color),
            ("t", "(%7.1f, %7.1f, %7.1f) mm" % tuple(result["t"]), VALUE_COLOR),
            ("RPY", "(%7.1f, %7.1f, %7.1f) deg" % (yaw, pitch, roll), VALUE_COLOR),
        ]

    label_col, value_col, line_h = _table_metrics(rows)
    n_rows = len(rows)
    panel_w = label_col + HUD_SEG_GAP + value_col + 2 * HUD_PAD_X
    panel_h = n_rows * line_h + (n_rows - 1) * HUD_LINE_GAP + 2 * HUD_PAD_Y
    _panel(img, HUD_MARGIN, HUD_MARGIN, HUD_MARGIN + panel_w, HUD_MARGIN + panel_h)
    _draw_table(img, rows, HUD_MARGIN + HUD_PAD_X, HUD_MARGIN + HUD_PAD_Y + line_h, label_col)

    # 左下角帧率
    fps_text = "%.1f fps" % fps
    (fw, fh), _ = cv2.getTextSize(fps_text, FONT, HUD_SCALE, HUD_THICKNESS)
    fy1 = img.shape[0] - HUD_MARGIN
    _panel(img, HUD_MARGIN, fy1 - fh - 2 * HUD_PAD_Y, HUD_MARGIN + fw + 2 * HUD_PAD_X, fy1)
    cv2.putText(img, fps_text, (HUD_MARGIN + HUD_PAD_X, fy1 - HUD_PAD_Y), FONT,
                HUD_SCALE, VALUE_COLOR, HUD_THICKNESS, cv2.LINE_AA)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Realtime ChArUco board pose")
    parser.add_argument("--index", type=int, default=0, help="摄像头索引，默认 0")
    parser.add_argument("--calib", default=OUT_CALIB, help="内参 yaml，默认 calib.yaml")
    parser.add_argument("--meta", default=common.BOARD_JSON, help="标定板描述 json")
    args = parser.parse_args(argv)

    # --- 1. 内参 + 板 ---
    try:
        K, dist, rms_calib, (cal_w, cal_h) = read_calib(args.calib)
        board, meta = common.load_board(args.meta)
    except (RuntimeError, OSError) as exc:
        print("ERROR: %s" % exc)
        return 2
    if K is None or dist is None:
        print("ERROR: calib.yaml 里没有 camera_matrix / distortion_coefficients")
        return 2

    R_fix, c = make_object_frame(board)
    detector = cv2.aruco.CharucoDetector(board)

    print("calib: %s  rms %.3f px  %dx%d"
          % (os.path.basename(args.calib), rms_calib, cal_w, cal_h))
    print("board: %s  %dx%d squares, square %.1f mm"
          % (meta["dictionary"], meta["squares_x"], meta["squares_y"],
             float(meta.get("actual_square_mm", meta["square_mm"]))))
    print("object frame: origin=board center (%.1f, %.1f) mm, x right, y up, z toward camera"
          % (c[0], c[1]))
    print("fx=%.2f fy=%.2f cx=%.2f cy=%.2f" % (K[0, 0], K[1, 1], K[0, 2], K[1, 2]))

    # --- 2. 相机必须与标定同分辨率 ---
    cap = open_camera(args.index)
    if cap is None:
        print("ERROR: cannot open camera index %d" % args.index)
        return 2
    width, height = set_resolution(cap, cal_w, cal_h)
    if (width, height) != (cal_w, cal_h):
        print("ERROR: 分辨率不匹配: 相机给出 %dx%d, 标定是 %dx%d"
              % (width, height, cal_w, cal_h))
        print("       内参不能跨分辨率复用（640x480 往往是另一种传感器模式）。")
        print("       请把摄像头设到 %dx%d 后重新标定, 或换一个支持该分辨率的相机。"
              % (cal_w, cal_h))
        cap.release()
        return 3
    print("camera: index=%d  %dx%d  [Q/ESC] quit  print every %.0f s"
          % (args.index, width, height, PRINT_INTERVAL))

    # 叠加图层按显示分辨率绘制，需把内参同步缩放
    K_disp = scale_intrinsics(K, PREVIEW_SIZE[0] / float(width),
                              PREVIEW_SIZE[1] / float(height))

    # --- 3. 取流循环 ---
    t_start = time.monotonic()
    next_print = t_start
    n_frames = 0
    n_miss = 0
    last_result = None
    last_ok = t_start
    last_frame_t = t_start
    fps = 0.0
    cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(WINDOW, PREVIEW_SIZE[0], PREVIEW_SIZE[1])
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print("ERROR: frame grab failed after %d frames" % n_frames)
                break
            n_frames += 1

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            result = solve_pose(gray, detector, board, K, dist, R_fix, c)
            now = time.monotonic()
            if result is None:
                n_miss += 1
            else:
                last_result = result
                last_ok = now

            dt = now - last_frame_t
            last_frame_t = now
            if dt > 0:
                inst = 1.0 / dt
                fps = inst if fps == 0.0 else 0.9 * fps + 0.1 * inst

            if now >= next_print:
                elapsed = now - t_start
                if result is not None:
                    print_pose(result, elapsed)
                elif last_result is not None and (now - last_ok) <= LOST_GRACE:
                    print_pose(last_result, elapsed, now - last_ok)
                else:
                    print("[t=%6.1fs] no board: 需要 >= %d 个 ChArUco 角点"
                          % (elapsed, MIN_PNP_CORNERS))
                    sys.stdout.flush()
                next_print = now + PRINT_INTERVAL

            # 显示链：先把帧缩放到 PREVIEW_SIZE，再在显示分辨率上画叠加，
            # 这样文字和线条都是 1:1 像素，不会被二次缩小糊掉。
            if (frame.shape[1], frame.shape[0]) != PREVIEW_SIZE:
                disp = cv2.resize(frame, PREVIEW_SIZE, interpolation=cv2.INTER_AREA)
            else:
                disp = frame
            draw_geometry(disp, result, K_disp, dist)
            draw_hud(disp, result, fps)
            cv2.imshow(WINDOW, disp)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    print("frames: %d, 未解出位姿的帧: %d" % (n_frames, n_miss))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
