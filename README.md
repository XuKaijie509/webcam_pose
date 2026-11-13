# webcam_pose

单目相机的**内参标定**与 **ChArUco 板 6DoF 位姿估计**，Python + OpenCV 实现。

用一块打印出来的 ChArUco 标定板完成两件事：

1. **内参标定**：采集 15 张以上不同姿态的帧，解出相机内参 `K` 与畸变系数，写入 `calib.yaml`；
2. **实时位姿估计**：摄像头实时检测标定板，输出板坐标系在相机坐标系下的 **4×4 齐次变换矩阵**，并在画面上叠加坐标轴、板面轮廓与数值面板。

*画面元素：黄色板面轮廓、板中心原点、X/Y/Z 三轴（红/绿/蓝）、左上角数值面板、左下角帧率。*

---

## 特性

- **采集与求解分离**：`calib_capture.py` 只负责存帧，`calib_solve.py` 只负责求解，采集不阻塞、可反复复标定
- **实时位姿**：基于 `solvePnP(IPPE)` + `solvePnPRefineLM`，单帧约 5 ms（≈199 fps 理论值）
- **明确的物体坐标系**：原点在板几何中心，x 沿宽边向右、y 沿高边向上、z 垂直板面指向相机
- **可视化**：实时三轴、板面轮廓、数值面板（在显示分辨率上绘制，文字不模糊）
- **中文/非 ASCII 路径可用**：绕开了 `cv2.FileStorage` 在 Windows 上的 ANSI 路径问题
- **可复现的质检**：输出总 RMS、每帧 RMS，支持 `--drop-worst` 剔除坏帧

## 环境要求

| 依赖 | 版本 | 说明 |
|---|---|---|
| Python | ≥ 3.8 | 实测 3.8.1 / 3.11.18 |
| OpenCV | **≥ 4.7** | 需要 `CharucoDetector`、`CharucoBoard.matchImagePoints`；实测 4.13.0 / 5.0.0 |
| NumPy | ≥ 1.21 | |
| Pillow | ≥ 9.0 | 可选，仅 `make_board.py` 生成 PDF 时需要 |

```bash
python -m pip install -r requirements.txt
```

## 快速开始

```bash
# 1) 生成 A4 标定板（PDF 用于打印，JSON 是程序读取的参数）
python webcam_pose/make_board.py

# 2) 用「实际大小 / 100%」打印 A4，量一下页面上的 100 mm 标尺
#    误差 > 1 mm 说明被缩放，重打或把实测值填进 board/charuco_a4.json 的 actual_square_mm

# 3) 采集：把板放在画面里，按 SPACE 采集，凑够 15 张后按 Q 退出
python webcam_pose/calib_capture.py

# 4) 求解内参 -> webcam_pose/calib.yaml + calib_report.json
python webcam_pose/calib_solve.py

# 5) 实时位姿（读取上一步的 calib.yaml）
python webcam_pose/pose_estimate.py
```

采集时的几点经验（直接决定标定质量）：

- 板占画面宽度的 **60–80%**，不要只占 1/4；
- 板要贴在**平整的硬板**上，纸面翘曲是最常见的 1 px 级误差来源；
- 倾角在 **±30°** 内分散，并覆盖画面**四角**；
- **停稳再按 SPACE**，避免运动模糊。

## 目录结构

```
.
├── webcam_pose/
│   ├── make_board.py       # 生成 A4 标定板（PDF / PNG / JSON）
│   ├── calib_capture.py    # 采集标定帧 -> calib_frames/
│   ├── calib_solve.py      # 由帧求解内参 -> calib.yaml + calib_report.json
│   ├── calib_common.py     # 共用：板加载、角点检测、内参读写、路径常量
│   ├── pose_estimate.py    # 实时位姿估计（叠加三轴 + 轮廓 + 数值）
│   ├── capture.py          # 最小摄像头闭环（取流 / 测 FPS / 按 S 存图）
│   └── board/
│       ├── charuco_a4.json # 板参数（字典、方格数、方格/码块边长）—— 唯一数据源
│       ├── charuco_a4.pdf  # A4 打印件（1:1）
│       └── charuco_a4.png  # 预览图
├── docs/                   # README 截图
├── requirements.txt
└── .gitignore
```

## 脚本说明

### `make_board.py`

生成 `board/charuco_a4.pdf`（打印用）、`.png`（预览）、`.json`（参数）。

默认板：`DICT_5X5_100` 字典、5×7 个 21 mm 方格、15.75 mm 码块 → 图案 **105 × 147 mm**，居中放在 A4 上，页面底部附 **100 mm 标尺**用于检查打印是否 1:1。

**改板型只需要改 4 个常量**（第 30–33 行）：

```python
SQUARES   = (5, 7)        # 行列数
SQUARE_MM = 21.0          # 方格边长
MARKER_MM = 15.75         # 码块边长（需 < SQUARE_MM）
DICT_NAME = "DICT_5X5_100"
```

改完重新运行脚本，JSON 会自动同步。

### `calib_capture.py`

| 参数 | 默认 | 说明 |
|---|---|---|
| `--index` | `0` | 摄像头索引 |
| `--frames-dir` | `webcam_pose/calib_frames` | 帧输出目录 |
| `--meta` | `board/charuco_a4.json` | 板描述 |
| `--min-corners` | `8` | 单帧最少角点数，不够则拒绝采集 |

- `SPACE` 采集（写 `view_001.png`…），`Q`/`ESC` 退出；
- 已有帧时**从最大序号续写，不覆盖旧帧**；
- 相机拿不到 1280×720 时会警告（`pose_estimate.py` 只接受与标定一致的分辨率）。

### `calib_solve.py`

| 参数 | 默认 | 说明 |
|---|---|---|
| `--frames-dir` | `calib_frames` | 帧目录 |
| `--pattern` | `*.png` | 文件名通配 |
| `--meta` | `board/charuco_a4.json` | 板描述 |
| `--out` | `calib.yaml` | 内参输出 |
| `--report` | `calib_report.json` | 报告输出 |
| `--min-corners` | `8` | 单帧最少角点数 |
| `--min-views` | `15` | 最少可用帧数 |
| `--gate` | `0.5` | RMS 合格阈值（px） |
| `--drop-worst` | `0` | 剔除每帧 RMS 最大的 N 张后重标 |

退出码：`0` 通过 / `1` RMS 超阈值 / `2` 帧目录缺失、帧数不足或读写失败。

> 720p USB 摄像头常见的 RMS 是 0.5–1.0 px，默认的 0.5 门槛偏严；若只是想判定"数据是否可用"，建议 `--gate 1.0`。

### `pose_estimate.py`

| 参数 | 默认 | 说明 |
|---|---|---|
| `--index` | `0` | 摄像头索引 |
| `--calib` | `calib.yaml` | 内参文件 |
| `--meta` | `board/charuco_a4.json` | 板描述 |

- 每 **2 s** 在终端打印一次 `T_cam<-obj`（4×4，平移单位 mm）+ 角点数 + 重投影 RMS + RPY + 板法向；
- 画面上叠加板面轮廓、物体系三轴（长 40 mm）、数值面板与帧率；
- `Q`/`ESC` 退出；
- 退出码：`0` 正常 / `2` 内参或相机不可用 / **`3` 相机分辨率与标定不一致**（内参不能跨分辨率复用）。

### `capture.py`

最小摄像头闭环：取流、显示实时 FPS、按 `S` 存图，用于快速确认相机是否可用。

```bash
python webcam_pose/capture.py --index 0 --out snap.png
```

## 坐标系与算法

### 物体坐标系

与板刚性固连，单位 **mm**：

| 元素 | 定义 |
|---|---|
| 原点 | 板图案的**几何中心** = (52.5, 73.5, 0) mm |
| x 轴 | 沿 105 mm 边，向右 |
| y 轴 | 沿 147 mm 边，向上 |
| z 轴 | 垂直板面，**指向相机**（观察者） |

换算关系（`R_fix = diag(1, -1, -1)`，`c = 板尺寸/2`）：

```
R_obj = R_board · R_fix^T
t_obj = R_board · c + t_board
```

> 注意：OpenCV 的 `CharucoBoard` 原生坐标系原点在板的**左上角**，+y 向下、+z 指向纸内。本项目的物体系是它的"中心化 + 翻转"版本，所以板正对相机时欧拉角 `roll ≈ 180°` 属正常现象。

### 求解流程

```
检测角点  CharucoDetector.detectBoard(gray)
   ↓
取对应点  board.matchImagePoints(corners, ids)      -> objPoints (mm), imgPoints (px)
   ↓
解位姿    cv2.solvePnP(..., SOLVEPNP_IPPE)          -> rvec, tvec
   ↓
精修      cv2.solvePnPRefineLM(...)                 -> 最小化重投影误差
   ↓
换坐标系  R_fix / c                                 -> T_cam<-obj
```

为什么用 `IPPE`：标定板是**平面目标**（所有 3D 点 z=0），IPPE 是专为共面点设计的解析解，不需要初值、在平面数据上比通用解法更准更快。若将来做非平面目标，必须换成 `EPNP` / `SQPNP` / `ITERATIVE`，因为 IPPE 在非平面数据上会直接失败。

## 实测数据

用 1280×720 摄像头、A4 打印板（105×147 mm）实采 15 帧：

| 指标 | 值 |
|---|---|
| 重投影 RMS | **0.868 px** |
| 每帧 RMS 范围 | 0.49 – 1.47 px |
| `fx / fy` | 799.54 / 800.20 |
| `cx / cy` | 620.99 / 413.15 |
| 畸变 `[k1 k2 p1 p2 k3]` | 0.0161, 0.0673, 0.0238, −0.0026, −0.3529 |
| 留一交叉验证（fx 标准差） | 1.4 px（≈0.17%） |
| 单帧位姿噪声（σ=0.87 px，250 mm） | **≈0.31 mm / 0.27°** |
| 单帧耗时 | ≈5 ms（角点检测占 4.8 ms） |

## 已知限制

- 只支持**单块 ChArUco 板**；同屏多板会按一块解，需扩展为按 ID 分组
- 逐帧独立求解，**无时序平滑**（静止时数值会有 ±0.3 mm 级抖动）
- **无外点剔除**（未使用 `solvePnPRansac`），对角点误检较敏感
- 平面目标专用（IPPE），不支持非平面 3D 特征点
- 采集端 Windows 优先：摄像头后端按 DSHOW → ANY → MSMF 顺序尝试

## 参考

- [OpenCV: ArUco marker detection](https://docs.opencv.org/4.x/d5/dae/tutorial_aruco_detection.html)
- [OpenCV: Camera calibration and 3D reconstruction](https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html)
- Collins & Bartoli, *Infinitesimal Plane-Based Pose Estimation*, IJCV 2014（IPPE 方法出处）
- 张正友, *A Flexible New Technique for Camera Calibration*, PAMI 2000（内参标定的基础）
