# 18 楼墙侧四相机标定

墙侧四台相机使用独立坐标系，正式标定为 `src/config/four_camera_calib_18_wall.json`。

| 用途 | 相机序列号 |
| --- | --- |
| 参考相机 | DB0260380 |
| 其余相机 | DB0260342、DB0260368、DB0260436 |

四台已实测可同步采集 2048×1536 全幅图像。采集配置 `src/config/camera_18_wall.json` 使用独立的 Action device/group key 2，曝光 1800 µs、增益 12.78 dB、digital shift 23.5。曝光应以现场棋盘黑白格清晰、白格不过曝为准。

## 正式标定与坐标定义

- 内参：`src/config/four_camera_intrinsics_18_wall.json`。
- 相对外参候选：`src/config/four_camera_calib_18_wall_candidate.json`。
- 地面注册候选：`src/config/four_camera_calib_18_wall_ground_candidate.json`。
- 已确认的正式结果：`src/config/four_camera_calib_18_wall.json`，内容与地面注册候选一致。

世界坐标使用米描述，标定 JSON 的平移和相机位置使用 **mm**。发球线靠近相机的下沿与发球区中线中心的交点为 **(0, 5.48, 0) m**；+Y 朝球网，面向球网的右手侧为 +X，+Z 向上。墙侧与窗侧分别使用各自坐标系。

本次数据为 `wall_calib_20261125_004`（476 组）和 `wall_calib_20261125_005`（516 组），合并目录为 `data/four_camera_calibration/wall_calib_20261125_004_005_merged`。

| 相机 | 内参拟合 RMS / px | 未参与内参拟合的照片验证 RMS / px |
| --- | ---: | ---: |
| DB0260380 | 0.2624 | 0.2690 |
| DB0260342 | 0.2256 | 0.2329 |
| DB0260368 | 0.2573 | 0.2544 |
| DB0260436 | 0.2824 | 0.2790 |

相对外参使用 169 组共同观测、37752 个角点，整体重投影 RMS 为 **0.3716 px**。六组相机对均通过极线检查，整体极线 RMS 为 **0.4588 px**。

地面注册使用 `004/0001` 的 14 颗至少三视图球，圆形外轮廓拟合、三角化后求球心平面，扣除 33 mm 球心高度得到地面。平面残差 RMS **4.47 mm**；另 7 颗未参与平面拟合的两视图球，高度验证 RMS **2.73 mm**。固定该地面，以发球线下沿和中线中心拟合正交 XY 方向及交点位置，保留原内参、尺度和相机间相对位姿。

标线重投影总体 RMS **1.26 px**，四路交点估计的空间散布 RMS **10.21 mm**。这些是内部一致性指标，不代表实测绝对坐标具有毫米精度。

## 1. 采集移动棋盘

在项目根目录的 PowerShell 执行，提前关闭 tracker 和 MVS 相机预览：

```powershell
$env:MVS_MVIMPORT_DIR = 'C:\Program Files (x86)\MVS\Development\Samples\Python\MvImport'
$env:PYTHONUTF8 = '1'
.\.venv_ros2\Scripts\python.exe -m calibration.four_camera_intrinsics_capture --camera-config src/config/camera_18_wall.json --count 600 --duration 500 --session wall_calib_20261125 --web-control
```

手机打开终端打印的预览网址，确认四路画面后点击 **Start capture**；此时才开始计时和保存。600 组约 8 分 20 秒，也可用网页 **Stop capture** 提前结束。

沿用原来的 **9×12 格、格边长 45 mm** 棋盘，即 **8×11 内角点**。先确认实物规格一致。移动时覆盖每台相机画面的中心、四周、远近位置，改变棋盘俯仰和左右倾角，每个位置短暂停稳。每台都需要数十个清晰且姿态不同的完整棋盘；相邻相机还要有足够多的同帧完整棋盘，用于连接四台相对外参。

数据目录为 `data/four_camera_calibration/wall_calib_20261125`，每个相机一个子目录。重名时自动追加 `_001` 等后缀，后续命令必须使用终端实际打印的目录。需要调整曝光时，在采集命令末尾加 `--exposure 2000` 等值（单位 µs）。

## 2. 复算本次内参和相对外参

本次逆光图片先在原分辨率灰度图上做 CLAHE（clipLimit=3，8×8 分块），再以 INTER_AREA 缩小至一半，使用 SB 的 NORMALIZE、EXHAUSTIVE、ACCURACY 检测角点，按像素中心映射回原分辨率，并用原图棋盘奇偶方向统一角点顺序。原始图片未修改，首次检测缓存在各源目录的 `corner_detections_initial.json`。

以下命令读取本次处理后的角点缓存。评分阈值为 0.85，每台抽取 120 帧拟合内参；外参固定内参计算。重新采集的数据须重新检测并检查角点质量，不能只依据评分判断标定精度。

```powershell
.\.venv_ros2\Scripts\python.exe -m calibration.four_camera_intrinsics_calibrate --images data/four_camera_calibration/wall_calib_20261125_004_005_merged --serials DB0260380 DB0260342 DB0260368 DB0260436 --range-end 992 --score-threshold 0.85 --max-frames 120 --output src/config/four_camera_intrinsics_18_wall.json
.\.venv_ros2\Scripts\python.exe -m calibration.run_calibration --images data/four_camera_calibration/wall_calib_20261125_004_005_merged --serials DB0260380 DB0260342 DB0260368 DB0260436 --reference DB0260380 --intrinsics src/config/four_camera_intrinsics_18_wall.json --range-end 992 --score-threshold 0.85 --output src/config/four_camera_calib_18_wall_candidate.json
```

第二条仅复算相对外参，不会生成新的地面注册结果。重新标定后应再次检查地面和标线，再更新墙侧正式文件。

## 3. 本地验证资料

原始照片、角点缓存、分析脚本和预览位于被 Git 忽略的采集数据目录：

- 合并目录下 `calibration_report.md`、`intrinsics_validation.json`：内参及相对外参验证。
- `wall_calib_20261125_004/ground_registration_0001/ground_registration_report.md`：地面注册报告。
- 同目录 `ground_reprojection_overview.jpg`：地面网格小点和 XY 方向。
- 同目录 `centerline_23p77_minus_17_to_23/overview.jpg`：X=0、Z=0，Y=6.77、5.77、4.77、3.77、2.77、1.77、0.77 m 的小点投影。
