# MEMORY

项目长期记忆文件。用于记录当前默认配置、近期决策和后续协作时容易忘的约定。

## 当前 Tracker 默认事实

- Tracker 默认使用四相机 rig：
  - `src/config/camera.json`
  - `src/config/four_camera_calib.json`
- 当前默认相机序列号：
  - 主相机 `DA7403103`
  - 从相机 `DA8571029`
  - 从相机 `DA7403087`
  - 从相机 `DA8474746`
- 当前默认采集参数：
  - 全画幅 `2048x1536`
  - `35fps`
  - `3000us` 曝光
  - `23.5dB` 增益
- Tracker 的 YOLO 分片默认是：
  - `1280x1280` 切片
  - 压缩到 `640x640` 推理
- 当前默认检测 engine：
  - `yolo_model/tennis_yolo26_v2_20260203_b4_640.engine`
- 3D 球定位默认规则：
  - 至少 `2` 台相机参与三角化
  - 默认最大重投影误差 `15px`
- ROS2 输出：
  - `/pc_car_loc`
  - `/predict_hit_pos`

## 启动约定

- 启动 tracker 优先使用根目录脚本：
  - `.\run_tracker.ps1`
- 探测当前会选哪个环境：
  - `.\run_tracker.ps1 -ProbeOnly`
- `.venv_ros2` 是优先环境；有 CUDA / TensorRT 时应优先使用它。

## 近期变更

- `2026-03-23`
  - 删除旧的 `CLAUDE.md`，项目上下文以 `DEV.md` 和本文件为准。
  - `BallLocalizer` / `CarLocalizer` 默认标定切到 `four_camera_calib.json`。
  - Tracker 默认分片从 `1000x1000` 调回 `1280x1280`。
  - `DEV.md` 的 step 16 更新为当前 tracker 能力摘要。
  - 用户已人工确认：视频记录、球识别、轨迹追踪、车定位均已具备；多个网球位置 spot check 后，3D 误差整体为 cm 级。
  - 固定相机 rig 在现场放置约 1 周、包括受撞击扰动后，标定结果仍保持稳定。
  - 性能 debug 结论：
    - 四相机同步采集本身正常，`src.benchmark --duration 10` 实测 `35.1 fps`
    - 优化前，tracker `--no-video` 实测约 `22.3 fps`
    - 优化前，tracker 开启原始拼接视频保存时实测约 `13.8 fps`
    - `2026-03-24` 已新增并接入 `yolo_model/tennis_yolo26_v2_20260203_b4_640.engine`
    - `BallDetector` 已支持固定 batch engine 自动补齐/分批，默认接口也可直接使用 `b4` engine
    - Bayer 解码快路径已改成“先旋转 raw Bayer，再 demosaic”，像素结果与旧路径一致
    - 该解码快路径在 4 相机并行 benchmark 中约从 `11.9ms` 降到 `8.6ms`
    - 优化后，tracker `--no-video` 短跑约 `24.9 fps`
    - 优化后，tracker 开启原始拼接视频保存 `10s` 短跑约 `23.1 fps`
    - 当前剩余主要开销约为：`decode ~11.6ms`，`yolo ~26.7ms`，后台写视频 `~35ms`

- `2026-07-03`: HTML 报表新增 Arm tab（仿 tennis-man arm_controller 的 session_viewer）。
  - `test_src/extract_arm_bag.py` 在 ROS2 环境（经 `ros2/run_ros2.bat`）读 `{run_id}_rosbag`，输出 `{run_id}_arm.json`：`/joint_states` 实际 + `/tennis/motor_command` 目标（首个轨迹点）+ status/arm_command/hit_pos/predict_hit_pos 事件；TCP 由 tennis-man 的 `arm_controller.compact_arm_kinematics.fk` 正解（`sys.path` 引用 `D:\tennis-man\arm_controller\src`，不在本项目重复实现）。
  - `generate_curve3_html.py` 新增 `--arm-json`（缺省自动探测 `<input>_arm.json`），Arm tab 为单 plot 四层 subplot（Position/Velocity/Effort/TCP，target 实线 vs actual 虚线，事件竖线）。
  - `run_tracker.py` post_run 链：bag 存在时先 Extract arm rosbag 再生成 HTML。
  - 性能教训（本机 Chrome 的 WebGL 是软渲染）：4 个独立 scattergl plot（4 个 WebGL context）或单 context 全量 15 万点都会把渲染器卡死几十秒；最终方案 = SVG `scatter`（lines 模式一条 trace 一条 path）+ 窗口化抽稀（hit/predict 事件 ±[-2,+4]s 内全分辨率，窗口外 2Hz），实测秒开。全量数据仍在 `_arm.json`，抽稀只发生在绘图端。
  - Arm tab 时间轴是 bag 相对时间（bag 首条消息 = 0s），与 tracker 主时间轴（首帧 exposure_pc）无对齐锚点，暂不混画。

## 协作提醒

- 如果更换相机 rig，必须同时检查：
  - `src/config/camera.json`
  - `src/config/four_camera_calib.json`
  - `src/config/tracker.json`
- `multi_calib.json` / 三目配置只保留作历史结果，不应再作为 tracker 默认入口。

## Camera API Notes

- `2026-03-24`: On live camera `DA7403103`, both `ReverseX` and `ReverseY` exist as bool nodes and can be read through the MVS universal node API.
- `ReverseX=True` + `ReverseY=True` can be used as device-side `180deg` rotation, which is more relevant to tracker performance than SDK-side image post-processing.
- `ReverseY` returned `0x80000106` (`MV_E_GC_ACCESS`) when written during grabbing, but became writable after `MV_CC_StopGrabbing()`. In practice, these nodes should be configured before `MV_CC_StartGrabbing()`.
- The SDK also exposes `MV_CC_RotateImage(...)`, but that is SDK-side rotation on acquired image data, not camera-side orientation.
- Independent probe on `2026-03-24` captured one frame without reverse and one frame with pre-grab `ReverseX=True, ReverseY=True`; the hardware-rotated frame matched the software `180deg` baseline strongly (`corr_rot180=0.958849` vs `corr_direct=-0.158286`, `mae_rot180=9.281` vs `mae_direct=84.769`).
- `src/ball_grabber.py` now supports temporary environment switches for A/B tests without changing defaults:
  - `BALL_TRACER_CAMERA_REVERSE_180=1` (or `BALL_TRACER_CAMERA_REVERSE_X/Y`)
  - `BALL_TRACER_SOFTWARE_ROTATE_180=0`
- With `BALL_TRACER_CAMERA_REVERSE_180=1` and `BALL_TRACER_SOFTWARE_ROTATE_180=0`, a real `run_tracker.ps1 -Duration 15 -NoVideo` run on `2026-03-24` reached `33.8 fps` (`519` frames / `15.4s`), close to the configured `35 fps`.
- `2026-03-24`: tracker mainline units are now meters end-to-end for ball 3D, car 3D, Curve3 state, JSON outputs, HTML, and ROS2 publish payloads.
- `src/run_tracker.py` now writes `config.distance_unit = "m"` into tracker JSON. Downstream tools should treat older JSON without that field as legacy mm data.
- `src/car_localizer.py` now applies `vehicle_reference.apriltag_center_to_car_base_offset_m = (0.06, 0.10, -0.34)` before returning `CarLoc`, so `/pc_car_loc` publishes the car base, not the AprilTag center.
- `2026-03-24`: tracker terminal/log output is forced to UTF-8 in both `run_tracker.ps1` and `src/run_tracker.py`, so redirected logs should no longer mix PowerShell UTF-8 with Python CP936 output.
- `2026-03-25`: tracker now supports `ball_detection_disabled_serials` in `src/config/tracker.json`. The current default disables camera `029` for ball YOLO and ball 3D only; capture, stitched video, saved video, JSON frame logs, and AprilTag car localization still keep all four cameras.
- `2026-03-27`: after re-checking the live image with hardware 180-degree camera reverse enabled, the current AprilTag appears in the lower part of the full image (`cy` about `1139-1413` on height `1536` for the detecting cameras). `src/car_localizer.py` therefore uses the lower 60% ROI, crop-only with native pixels and no resize.
- `2026-03-27`: HTML time axes and offline annotated-video overlay time are aligned to the same reference: the first frame's `exposure_pc` (`t=0`). `src/run_tracker.py` now records `config.first_frame_exposure_pc` and per-frame `elapsed_s`; `test_src/generate_curve3_html.py` and `test_src/annotate_video.py` both fall back to `frames[0].exposure_pc` for older JSON.
- `2026-03-27`: raw stitched tracker video now also prints the same time base as HTML: `#frame  t=...s  HH:MM:SS.mmm`, where `t` is relative to the first frame's `exposure_pc`. This is intended to make dropped-frame cases debuggable by eye.
- `2026-03-27`: added `run_tracker_terminal.ps1` for foreground tracker launches from terminal. It defaults to `ROS_DOMAIN_ID=2`, hardware 180-degree reverse on, software rotate off, and relies on `run_tracker.py`'s existing `KeyboardInterrupt` cleanup so `Ctrl+C` still flushes video/JSON cleanly.
- `2026-03-27`: added `annotate_latest_tracker.ps1` plus dual-JSON support. `test_src/annotate_video.py` now supports `--racket-json-output` to write a separate racket-only JSON (`frames[*].video_frame_idx` / `racket3d` / `racket_observations`), and `test_src/generate_curve3_html.py` now supports `--racket-json` so HTML can be generated from the base tracker JSON plus the separate racket JSON.
- `2026-03-27`: offline racket annotation was switched from the temporary tracker-style bbox/tile center logic to the ArmCalibration production logic in `src/racket_localizer.py`: `racket.onnx + racket_pose.onnx`, only keypoints `0-3` define the racket center, and one camera is accepted only when all center keypoints satisfy the configured score threshold (default `40.0`, min valid face keypoints `4`). The offline annotation path now converts the resulting racket 3D from mm to m before writing JSON/HTML artifacts.
- `2026-03-28`: tracker no longer replies to `/time_sync/ping` inside `DirectRos2Sink`. Instead, tracker startup now launches `ros2/start_time_sync.bat` as a dedicated child process and closes it on tracker exit. This keeps `time_sync` pong handling independent of tracker main-process scheduling while `/pc_car_loc` and `/predict_hit_pos` stay in-process under direct mode.
- `2026-03-28`: raw stitched tracker video is now saved as a `2x2` grid instead of a `1x4` strip, still using row-major camera order `103, 746 / 087, 029`. `test_src/annotate_video.py` was updated to annotate the same `2x2` layout, and it can still auto-detect older `1x4` recordings by video dimensions for backward compatibility.
- `2026-07-03`: tracker 用 rosbag 录制取代 pc_logger 事件接收器。
  - 新增独立 rosbag 录制边车 `src/rosbag_recorder.py`，由 `ros2/run_ros2.bat` 拉起，录制局域网内全部 ROS2 topic（`RecordOptions.all_topics`）到 `tracker_output/{run_id}_rosbag/`，与 tracker JSON 同 id；tracker JSON 的 `config.rosbag.bag_dir` 也记录该目录，便于 report viewer 配对加载。仅在保存日志时启动（`--no-log` 跳过），与 ros2_mode（发送方向，当前强制 off）无关。
  - 停止协议：`_close_sidecars` 创建 `{run_id}_rosbag.stop` 文件，子进程轮询到后 `Recorder.stop()` 写出 metadata.yaml，超时才 taskkill 兜底。jazzy 的 rosbag2_py 新 API：`record()` 非阻塞，必须先 `start_spin()`，否则本地 topic 都录不到。
  - 已删除 `src/pc_event_logger.py`、`src/pc_logger_protocol.py`，并从 `run_tracker.py` 移除 pc_logger 启动、`logger_control` 控制面（含各 Ros2Sink 的 `publish_logger_control` 与 `_publish_logger_control`）。`tracker_report_server.py` 仍能读旧 run 的 `_pc_logger.json`（找不到时优雅降级），未改动。
  - 组网（关键）：RK 端全用静态单播、`AllowMulticast=false`、`ROS_DOMAIN_ID=2`、cyclonedds。本机 PC 实际 IP 为 `192.168.50.230`（Wi-Fi），另有 Meta/VPN 虚拟网卡 `198.18.0.1`。已把 `ros2/cyclonedds.xml` 改成：`<NetworkInterface address="192.168.50.230" multicast="false"/>` 固定绑 LAN 网卡（避免选到 Meta 网卡，这是之前收不到 RK 的根因）、`AllowMulticast=false`、Peers 只留两台 RK（臂 `192.168.50.17`、底盘 `192.168.50.143`）。`src/ros2_support.py` 常量同步：`TRACKER_PC_IP=.230`、`CHASSIS_RK_IP=.143`（原为 `.68`，错的）。
  - RK 端已把 PC 配成 `.230` peer（用户完成），加 peer 前录制器要靠 learned-locator 发现远程、慢且不稳（普通 rclpy probe 能发现但 rosbag Recorder 常错过窗口）；加 peer 后发现秒级。实测：录制器 100ms 内订阅到 RK 的 `/net_test`，10s 录到 47 条，跨机录制通。
  - 防火墙：DDS 用 UDP `7400-7500`，需对该网卡的网络放行入站（本机实测已能收到，说明当前 Wi-Fi 网络已放行）。
- `2026-03-29`: `src/win_time_sync.py` now prints a 5-second summary while tracker is running, including ping receive count, receive rate, seq range/gaps, local inter-arrival interval, RK-side `t1` interval, inferred one-way delay jitter `((recv_i-recv_{i-1}) - (t1_i-t1_{i-1}))`, and local callback cost. `TimeSyncResponderProcess` in `src/run_tracker.py` no longer silences the child process, so these stats are visible in the same terminal/log stream as tracker when launched from `run_tracker_terminal.ps1`.
- `2026-07-05`: AprilTag 到车底盘中心的固定偏移按现场重新测量更新：`apriltag_center_to_car_base_offset` 从 `(60, 100, -340) mm` 改为 `(40, 160, -610) mm`（tag 中心在小车原点左方 40mm、后方 160mm、地面上方 610mm，偏移沿世界轴直接相加、不乘 yaw 的约定不变）。同步更新了 `src/config/arm_poe_racket_center*.json`（含两个 pruned15 变体）、`ArmCalibration/calibrate_poe_reprojection.py` 的导出常量、`src/config/arm_poe_racket_center.md` 与 `DEV.md` 的说明。
