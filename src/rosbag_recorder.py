"""独立 rosbag 录制进程。

由 run_tracker.py 通过 ros2/run_ros2.bat 启动，录制当前 ROS_DOMAIN_ID 下
局域网内全部 ROS2 topic，bag 目录与 tracker run id 同名（{run_id}_rosbag）。
--stop-file 指定的文件出现（或收到 KeyboardInterrupt）时停止录制，
并等待 rosbag2 写出 metadata.yaml。
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import rosbag2_py


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output", type=Path, required=True,
        help="bag 输出目录（必须不存在，由 rosbag2 创建）",
    )
    parser.add_argument(
        "--stop-file", type=Path, required=True,
        help="该文件出现时停止录制并退出",
    )
    args = parser.parse_args()

    storage_options = rosbag2_py.StorageOptions(uri=str(args.output))
    record_options = rosbag2_py.RecordOptions()
    record_options.all_topics = True
    record_options.disable_keyboard_controls = True
    recorder = rosbag2_py.Recorder(storage_options, record_options)

    # jazzy 新版 API：record() 非阻塞；订阅回调靠 start_spin() 的 executor 线程驱动，
    # 不 spin 的话 discovery 仍会打印 Subscribed 但消息一条都不会写入。
    recorder.start_spin()
    recorder.record()
    try:
        while not args.stop_file.exists():
            time.sleep(0.2)
    except KeyboardInterrupt:
        pass
    finally:
        recorder.stop()
        recorder.stop_spin()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
