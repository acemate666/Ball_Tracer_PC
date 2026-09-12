"""独立 rosbag 录制进程。

由 run_tracker.py 通过 ros2/run_ros2.bat 启动，录制当前 ROS_DOMAIN_ID 下
局域网内全部 ROS2 topic，bag 目录与 tracker run id 同名（{run_id}_rosbag）。
--stop-file 指定的文件出现（或收到 KeyboardInterrupt）时停止录制，
并等待 rosbag2 写出 metadata.yaml。
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import time
from pathlib import Path

import rosbag2_py


async def record_imu_until_stopped(stop_file: Path, address: str) -> None:
    import rclpy
    from std_msgs.msg import String
    from racket_imu import TOPIC, stream_racket_imu

    rclpy.init()
    node = rclpy.create_node("racket_imu_recorder")
    publisher = node.create_publisher(String, TOPIC, 200)
    task = None
    try:
        # Start BLE only once the bag recorder has discovered the topic.
        deadline = time.perf_counter() + 5
        while publisher.get_subscription_count() == 0:
            if stop_file.exists():
                return
            if time.perf_counter() >= deadline:
                raise RuntimeError("Rosbag recorder did not subscribe to /racket/imu")
            await asyncio.sleep(0.05)
        task = asyncio.create_task(stream_racket_imu(
            address, lambda sample: publisher.publish(String(data=json.dumps(sample))),
        ))
        while not stop_file.exists():
            if task.done():
                task.result()
                raise RuntimeError("Racket IMU stream stopped unexpectedly")
            await asyncio.sleep(0.05)
    finally:
        try:
            if task is not None:
                task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await task
        finally:
            node.destroy_node()
            rclpy.shutdown()


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
    config = json.loads((Path(__file__).parent / "config" / "tracker.json").read_text(encoding="utf-8"))
    imu = config["racket_imu"]

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
        if imu["enabled"]:
            asyncio.run(record_imu_until_stopped(args.stop_file, imu["address"]))
        else:
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
