#!/usr/bin/env python3
"""Fetch one full-resolution frame from the camera node and write it to a file.

`ros2 service call` prints the response as a ~2 MB decimal byte array, which is
unusable, so this is the practical way to call image_full_res. It is also the
first tool of the calibration workflow: collect a board set with it, calibrate at
640x480 (NOT at 1600x1200 -- see p4p_camera/frames.scale_intrinsics), then fill
in cfg/cfg.yaml's CAMERA.K and CAMERA.dist.

    python3 tools/grab_full_res.py -o /tmp/frame.png
    python3 tools/grab_full_res.py -o board_%03d.png -n 20 --interval 1.5
    python3 tools/grab_full_res.py --max-age 0.5 -o fresh.png

Look at the first frame you capture. Two failure signatures matter, because both
produce a NEARLY right image and neither is visible in a thumbnail:

  * a diagonal shear, or a garbage strip down the right edge -> the raw buffer's
    line stride is being mishandled;
  * vertical banding with a 4-pixel period -> the RAW10 unpack grouping is off.

Both are fatal to the refiner, which reads sub-pixel offsets out of 24x24 crops.
"""
import argparse
import sys
import time

from convchart_interfaces.srv import GetFullResImage
import rclpy
from rclpy.node import Node


def parse_args(argv):
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('-o', '--output', default='frame.png',
                   help='output path; with -n, may contain a printf index such '
                        'as board_%%03d.png (default: frame.png)')
    p.add_argument('-n', '--count', type=int, default=1,
                   help='how many frames to grab (default: 1)')
    p.add_argument('--interval', type=float, default=1.0,
                   help='seconds between grabs when -n > 1 (default: 1.0)')
    p.add_argument('--max-age', type=float, default=0.0,
                   help='refuse a retained frame older than this; 0 accepts any '
                        'age (default: 0)')
    p.add_argument('--service', default='image_full_res',
                   help='service name, if remapped (default: image_full_res)')
    p.add_argument('--timeout', type=float, default=10.0,
                   help='seconds to wait for the service and each response')
    return p.parse_args(argv)


class Grabber(Node):
    """A one-shot client for the camera node's full-resolution service."""

    def __init__(self, service: str):
        super().__init__('grab_full_res')
        self._client = self.create_client(GetFullResImage, service)
        self._service = service

    def wait(self, timeout_s: float) -> bool:
        return self._client.wait_for_service(timeout_sec=timeout_s)

    def grab(self, max_age: float, timeout_s: float):
        future = self._client.call_async(GetFullResImage.Request(max_age=max_age))
        rclpy.spin_until_future_complete(self, future, timeout_sec=timeout_s)
        if not future.done():
            return None, f'{self._service} did not answer within {timeout_s:.1f} s'
        res = future.result()
        if not res.success:
            return None, res.message
        return res, res.message


def main(argv=None) -> int:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    rclpy.init()
    node = Grabber(args.service)
    try:
        if not node.wait(args.timeout):
            print(f'error: no {args.service} service on the graph. Is camera_node '
                  'running, and are ROS_DOMAIN_ID and RMW_IMPLEMENTATION the same '
                  'as its?', file=sys.stderr)
            return 2

        failures = 0
        for i in range(args.count):
            if i:
                time.sleep(args.interval)
            res, message = node.grab(args.max_age, args.timeout)
            if res is None:
                print(f'error: {message}', file=sys.stderr)
                failures += 1
                continue
            # printf-style only when the user actually asked for one, so a plain
            # name with a literal % in it is not mangled.
            try:
                path = args.output % i if args.count > 1 else args.output
            except TypeError:
                path = args.output if args.count == 1 else f'{i:03d}_{args.output}'
            with open(path, 'wb') as fh:
                fh.write(bytes(res.image.data))
            stamp = res.capture_stamp
            print(f'{path}: {len(res.image.data)} bytes, captured at '
                  f'{stamp.sec}.{stamp.nanosec:09d} ({message})')
        return 1 if failures else 0
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == '__main__':
    sys.exit(main())
