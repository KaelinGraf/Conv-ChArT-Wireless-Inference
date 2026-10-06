"""QoS for the image and inference topics, in one place.

The Pi's camera publisher and the laptop's inference subscriber live in different
packages on different machines, and a QoS disagreement between them does not
raise: the two endpoints simply never match, nothing is delivered, and the only
symptom is silence. That failure mode is why these profiles are defined once here
rather than restated at each use site.

Only three of the policies below are actually negotiated (the DDS "requested vs
offered" rules), and all three failures look identical from the outside:

    reliability   a RELIABLE writer satisfies a RELIABLE or a BEST_EFFORT reader
    deadline      the writer's OFFERED period must be <= the reader's REQUESTED
    liveliness    the writer's lease must be <= the reader's, kind AUTOMATIC

history, depth and lifespan are NOT negotiated -- each end applies its own to its
own cache. In particular the reader's lifespan is inert: it has to be set on the
PUBLISHER to stop a stale frame being delivered late, which is why
image_pub_qos() carries one and IMAGE_SUB_QOS's is vestigial. The reader's depth
of 1 is what drops a frame that waited out a slow inference callback rather than
queueing it (test_node_pose_output.py depends on exactly that).

This lives in its own package, rather than in convchart_ros next to the node that
first used it, so that the Pi can import it without depending on convchart_ros --
whose cv_bridge exec_depend resolves to libopencv-dev and some fifty further apt
packages a camera node has no use for. Keep this module's imports to rclpy alone
and that stays true.
"""
from rclpy.duration import Duration
from rclpy.qos import (
    QoSDurabilityPolicy,
    QoSHistoryPolicy,
    QoSLivelinessPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)

# What IMAGE_SUB_QOS requests. A publisher has to offer no more than these or the
# subscription will not match it. Exported so a publisher can clamp itself against
# them instead of hard-coding the numbers a second time.
IMAGE_DEADLINE_CEILING_S = 0.2
IMAGE_LEASE_CEILING_S = 1.0

# The inference node's view of the image stream. Depth 1 and a 0.15 s lifespan
# together mean "the newest frame or nothing": a pose computed from a frame that
# has been sitting in a queue is worse than no pose, because the filter would
# fuse it as though it were current.
IMAGE_SUB_QOS = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    deadline=Duration(seconds=IMAGE_DEADLINE_CEILING_S),
    lifespan=Duration(seconds=0.15),
    liveliness=QoSLivelinessPolicy.AUTOMATIC,
    liveliness_lease_duration=Duration(seconds=IMAGE_LEASE_CEILING_S),
)

# Pose results back to whoever is steering. TRANSIENT_LOCAL so a consumer that
# starts late still gets the most recent poses rather than waiting for the next
# detection, and depth 10 because these are small and worth not dropping.
INFERENCE_RESULT_QOS = QoSProfile(
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
)


def image_qos_pub(deadline_s: float = 0.15,
                  lease_s: float = 0.5,
                  lifespan_s: float = 0.15) -> QoSProfile:
    """Build the camera publisher's side of IMAGE_SUB_QOS.

    @args:
        deadline_s: the period we promise to publish within. Must be at least the
            publish period (or we breach our own promise) and at most
            IMAGE_DEADLINE_CEILING_S (or the subscription will not match us). The
            default suits a 10 Hz stream: a 100 ms period plus 50 ms of jitter.
        lease_s: liveliness lease. 0.5 s matches what test_node_pose_output.py
            already offers, so the test probe and the real publisher agree.
        lifespan_s: how long a sample stays worth delivering. A frame that has not
            crossed the WiFi link within a frame and a half is of no use to a
            10 Hz pose pipeline and only competes with its successor for airtime.

    Raises ValueError rather than silently publishing into the void if either
    ceiling is exceeded. Callers are expected to clamp first and log why; this is
    the backstop against a later edit moving a default past the limit.
    """
    if deadline_s > IMAGE_DEADLINE_CEILING_S:
        raise ValueError(
            f'offered deadline {deadline_s} s exceeds the {IMAGE_DEADLINE_CEILING_S} s '
            'requested by IMAGE_SUB_QOS; the inference node would never match this '
            'publisher and the topic would be silent with no error anywhere')
    if lease_s > IMAGE_LEASE_CEILING_S:
        raise ValueError(
            f'offered liveliness lease {lease_s} s exceeds the {IMAGE_LEASE_CEILING_S} s '
            'requested by IMAGE_SUB_QOS; same silent non-match as the deadline')
    return QoSProfile(
        history=QoSHistoryPolicy.KEEP_LAST,
        depth=1,
        reliability=QoSReliabilityPolicy.RELIABLE,
        durability=QoSDurabilityPolicy.VOLATILE,
        deadline=Duration(seconds=deadline_s),
        lifespan=Duration(seconds=lifespan_s),
        liveliness=QoSLivelinessPolicy.AUTOMATIC,
        liveliness_lease_duration=Duration(seconds=lease_s),
    )
