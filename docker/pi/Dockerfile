# syntax=docker/dockerfile:1
# =============================================================================
# p4p Raspberry Pi image (linux/arm64)
#   ROS 2 Jazzy (ros-base) + numpy/scipy (Kalman filter, unconstrained MPC gain)
#   + osqp/qpsolvers (constrained MPC QP) + pyserial (Arduino Mega 2560 over
#   /dev/ttyACM0, IMU packets + motor commands) + tf2 / tf_transformations + both
#   WiFi-friendly RMWs (CycloneDDS, Zenoh). No ML dependencies of any kind.
#   Camera stack is opt-in: --build-arg WITH_CAMERA=1.
#
# Build context is ./docker (see compose.yaml). Same layout / entrypoint / user
# as the GPU image so a node developed on the laptop runs unchanged here.
# =============================================================================
ARG ROS_IMAGE=ros:jazzy-ros-base
FROM ${ROS_IMAGE}

ARG DEBIAN_FRONTEND=noninteractive
ARG WITH_CAMERA=0
ARG USERNAME=dev
ARG USER_UID=1000
ARG USER_GID=1000

ENV LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONUNBUFFERED=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

# ROS_DISTRO=jazzy comes from the base image. Python deps come from apt on purpose:
# they are prebuilt for arm64, ABI-matched to the ROS python packages, and the
# Pi never has to compile anything. (pip is there for anything not in apt.)
RUN apt-get update && apt-get install -y --no-install-recommends \
        sudo nano less usbutils iproute2 iputils-ping \
        python3-pip python3-numpy python3-scipy python3-serial python3-yaml \
        ros-${ROS_DISTRO}-tf2-ros \
        ros-${ROS_DISTRO}-tf2-geometry-msgs \
        ros-${ROS_DISTRO}-tf-transformations \
        ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
        ros-${ROS_DISTRO}-rmw-zenoh-cpp \
        ros-${ROS_DISTRO}-demo-nodes-py \
    && if [ "${WITH_CAMERA}" = "1" ]; then apt-get install -y --no-install-recommends \
        v4l-utils \
        ros-${ROS_DISTRO}-camera-ros \
        ros-${ROS_DISTRO}-v4l2-camera \
        ros-${ROS_DISTRO}-image-transport-plugins ; fi \
    && rm -rf /var/lib/apt/lists/* \
    # QP solver for the constrained-MPC stage (prebuilt aarch64 wheel; numpy/scipy stay apt's)
    && python3 -m pip install osqp==1.1.3 qpsolvers==4.13.0 \
    && source /opt/ros/${ROS_DISTRO}/setup.bash \
    && python3 -c "import rclpy, numpy, scipy, serial, yaml, tf_transformations, transforms3d, osqp, qpsolvers; print('pi deps OK numpy', numpy.__version__, 'scipy', scipy.__version__, 'pyserial', serial.__version__, 'osqp', osqp.__version__)"

# ros2_numpy (msgify / numpify between numpy arrays and ROS messages), same recipe as
# the GPU image. --no-deps: its metadata pins numpy==1.24.2 (no Python 3.12 wheel), and
# apt's numpy stays. 0.0.5 (PyPI) and upstream's humble/jazzy branches all carry two
# bugs, patched here and checked by the assertions below:
#   image.py     ndarray.tostring() (deprecated in numpy 1.26, gone in 2.3) -> tobytes()
#   geometry.py  Pose/Transform mix (w,x,y,z) and (x,y,z,w) quaternion orders; it now
#                imports tf_transformations (ROS x,y,z,w order, as upstream's jazzy
#                branch does) and zips every quaternion in that order
# Upstream also returns None for batched (N,4,4) input: msgify one 4x4 at a time.
RUN python3 -m pip install --no-deps ros2-numpy==0.0.5 \
    && d="$(python3 -m pip show ros2-numpy | sed -n 's/^Location: //p')/ros2_numpy" \
    && sed -i 's/contig\.tostring()/contig.tobytes()/' "$d/image.py" \
    && sed -i -e 's/^import transformations as transformations$/import tf_transformations as transformations/' \
              -e "s/zip(\['w', 'x', 'y', 'z'\]/zip(['x', 'y', 'z', 'w']/g" "$d/geometry.py" \
    && source /opt/ros/${ROS_DISTRO}/setup.bash \
    && python3 -c "import numpy as np, ros2_numpy as rnp; \
        from sensor_msgs.msg import Image; from geometry_msgs.msg import Pose, Transform; \
        a = np.arange(12, dtype=np.uint8).reshape(3, 4); \
        assert (rnp.numpify(rnp.msgify(Image, a, encoding='mono8')) == a).all(); \
        T = np.eye(4); T[:2, :2] = [[0, -1], [1, 0]]; q = [0, 0, 0.5 ** 0.5, 0.5 ** 0.5]; \
        assert all(np.isclose(abs(np.dot([m.x, m.y, m.z, m.w], q)), 1) \
                   for m in (rnp.msgify(Pose, T).orientation, rnp.msgify(Transform, T).rotation)); \
        assert all(np.allclose(rnp.numpify(rnp.msgify(M, T)), T) for M in (Pose, Transform)); \
        print('ros2_numpy OK: Image mono8, Pose/Transform in ROS (x, y, z, w) order')"

# Non-root user matching the Pi's login uid/gid; dialout for the Arduino's tty.
RUN if id ubuntu >/dev/null 2>&1; then userdel -r ubuntu; fi \
    && (getent group "${USER_GID}" >/dev/null || groupadd --gid "${USER_GID}" "${USERNAME}") \
    && useradd --uid "${USER_UID}" --gid "${USER_GID}" -m -s /bin/bash "${USERNAME}" \
    && usermod -aG sudo,dialout,plugdev,video "${USERNAME}" \
    && echo "${USERNAME} ALL=(ALL) NOPASSWD:ALL" > /etc/sudoers.d/"${USERNAME}" \
    && chmod 0440 /etc/sudoers.d/"${USERNAME}" \
    && mkdir -p /workspace && chown "${USER_UID}:${USER_GID}" /workspace

COPY common/p4p-env.sh /etc/p4p-env.sh
COPY common/entrypoint.sh /usr/local/bin/p4p-entrypoint
RUN chmod 0755 /usr/local/bin/p4p-entrypoint \
    && echo 'source /etc/p4p-env.sh' >> /etc/bash.bashrc
# interactive shells source it via bash.bashrc, `docker run` via the entrypoint, and
# non-interactive `bash -c` (docker compose exec, Makefile targets, IDE tasks) via BASH_ENV
ENV BASH_ENV=/etc/p4p-env.sh

ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

LABEL org.opencontainers.image.title="p4p-pi" \
      org.opencontainers.image.description="ROS 2 Jazzy pose receiver / Kalman filter / Arduino serial bridge for the Raspberry Pi" \
      p4p.ros="jazzy" p4p.camera="${WITH_CAMERA}"

WORKDIR /workspace
USER ${USERNAME}
ENTRYPOINT ["/usr/local/bin/p4p-entrypoint"]
CMD ["bash"]
