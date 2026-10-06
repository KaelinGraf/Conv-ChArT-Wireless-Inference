# syntax=docker/dockerfile:1
# =============================================================================
# p4p Raspberry Pi image (linux/arm64)
#   ROS 2 Jazzy (ros-base) + numpy/scipy (Kalman filter, unconstrained MPC gain)
#   + osqp/qpsolvers (constrained MPC QP) + pyserial (Arduino Mega 2560 over
#   /dev/ttyACM0, IMU packets + motor commands) + tf2 / tf_transformations + both
#   WiFi-friendly RMWs (CycloneDDS, Zenoh). No ML dependencies of any kind.
#   + OpenCV (p4p_camera's v4l2 capture, downscale and PNG encode -- all the
#   camera needs; its CSI pipeline is set up on the host, see docker/README.md).
#   --build-arg WITH_CAMERA=1 adds v4l-utils, v4l2_camera and the image_transport
#   plugins; its Arducam libcamera + picamera2 step cannot install on this base.
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
        python3-opencv wget ca-certificates \
        ros-${ROS_DISTRO}-tf2-ros \
        ros-${ROS_DISTRO}-tf2-geometry-msgs \
        ros-${ROS_DISTRO}-tf-transformations \
        ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
        ros-${ROS_DISTRO}-rmw-zenoh-cpp \
        ros-${ROS_DISTRO}-demo-nodes-py \
    # No ros-${ROS_DISTRO}-camera-ros on purpose: Arducam's installer REPLACES the
    # distro libcamera, and camera-ros is built against the distro libcamera's ABI,
    # so keeping it invites a confusing load failure in a package p4p_camera does
    # not use. v4l-utils is in the bring-up checklist, image-transport-plugins is
    # handy for `ros2 run image_transport republish` when debugging, and
    # python3-prctl / libatlas-base-dev are picamera2's dependencies. (So is
    # python3-kms++, for its DRM preview only, but noble does not package it and
    # listing it failed the whole build.)
    && if [ "${WITH_CAMERA}" = "1" ]; then apt-get install -y --no-install-recommends \
        v4l-utils \
        python3-prctl libatlas-base-dev \
        ros-${ROS_DISTRO}-v4l2-camera \
        ros-${ROS_DISTRO}-image-transport-plugins ; fi \
    && rm -rf /var/lib/apt/lists/* \
    # QP solver for the constrained-MPC stage (prebuilt aarch64 wheel; numpy/scipy stay apt's)
    && python3 -m pip install osqp==1.1.3 qpsolvers==4.13.0 \
    && source /opt/ros/${ROS_DISTRO}/setup.bash \
    && python3 -c "import rclpy, numpy, scipy, serial, yaml, cv2, tf_transformations, transforms3d, osqp, qpsolvers; print('pi deps OK numpy', numpy.__version__, 'scipy', scipy.__version__, 'pyserial', serial.__version__, 'cv2', cv2.__version__, 'osqp', osqp.__version__)"

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

# Arducam's libcamera fork + their picamera2, for the Pivariety OV2311 (WITH_CAMERA=1).
# The kernel driver and the dtoverlay are the HOST's job -- see docker/README.md.
#
# CONFIRMED ON HARDWARE (2026-10): this step cannot succeed on this base. Arducam
# publish libcamera packages for Raspberry Pi OS (Debian bookworm, trixie) only;
# on Ubuntu noble their installer stops with "Unsupported package", so picamera2
# never imports. That is why p4p_camera defaults to the v4l2 backend, with the
# CSI pipeline configured on the host by docker/camera-pipeline-pi.sh. The way
# back to picamera2 would be building Arducam's libcamera fork from source here.
# The build is NOT allowed to fail on this: it warns and carries on, so a
# WITH_CAMERA=1 image is still usable with the other backends.
ARG ARDUCAM_SCRIPT_URL=https://github.com/ArduCAM/Arducam-Pivariety-V4L2-Driver/releases/download/install_script/install_pivariety_pkgs.sh
ARG ARDUCAM_PICAMERA2_REF=main
RUN if [ "${WITH_CAMERA}" != "1" ]; then exit 0; fi; \
    warn() { echo "WARNING(arducam): $*" >&2; }; \
    ok=1; \
    echo 'APT::Get::Assume-Yes "true";' > /etc/apt/apt.conf.d/90assumeyes; \
    apt-get update || { warn "apt-get update failed"; ok=0; }; \
    if [ "$ok" = 1 ]; then \
        wget -qO /tmp/install_pivariety_pkgs.sh "${ARDUCAM_SCRIPT_URL}" \
            || { warn "could not fetch ${ARDUCAM_SCRIPT_URL}"; ok=0; }; \
    fi; \
    if [ "$ok" = 1 ]; then \
        chmod +x /tmp/install_pivariety_pkgs.sh; \
        ( cd /tmp && ./install_pivariety_pkgs.sh -p libcamera_dev ) || warn "libcamera_dev failed"; \
        ( cd /tmp && ./install_pivariety_pkgs.sh -p libcamera_apps ) || warn "libcamera_apps failed"; \
        python3 -m pip install --no-deps \
            "git+https://github.com/ArduCAM/picamera2@${ARDUCAM_PICAMERA2_REF}" \
            || warn "the picamera2 fork would not install"; \
        apt-mark hold libcamera0 libcamera-dev libcamera-apps libcamera-ipa >/dev/null 2>&1 \
            || warn "could not apt-mark hold libcamera (an apt upgrade may silently replace it)"; \
    fi; \
    python3 -c "import libcamera, picamera2; print('picamera2 OK', picamera2.__version__)" \
        || warn "picamera2 does not import in this image; use p4p_camera's v4l2 backend or build the fork from source (see above)"; \
    rm -f /etc/apt/apt.conf.d/90assumeyes /tmp/install_pivariety_pkgs.sh; \
    rm -rf /var/lib/apt/lists/*; \
    exit 0

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
