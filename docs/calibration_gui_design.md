# Camera calibration GUI: design

Status: agreed 2026-10-05; updated 2026-10-10. This file is the spec the implementation follows.
Update 2026-10-10: the Pi camera node (`p4p_camera`) exists and serves full-resolution stills as
`image_full_res` (`GetFullResImage`), so `calibrate` and `fake_camera` use that service; the original
`capture` (`CaptureFrame`) service was never served by the Pi. `fake_camera` stands in for the Pi
camera during development and in tests.
Update 2026-10-14: the corners come from the deployed model by default (`detector` parameter, `model` or
`opencv`), and with the model the live view shows the board's distance from the camera in large type, to
set the lens focus by before calibrating. See "Corner detector" and "Distance readout".
Update 2026-10-14: `resume` carries on in an earlier session's folder, so views can be added until the
solve converges. See "Resume".

## Goal

Calibrate the OV2311's intrinsics `K` and distortion `dist` for the inference pipeline with a ChArUco
board: a person positions the board, watches a live preview, and triggers full-resolution captures.
Each capture is checked immediately; once enough are kept, the calibration is solved and then refined
with every further capture, for as long as the user keeps going.

## Decisions

| topic | decision |
|---|---|
| frame selection | human-triggered stills over a service, not automatic selection from video |
| preview | the `image` stream (640x480, the inference resolution) |
| stills | `image_full_res` service, full resolution: 1600x1200, the 4:3 crop of the sensor's 1600x1300 mode |
| solve resolution | full resolution; `K` is then scaled to the stream resolution for inference |
| board | user-configurable (squares, sizes, dictionary, first marker id, legacy pattern); defaults = Conv-ChArT's `dcc` board |
| `auto_replace` | `True`: `K`, `dist`, `image_size` (at stream resolution) are written into the config's `CAMERA` block after every successful solve, after a one-time backup of the config. `False`: config untouched |
| log file | always written, in both modes, after every solve and on exit |
| coverage map | shown over the live view (toggle `M`) |
| corner detector | the deployed model (`MODEL` block, via `inference.inference_pipeline`) by default; OpenCV's ChArUco detector with `detector:=opencv` |
| distance readout | with the model: the board centre's z and range from the stream and the config's CAMERA, top right of the live view (toggle `D`) |
| resume | `resume:=<session folder>|latest` re-detects that session's kept views from their frames, solves at once and adds new captures to the same folder |
| Pi camera node | `p4p_camera`; `fake_camera` implements the same `image` + `image_full_res` interface |

## Components

| file | owner layer | role |
|---|---|---|
| `convchart_interfaces/srv/GetFullResImage.srv` | interface | full-res still on request, shared with `p4p_camera` |
| `convchart_ros/calib_utils/calib_board.py` | pure | `BoardSpec`, `board_spec_from_cfg`, `make_board` |
| `convchart_ros/calib_utils/calib_session.py` | pure (no ROS, no GUI) | detection + usability, kept views, solve, coverage, intrinsics scaling, config + log writing |
| `convchart_ros/calib_utils/calib_model.py` | no ROS, GPU | the model as corner detector (`corners`) and range finder (`board_range`); `check_model_board` |
| `convchart_ros/calib_utils/calib_view.py` | pure (no ROS) | composes the window canvas: live / review modes, overlays, HUD, coverage map |
| `convchart_ros/calib_utils/calib_synth.py` | pure | synthetic ChArUco renders from a known camera (fake camera + tests) |
| `convchart_ros/calib_utils/fake_camera.py` | ROS node | serves `image` and `image_full_res` from synthetic renders |
| `convchart_ros/calibrate.py` | ROS node + GUI | `CalibrateCam` node, `run_gui`, `main` |
| `cfg/cfg.yaml` | config | `CALIBRATION` block (all keys optional) |

All paths under `src/ros/src/` unless they start with `cfg/` or `docs/`.

## Interfaces

`GetFullResImage.srv` (defined with `p4p_camera`; see its comments for the full contract):

```
float32 max_age                     # refuse a retained frame older than this (s); 0 accepts any
---
bool success
string message                      # why not, when success is false
builtin_interfaces/Time capture_stamp
sensor_msgs/CompressedImage image   # PNG, mono8, full resolution
```

| name | type | notes |
|---|---|---|
| `image` | `sensor_msgs/CompressedImage` | preview / inference stream, PNG mono8. QoS: RELIABLE, VOLATILE, KEEP_LAST 1 |
| `image_full_res` | `convchart_interfaces/srv/GetFullResImage` | the camera's newest full-res frame; `calibrate` sends `max_age = capture_max_age_s` (0.5 s) so a stopped camera's last frame is refused |

## Configuration (`cfg/cfg.yaml`)

```yaml
CALIBRATION:
  board:                       # physical calibration board; defaults reproduce dcc's board
    squares: [5, 5]            # [cols, rows] of chessboard squares (rectangular boards allowed)
    square_length_m: 1.0       # square side; any unit works for K/dist (1.0 = unitless, like dcc)
    marker_length_m: null      # marker side; null -> marker_ratio * square_length_m
    marker_ratio: 0.7
    dictionary: DICT_5X5_50    # cv2.aruco predefined dictionary name
    first_marker_id: 0         # markers use consecutive ids from here
    legacy_pattern: false      # true for boards generated by OpenCV < 4.6
  min_corners: null            # null -> max(6, ceil(0.5 * inner corners)); 8 for a 5x5 board
  min_views: 12                # kept views before the first solve
  review_s: 2.0                # how long a capture's result stays on screen
  outlier_factor: 3.0          # drop views whose reprojection error > factor * median
  fix_k3: false
  coverage_grid: [6, 8]        # [rows, cols] of the coverage map over the full frame
  converge_rel_sigma_f: 0.002  # converged when sigma(fx)/fx and sigma(fy)/fy below this ...
  converge_sigma_c_px: 1.0     # ... and sigma(cx), sigma(cy) below this (full-res pixels)
```

`auto_replace` writes (stream resolution; other config keys and their order are preserved; comments are
not, since the file is rewritten through PyYAML):

```yaml
CAMERA:
  K: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
  dist: [k1, k2, p1, p2, k3]
  image_size: [640, 480]
```

`inference.py` reads only `K` and `dist`; `image_size` records which resolution `K` belongs to.

## Module APIs

### `calib_board.py`

```python
@dataclass(frozen=True)
class BoardSpec:
    squares: tuple[int, int] = (5, 5)    # (cols, rows)
    square_length: float = 1.0
    marker_length: float | None = None   # None -> marker_ratio * square_length
    marker_ratio: float = 0.7
    dictionary: str = "DICT_5X5_50"
    first_marker_id: int = 0
    legacy_pattern: bool = False
    marker_side: float                   # property
    n_inner_corners: int                 # property, (cols-1)*(rows-1)

def board_spec_from_cfg(board_cfg: dict | None) -> BoardSpec   # CALIBRATION.board keys above
def make_board(spec: BoardSpec) -> cv2.aruco.CharucoBoard      # validates; default == dcc board
```

### `calib_session.py` (pure)

```python
@dataclass(frozen=True)
class CalibrationSettings:
    board: BoardSpec = BoardSpec()
    min_corners: int | None = None       # resolved by .min_corners_resolved
    min_views: int = 12
    review_s: float = 2.0
    outlier_factor: float = 3.0
    fix_k3: bool = False
    coverage_grid: tuple[int, int] = (6, 8)
    converge_rel_sigma_f: float = 0.002
    converge_sigma_c_px: float = 1.0
    min_corners_resolved: int            # property

def settings_from_cfg(cfg: dict | None) -> CalibrationSettings   # reads cfg["CALIBRATION"]

@dataclass
class Detection:
    kept: bool
    reason: str                          # "" when kept; else exactly one of the reasons below
    corners: np.ndarray                  # (N, 2) float64, full-res pixel coords (pixel-centre convention)
    ids: np.ndarray                      # (N,) int
    image_size: tuple[int, int]          # (W, H)
    obj_points: np.ndarray | None        # (N, 3) float32 when kept
    img_points: np.ndarray | None        # (N, 2) float32 when kept

REJECT_NO_BOARD = "no board found"
# "too few corners (n/min)"  e.g. "too few corners (5/8)"
REJECT_COLLINEAR = "corners in a line"   # ids span < 2 rows or < 2 cols of the inner-corner grid

@dataclass
class Solution:
    rms: float
    K: np.ndarray                        # (3, 3) full resolution
    dist: np.ndarray                     # (5,)
    std: dict[str, float]                # fx fy cx cy k1 k2 p1 p2 k3
    per_view_errors: dict[int, float]    # view index -> RMS px, active views only
    n_views: int
    dropped: list[int]                   # view indices dropped as outliers by this solve
    image_size: tuple[int, int]
    converged: bool

class CalibrationSession:            # thread-safe; solve() works on a snapshot of the views
    def __init__(self, settings: CalibrationSettings, detect: CornerDetector | None = None): ...
        # detect: grey frame -> (corners (N, 2) pixel-centred, ids (N,) OpenCV ChArUco ids); None: OpenCV
    def evaluate(self, frame: np.ndarray) -> Detection            # detect + checks; does not record
    def submit(self, det: Detection) -> int | None                # kept -> view index; rejected -> counted, None
    def submit(self, det, index=None, dropped=False) -> int | None  # index/dropped: a view restored on resume
    def carry_over(self, rejected=0, next_index=0, history=None)  # an earlier session's counts and solves
    def undo(self) -> int | None                                  # removes last kept view, returns its index
    kept_count, rejected_count, active_count, dropped_count       # properties
    ready: bool                                                   # active_count >= min_views
    def solve(self) -> Solution                                   # RuntimeError if not ready
    solution: Solution | None                                     # latest
    history: list[dict]                                           # {n_views, rms, fx, fy, cx, cy} per solve
    def coverage(self) -> np.ndarray                              # (rows, cols) int: kept corners per cell
    def coverage_fraction(self) -> float                          # cells with >= 1 corner / all cells

@dataclass(frozen=True)
class SavedSession: folder; views: [(index, file, dropped)]; rejected: int; next_index: int; log: dict | None
def read_saved_session(folder) -> SavedSession                  # ValueError without frames/ or with a bad log
def latest_saved_session(root) -> Path | None                   # newest folder (by name) holding a kept view

def scale_intrinsics(K, full_size, target_size) -> np.ndarray    # uniform scale only (ValueError otherwise)
def build_log(session, settings, meta: dict) -> dict             # log document, see "Log file"
def write_log(path, log: dict) -> None                            # atomic YAML write
def write_camera_to_cfg(cfg_path, K, dist, image_size, backup_path=None) -> None
```

- Detection: `detect(gray)` when given (the model), else `cv2.aruco.CharucoDetector(board).detectBoard(img)`;
  points from `board.matchImagePoints`.
  BGR input is converted to grey. Corners are moved into the pixel-centre convention by subtracting
  `charuco_offset_px()`: OpenCV 4.7-4.13 report ChArUco corners +0.5 px off pixel centres on both axes
  (removed upstream in 4.14 / 5.0, opencv#28380). The offset is measured once per process on an exactly
  known board; an unknown convention raises instead of biasing `cx`, `cy`.
- `solve()`: `cv2.calibrateCameraExtended` over active views; warm-started with
  `CALIB_USE_INTRINSIC_GUESS` from the previous solution when the image size matches;
  `CALIB_FIX_K3` when `fix_k3`. Outlier pass: views with error > `outlier_factor` * median are marked
  dropped (never below `min_views` active) and the solve is repeated once. `converged` uses the
  thresholds above on the full-res standard deviations.
- `scale_intrinsics`: `s = target_w / full_w` (must equal `target_h / full_h`);
  `fx' = s*fx`, `fy' = s*fy`, `cx' = s*(cx + 0.5) - 0.5`, `cy' = s*(cy + 0.5) - 0.5`; `dist` unchanged.
- `write_camera_to_cfg`: load YAML, set `CAMERA.K` (3x3 nested floats), `CAMERA.dist`,
  `CAMERA.image_size`; keep all other keys and order; copy the original to `backup_path` first if given
  and not already present; write atomically.

### `calib_model.py`

```python
MODEL_BOARD = BoardSpec()                                       # the only board the model knows, any square size
def check_model_board(spec: BoardSpec) -> None                 # ValueError "... use detector:=opencv" otherwise
@dataclass(frozen=True)
class BoardRange: z: float; range: float                        # board centre, BOARD.square_length_m units
class ModelDetector:                                            # one inference_pipeline, one call at a time
    def __init__(self, cfg_path): ...                           # MODEL + BOARD of the config; RuntimeError if unloadable
    def corners(self, frame) -> tuple[np.ndarray, np.ndarray]   # any resolution; sorted by id, each id once
    def board_range(self, frame, K, dist) -> BoardRange | None  # the pipeline's pose; None without one
```

#### Corner detector

- `inference_pipeline.run_inference` on the full-resolution still: the detector sees it downsized to its
  640x480 input, the refiner places each corner at full resolution. Corners in the still's pixels,
  pixel-centre convention (no offset, unlike OpenCV 4.7-4.13's detector).
- The model's corner ids are OpenCV's ChArUco ids for the 5x5 `DICT_5X5_50` board (`canon_lattice`:
  index i at ((i % 4) + 1, (i // 4) + 1) squares, the same row-major order as
  `board.getChessboardCorners()`), so no mapping. Checked on renders against OpenCV's detector and the
  projected truth (`test_calib_model.py`).
- The pipeline undistorts the corners with K and dist before its lattice gate (RANSAC homography from
  the ids' lattice positions; disagreeing corners lose their id) and its id recovery. Calibration must
  not depend on the CAMERA block it replaces, and run without K the pipeline would use the config's
  stream-resolution K on a full-resolution still. `corners()` therefore passes `dist = 0` (the
  undistortion becomes the identity, whatever K) and a nominal K, and scales `lattice_tol_px` by
  still width / detector input width (3 px -> 7.5 px at 1600 wide): the tolerance as tuned on the
  detector's input, where the coarse positions it gates are quantised. On 48 synthetic stills
  (fx 892, k1 0.05, k2 -0.10) this keeps 766/768 corners with no wrong id, the same as gating with the
  true K/dist; a stale K/dist at the default tolerance kept 744.
- `src/inference.py` is not changed: the tolerance is set on the pipeline instance's `PipelineConfig`
  for the call and restored after it.

#### Distance readout

- A thread in `calibrate` runs `board_range` on the newest stream frame, at most every 0.1 s, with the
  config's CAMERA K (scaled from `CAMERA.image_size` to the stream size when they differ) and dist, as
  they were when the node started. The board centre is ((n + 1) / 2, (n + 1) / 2) squares from the
  pose's origin; `z` is its depth, `range` its distance from the camera centre.
- A reading stays up for 0.5 s after its frame; then "no board". Without `CAMERA.K` the readout says
  "no CAMERA.K"; with `detector:=opencv` there is none.

### `calib_view.py` (pure)

```python
@dataclass
class HudState:
    kept: int
    rejected: int
    min_views: int
    status: str                          # "collecting 5/12", "solving...", "rms 0.21 px | fx 1402.3 +- 0.9 | ..."
    coverage_fraction: float | None = None
    converged: bool = False
    verdict: str | None = None           # review only: "KEPT" or "REJECTED: <reason>"
    verdict_ok: bool | None = None
    message: str | None = None           # transient error, e.g. "capture timed out"
    help: str = "SPACE capture   U undo   M coverage   Q quit"
    distance: tuple[float, float] | str | None = None   # live: (z, range) m drawn large, or "no board"

def scale_points(points, src_size, dst_size) -> np.ndarray     # pixel-centre convention, per axis
def compose_live(frame, hud: HudState, coverage=None, canvas_size=(640, 480), show_coverage=True) -> np.ndarray
def compose_review(frame_full, corners_full, kept: bool, hud: HudState, canvas_size=(640, 480)) -> np.ndarray
```

- Output: BGR uint8 canvas of exactly `canvas_size` (W, H) in both modes, whatever the input size, so
  the window never changes size. Mono input is converted to BGR.
- Review: the full-res frame is resized to the canvas (`INTER_AREA`) and the corners are scaled with
  `scale_points`; green markers when kept, red when rejected; a verdict banner. Markers are drawn after
  the HUD bands so they stay visible when the board is near the top or bottom edge.
- Coverage: the grid spans the frame uniformly; empty cells get only grid lines, covered cells a
  green fill whose opacity rises with the count (saturating at 3).
- HUD: counters, coverage %, status, transient message (red), key help.
- Distance readout (live only): a dark box at the top right under the top band, "z 0.80 m" in large white
  type over "range 0.83 m"; a word ("no board") in large amber type instead; nothing when `None`.

### `calib_synth.py` and `fake_camera.py`

```python
DEFAULT_FULL_SIZE = (1600, 1200); DEFAULT_STREAM_SIZE = (640, 480)
DEFAULT_K_FULL = [[1400.0, 0.0, 799.5], [0.0, 1400.0, 599.5], [0.0, 0.0, 1.0]]
DEFAULT_DIST = [-0.12, 0.05, 0.0008, -0.0005, 0.0]

def diverse_poses(spec, K, size, n=24) -> list[tuple[np.ndarray, np.ndarray]]   # deterministic
def render_view(board, spec, K, dist, rvec, tvec, size, px_per_square=80, background=140,
                noise_std=2.0, blur_sigma=0.7, seed=0) -> tuple[np.ndarray, np.ndarray, np.ndarray]
    # (frame uint8 (H, W), corners (n_inner, 2) float64 via cv2.projectPoints with dist, ids (n_inner,))
def blank_frame(size, seed=0) -> np.ndarray
```

- Rendering: ideal pinhole render via the homography `K [r1 r2 t] S` (`S` maps board-image pixels to
  board metres, pixel-centre convention), then distortion by remapping each output pixel to its
  undistorted location (`cv2.undistortPoints` on the pixel grid, cached per camera), blur, noise.
- Poses cover a 3x3 grid of image regions, near and far, tilts of about 15-40 degrees about both board
  axes and +-20 degrees in-plane, with the whole board inside the frame.
- `FakeCamera(Node)`: publishes `image` (current pose rendered at full res, `INTER_AREA` to the stream
  size, PNG) at `rate_hz`; serves `image_full_res` (current pose at full res, fresh noise), then advances to
  the next pose (cycling). Options: `blank_every` (every Nth capture is blank), `reply_delay_s`.
  Exposes `K_full`, `dist`, `poses`. Board from an optional config's `CALIBRATION.board`.

### `calibrate.py`

```python
class CalibrateCam(Node):
    def __init__(self, cfg_pth: str = "", auto_replace: bool = False, output_dir: str = "",
                 capture_timeout_s: float = 5.0, capture_max_age_s: float = 0.5,
                 detector: str = "model", resume: str = ""): ...
    def request_capture(self) -> bool       # False while a request is in flight or a review is showing
    def undo(self) -> bool
    def state(self) -> GuiState             # snapshot for the GUI loop
    def close(self) -> None                 # stop the solve worker, final log write
def run_gui(node: CalibrateCam) -> None     # main thread
def main(args=None) -> None
```

- Constructor args are the defaults of the ROS parameters `config`, `auto_replace`, `output_dir`,
  `capture_timeout_s`, `capture_max_age_s`, `detector`, `resume`; parameters win. `detector:=model` refuses a
  board other than the model's and a config whose model does not load, naming `detector:=opencv`. Loads the config with `yaml.safe_load`. `main()` needs
  `-p config:=<cfg.yaml>`; a bad config or parameter ends it with one line, `calibrate: <reason>`.
- Subscribes to `image` (callback group "video"); the `image_full_res` client uses callback group "calib".
- Capture flow: `request_capture` -> `call_async`; on the response, decode the PNG, `evaluate`, `submit`,
  save the PNG (`frames/view_NNN.png` or `frames/rejected_NNN.png`), start a review of `review_s`
  seconds, and queue a solve when `ready`. A timeout cancels the request and shows a message.
- Solve worker: one background thread, single-flight with a "dirty" flag (a capture during a solve
  queues exactly one more solve). After each solve: write the log; with `auto_replace`, back up the
  config once (to `<session folder>/<config name>.bak`) and write `CAMERA` at stream resolution (skipped, and noted in the log, until a stream
  frame has been seen).
- `undo` removes the last kept view, renames its file to `undone_NNN.png`, and queues a re-solve if
  still `ready`.

### Resume

`resume` names an earlier session folder, or `latest`: the newest folder under `output_dir` (by name,
which is its start time) holding a `frames/view_NNN.png`. No new folder is made; the session carries on
in that one.

- The frames are the record of the views: `view_NNN.png` are the kept views (undo renamed the others to
  `undone_NNN.png`), `rejected_NNN.png` the rejected captures. The log adds which views a solve
  dropped, the start time, `config_written`, earlier resumes and the solve history.
- Each kept view's frame goes through `evaluate` and `submit` with the configured detector, in index
  order, under its own index; views a solve dropped stay dropped. A view the detector now rejects (or
  whose frame cannot be read) counts as rejected and is left out of the views and the log; its file
  keeps its name, so a later resume tries it again. Rejected captures are not detected again: they
  stay rejected.
- The rejected captures, the solve history and the indices used carry over: new views are numbered
  above every `view_`, `undone_` and logged index, and rejected files go on from the last number.
- A solve starts at once when `min_views` views are active, so status, coverage and convergence come
  back as they were (the status line says `resumed <folder>: N views` for 6 s); otherwise the log is
  rewritten at once.
- A log naming a board of another layout than `CALIBRATION.board` is refused (`ValueError`).
- The log keeps the first `started`, and `session.resumed` gets
  `{at, detector, views_restored, views_now_rejected}` per resume. `session.detector` is the detector
  of the latest run. `auto_replace`, undo (restored views included) and the backup work as before; the
  backup is made once per folder.

## Threading model

- Main thread: `run_gui` (OpenCV window functions must stay on one thread that keeps calling
  `waitKey`).
- Background thread: a `MultiThreadedExecutor.spin_once(0.1 s)` loop with a stop flag (stream callback,
  capture response). `main()` ends it through the flag and never calls `executor.shutdown()` while it
  is spinning, which can hang.
- Solve worker thread: `CalibrationSession.solve()` and file writes.
- Shared state behind one lock in the node; the GUI only reads snapshots from `state()`.

## GUI behaviour

- Window created once at the canvas size (the stream frame size; (640, 480) until a frame arrives).
- Live mode: latest stream frame, coverage overlay, HUD.
- `SPACE`: capture (ignored while one is in flight or a review is showing). Review mode for `review_s`
  seconds: the captured frame scaled into the same canvas, corners drawn, verdict banner
  ("KEPT" / "REJECTED: reason"), counters updated.
- `U` undo the last kept view, `M` toggle the coverage map, `D` toggle the distance readout,
  `Q`/`ESC`/closing the window: quit (final
  log written). Keys are read with `waitKeyEx`, so arrow and page keys don't alias to commands.
- Status line: `collecting n/min_views`, then `solving...`, then RMS with fx, fy, cx, cy and their
  standard deviations, and `converged` once the thresholds are met; `| N outliers dropped` once a solve
  drops views, `| capturing...` while a request is in flight, and a message while no stream frame has
  arrived or the stream has stalled.

## Output

Session folder: `<output_dir>/<YYYYmmdd-HHMMSS>/`, where `output_dir` defaults to
`<config dir>/../calibration_output` (gitignored). It holds `calibration_log.yaml`, `frames/` and, with
`auto_replace`, the config backup. Folder names use the container's clock, which is UTC in p4p-gpu.

```yaml
session: {started, config, auto_replace, config_written, config_backup, detector, resumed: []}
board: {squares, square_length_m, marker_length_m, dictionary, first_marker_id, legacy_pattern}
images: {full_size: [W, H], stream_size: [W, H] | null}
counts: {kept, rejected, dropped, active}
coverage_fraction: 0.0-1.0
solution:                      # null until the first solve
  rms_px, converged
  K_full, dist, std: {fx, fy, cx, cy, k1, k2, p1, p2, k3}
  K_stream, stream_size        # null until a stream frame has been seen
views: [{index, file, n_corners, error_px, dropped}]
history: [{n_views, rms_px, fx, fy, cx, cy}]
```

## Testing

| file | checks |
|---|---|
| `test/test_calib_board.py` | default board identical to dcc's; ids, rectangular boards, validation |
| `test/test_calib_session.py` | evaluate (kept / no board / too few / collinear), submit/undo/counts, solve recovers a known K/dist from synthetic correspondences, uncertainty shrinks with views, outlier dropped, warm start, coverage, `scale_intrinsics` exact vs projection, settings defaults, config write keeps other keys + backup, log contents |
| `test/test_calib_view.py` | canvas size constant across modes and input sizes, overlay pixels where `scale_points` says, `scale_points` exact, coverage fill, HUD drawn |
| `test/test_calib_model.py` | model ids are OpenCV's ids and its corners pixel-centred and on the truth, calibration on its corners recovers K/dist, no board, `board_range` within 2 % from 0.19 to 1.5 m, other boards refused (GPU, checkpoints) |
| `test/test_calib_synth.py` | detected corners on renders match the projected ground truth (< 0.3 px median), poses keep the board in frame |
| `test/test_fake_camera.py` | stream at stream size, capture at full size, captures advance poses, `blank_every`, `reply_delay_s` |
| `test/test_calibrate_pi_camera.py` | captures through the Pi's own camera node (mock backend): kept at full resolution, a frame older than `capture_max_age_s` refused |
| `test/test_calibrate_node.py` | headless end to end with `fake_camera`: K/dist recovered, log written in both modes, `auto_replace` writes scaled `CAMERA` + backup and `False` leaves the config untouched, capture timeout handled without hanging, ROS parameters, a capture during a solve queues one more solve; with the model (GPU, checkpoints): the distance readout matches the fake camera's pose and K is recovered; OpenCV shows no readout; resume restores views, counts, solution and history, numbers new views after undone ones, undoes restored views, and refuses another board's session |
| `test/test_gitignore.py` | `calibration_output/` stays out of git |

ROS tests use an isolated domain and localhost-only discovery, like `test_node_pose_output.py`.
Manual check (fake camera, needs the X display): window size stable across modes, overlay alignment,
2 s review, counters, keys.

```bash
# inside p4p-gpu
ros2 run convchart_ros fake_camera
ros2 run convchart_ros calibrate --ros-args -p config:=/workspace/Conv-ChArT-Wireless-Inference/cfg/cfg.yaml
# or: ros2 launch convchart_ros convchart.launch.py task:=calibrate [source:=fake] [detector:=opencv]
#     [resume:=latest|<session folder>]
```
