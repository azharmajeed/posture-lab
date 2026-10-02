# Posture Lab

A webcam posture reminder that starts on your Ryzen 5700H laptop and can move inference
to a Coral Dev Board Mini. Intel and Apple Silicon Macs use the same CPU backend.
There is no dependency on a dedicated GPU, CUDA, or AMD graphics drivers.

## Start on this Windows laptop

The project uses its own environment; it does not change your global Python installation.
From this directory, after setup:

```powershell
.\start.ps1
```

Keep your head and both shoulders in view. Calibrate each comfortable position once:

1. Press **1**, look at the laptop, then press **C** and hold still for three seconds.
2. Press **2**, look at the left monitor, then press **C** and hold still for three seconds.
3. Press **A** to accept either saved position automatically while working.

Both profiles save to `.state/calibrations.json` and load on the next launch. Only
numeric posture metrics are saved. Profiles are separate for each camera index,
configured resolution, inference backend, and local model. **1**/**2** also let you
restrict detection to one profile; `-Profile laptop` or `-Profile left-monitor`
selects that profile at launch. The default is automatic mode.
Recalibrate after physically moving the camera or changing the remote Coral model.
**P** pauses/resumes; **Q** or Escape exits and releases the camera.

**M** cycles notification / English voice / Hindi roast. **T** tests the current
reminder immediately. **V** mutes/unmutes speech (desktop notifications remain
controlled by the `notifications` setting). Pause, recalibration, loss of tracking,
returning to your baseline, and quitting cancel pending speech.

The preview shows landmarks, inference time, current status, and the duration of a
detected change. A change must last five seconds before `ADJUST` or a reminder;
the preview shows `PENDING` while checking a brief movement. Reminders have a
30-second repeat cooldown. `next nudge Ns` shows the remaining dwell or cooldown.
**T** bypasses these timers to test playback immediately.
Notifications are best-effort OS notifications plus a reliable
on-screen banner. Keep the preview running while working; minimize it if desired.

## Report missed detections and false alarms

Click the **FN: missed bad posture** button (or press **N**) when your posture is
bad but the app misses it. Click **FP: false alarm** (or press **F**) when the app
asks you to adjust even though your posture is comfortable. The confirmation row
shows that the label was saved and the counts for this session.

Feedback is appended to `.state/feedback.jsonl`. Each report contains the displayed
prediction, selected profile, both calibrated baselines, current raw and smoothed
metrics, keypoints/confidence, thresholds, reminder timing, and six seconds of recent
numeric measurements. No camera images or video are recorded. Reports are disabled
while paused, calibrating, uncalibrated, or without fresh tracking.

The bottom row shows each measured deviation / threshold: **Side**, **Drop**,
**Tilt** (degrees), and **Closer** (relative shoulder size). A negative Drop or
Closer means the head rose or moved away. `PENDING` means a threshold was exceeded
but the five-second dwell is still running. Feedback saves examples for review;
it does not automatically change thresholds or overwrite your calibration.
You can change the log destination using `feedback.path` in your configuration.

`ADJUST` and spoken reminders have separate timing: a confirmed change can remain
`ADJUST` while the repeat cooldown runs. Muting speech also keeps detection active.
The current detector uses one threshold per metric; a brief move below all thresholds
returns to `GOOD` and resets the five-second dwell. Separate entry/recovery thresholds
(hysteresis) are a potential improvement for positions close to the boundary.

## Spoken reminders and Hindi roast mode

Install the voice extra and fetch both offline neural voice models once:

```powershell
uv sync --frozen --python 3.11 --extra local --extra voice --extra dev
uv run --frozen --extra voice posture-lab download-voices
uv run --frozen --extra voice posture-lab prepare-voices --config configs/hindi-roast.json
.\start-roast.ps1
```

The PowerShell launchers prepare the reminder WAVs before opening the camera, and
reuse them on later starts. `-Background` leaves the terminal free and records the
launcher PID for resource measurements. `-NoSync` reuses the installed uv environment.

| Mode | Behavior |
| --- | --- |
| `notification` | Desktop notification and preview banner |
| `voice` | Local English voice says to fix your posture |
| `roast` | Local Hindi voice reads a rotating Hindi roast |

The Hindi preset uses `roast_intensity: "spicy"`, including the gaali requested
for this project (for example, “अबे चूतिये, स्क्रीन में घुसने का इरादा है क्या? सीधे बैठ।” ).
Choose `"gentle"` for Hindi jokes without gaali. The text bank is in
`src/posture_lab/reminders.py`; it doesn't need an LLM or a cloud service.

The `reminders` configuration accepts `mode`, `tts_engine`, `english_model`,
`hindi_model`, `cache_directory`, `volume` (0–1), `speed` (0.5–2), `roast_intensity`, and `muted`.
CLI overrides: `--reminder voice` or `--reminder roast`, and `--tts-engine piper`.
Voice/roast modes use speech plus the preview banner by default. Set
`reminders.also_notify: true` to add desktop popups in those modes. The top-level
`notifications: false` disables desktop popups in every mode.

Piper runs locally on the laptop/Mac CPU, independent of the inference backend.
The English voice is `en_US-lessac-medium`; Hindi is `hi_IN-pratham-medium`.
All ten stock reminder phrases are cached in `.state/voice-cache` before launch,
so playback can start without loading the neural model or generating speech.
Changing a voice model, text, volume, or speed invalidates the corresponding cache.
Uncached speech renders on a separate worker with two ONNX CPU threads.
Duplicate reminders are dropped while busy; no spoken reminder queue builds up.
Missing models/audio devices report `TTS failed` in the preview; the preview banner
continues to work. The terminal logs the delay from reminder request to playback start.

Test without opening the webcam:

```powershell
uv run --frozen --extra voice posture-lab voice-test --mode voice
uv run --frozen --extra voice posture-lab voice-test --mode roast
uv run --frozen --extra voice posture-lab voice-test --mode roast --output .run/hindi-test.wav
```

`--output` writes a WAV without playing it. Cached reminder audio is read locally;
camera frames are never saved. `tts_engine: "system"` optionally uses an installed OS voice instead of
Piper; a Hindi system voice must be installed for Hindi roasts. System voices on
Mac/Linux are not yet tested here. Mac `say` manages its own output volume.

Voice code uses the [Piper Python API](https://github.com/OHF-Voice/piper1-gpl/blob/main/docs/API_PYTHON.md).
Piper is GPL-3.0; each downloaded voice's model card is saved alongside its model
in `models/voices/`. Consult those files when redistributing voices.

## Fresh installation (Windows)

Install [uv](https://docs.astral.sh/uv/), then use Python 3.11 or 3.12. Dependencies
are managed by `uv` and locked in `uv.lock`:

```powershell
uv sync --frozen --python 3.11 --extra local --extra dev
uv run --frozen --extra local posture-lab download-model
uv run --frozen --extra local posture-lab check
uv run --frozen --extra local posture-lab run --config configs/laptop.json
```

If Python discovery hits an unconfigured pyenv shim, pass the absolute path to an
installed Python with `uv sync --python PATH_TO_PYTHON --extra local --extra dev`.
The default model is Google's lightweight MediaPipe Pose Landmarker bundle. Downloads
happen only through the explicit `download-model` command. Relative model paths are
resolved from your current working directory, so run commands from this project folder.

## Mac installation

```bash
uv sync --frozen --python 3.11 --extra local
uv run --frozen --extra local posture-lab download-model
uv run --frozen --extra local posture-lab check --config configs/mac.json
uv run --frozen --extra local posture-lab run --config configs/mac.json
```

Allow camera access for the terminal/Python application in macOS privacy settings.
The Mac preset explicitly uses CPU inference. Metal/Core ML/Apple Neural Engine
acceleration is a future adapter, not implemented or required in this version.
Mac execution has not been tested on this Windows machine. Intel Macs select
MediaPipe 0.10.21 because newer releases currently ship only Apple Silicon Mac
wheels; Apple Silicon and Windows select the newer compatible release in the lockfile.
The legacy Intel Mac dependency also pins JAX/JAXlib 0.4.38, since
[JAX dropped Intel Mac wheels in 0.5](https://docs.jax.dev/en/latest/changelog.html#jax-0-5-0-jan-17-2025).

## Choose inference independently of the application

| Target | Backend | Execution location | Model |
| --- | --- | --- | --- |
| Windows AMD laptop | `cpu` | Laptop CPU | MediaPipe Pose Landmarker Lite |
| Mac (Intel / Apple Silicon) | `cpu` | Mac CPU | Same MediaPipe model |
| Coral Dev Board Mini | `coral` | Board's Edge TPU via local service | Coral MoveNet Lightning PTQ |

```powershell
uv run --frozen --extra local posture-lab run --backend cpu
uv run --frozen posture-lab run --config configs/coral.json
```

Configuration separates inference, camera, posture rules, and notifications. CLI
overrides include `--backend`, `--model`, `--camera`, `--endpoint`, and
`--no-notifications`. Copy a preset to ignored `local.json` for personal settings.
All `PostureConfig` thresholds can be overridden in the `posture` object.

There is no automatic hardware fallback: an unavailable Coral reports an error,
clears stale pose evidence, and never labels a missing frame as bad posture.
Recalibrate after changing inference backend, camera position, or seated position.

## What this first version measures

It compares your current position with your own stable calibration:

- Head movement sideways relative to shoulder width.
- A lower head position relative to the shoulders.
- A change in shoulder tilt, corrected for frame aspect ratio.
- Increased apparent shoulder size (moving closer to the camera).

Measurements are smoothed over time and ignore low-confidence or out-of-frame points.
Gaps, pauses, and inference errors reset the continuous-change timer. A changing
calibration is retried until you hold still.

These are visible 2D position changes, not a medical posture assessment. A front-facing
laptop camera cannot reliably measure spinal curvature or neck angle in depth. Moving
closer is a camera-relative proxy, not proof of slouching. Side-view analysis and
additional landmark rules can be added without changing the inference interface.

## Coral setup once the Mini powers on

The Mini is a separate Linux computer, not a USB accelerator plugged directly into
the laptop's inference process. The laptop retains webcam capture and notifications;
the board receives JPEG frames and returns only normalized body keypoints.

Follow [Coral's Mini setup guide](https://www.coral.withgoogle.com/docs/dev-board-mini/get-started/)
to get Mendel Linux and SSH access working. The guide specifies a **5 V / 2 A USB-C
power supply**. Consult its port diagram and power sequence when checking the board.

1. Download the official precompiled model on the laptop:

   ```powershell
   uv run --frozen posture-lab download-model --backend coral
   scp board/coral_server.py models/movenet_single_pose_lightning_ptq_edgetpu.tflite mendel@BOARD_IP:~/
   ```

2. On the Mini, install the board's matching runtime packages using its configured
   Coral/Mendel package repositories (do not install the laptop package on the board):

   ```bash
   sudo apt-get update
   sudo apt-get install python3-tflite-runtime libedgetpu1-std python3-numpy python3-pil
   export POSTURE_CORAL_TOKEN='your-own-long-random-token'
   python3 coral_server.py --model movenet_single_pose_lightning_ptq_edgetpu.tflite
   ```

   The worker is compatible with Python 3.7+ and uses TensorFlow Lite directly with
   `libedgetpu.so.1`. Keep the runtime packages matched to the board's OS; the current
   upstream PyCoral repository is archived. Hardware execution remains unverified
   until your board works.

3. On the laptop, start an SSH tunnel in another terminal and leave it running:

   ```powershell
   ssh -N -L 8765:127.0.0.1:8765 mendel@BOARD_IP
   ```

4. In the application terminal, set the same token and select Coral:

   ```powershell
   $env:POSTURE_CORAL_TOKEN = 'your-own-long-random-token'
   uv run --frozen posture-lab check --config configs/coral.json
   uv run --frozen posture-lab run --config configs/coral.json
   ```

The defaults bind the board service to localhost and use an encrypted SSH tunnel.
For a direct trusted-LAN connection, the worker supports `--host 0.0.0.0` and the
client supports `--endpoint http://BOARD_IP:8765`; that direct HTTP transport is
unencrypted. Do not expose it to the internet. Neither runtime saves video or frames.

The Coral model has 17 keypoints and MediaPipe has 33. The CPU adapter maps to the same
COCO-17 names. Coral inference preserves aspect ratio with letterboxing and transforms
outputs back to original image coordinates. Model confidences differ, so thresholds
may need tuning after a switch.

## Architecture and more vision projects

```text
Webcam -> frame scheduler -> PoseBackend -> normalized Pose
                                  |              |
                           CPU or Coral     PostureMonitor -> preview / nudges
```

`src/posture_lab/types.py` defines `PoseBackend.infer(frame_bgr, timestamp_ms)` and
`close()`. To add another pose runtime, implement this interface and register it in
`backends/__init__.py`. No posture or camera code needs changing.

For object detection, classification, or hand tracking, introduce task-specific
result interfaces (boxes, labels, hand landmarks) rather than forcing them into a
pose result. The CPU/remote separation is reusable; model compatibility remains
specific to each task. MediaPipe's `.task` bundle cannot simply be compiled for Coral.

The camera loop keeps one outstanding inference and drops excess frames instead
of building a queue. Pause and recalibration invalidate results already in flight.
Slow or stale results cannot drive posture alerts. Default inference is 10 FPS at
640x480; tune this after measuring your actual laptop workload.

## Development checks

Measure the running Windows app's process tree for ten seconds:

```powershell
.\measure-resources.ps1 -Seconds 10
```

The script uses `.run/posture.pid` from `start.ps1 -Background`, or accepts
`-RootProcessId PID` when launching manually. It reports CPU as a percentage of
all logical processors, plus resident RAM and private committed memory. These
are different memory counters; resident RAM can contain shared library pages.
CPU and RAM can increase briefly while generating uncached speech. Prepared audio
does not load the neural speech model in the camera process.

On this Ryzen 5700H laptop, an initial 10.8-second sample across the five app
processes measured 1.2% average / 1.4% peak total CPU and 432.5 MiB resident RAM.
Private committed memory was 985.6 MiB; this is not all resident physical RAM.
This is a sample of that running session, not a guaranteed limit for other
camera settings, TTS activity, or machines.

Test webcam access independently of inference (no camera images are saved):

```powershell
uv run --frozen posture-lab camera-check --camera 0 --seconds 3
uv run --frozen posture-lab camera-check --camera 0 --seconds 5 --preview
```

Or run `.\camera-check.ps1 -Seconds 5 -Preview`. It reports readable frames, dimensions, capture
rate, and aggregate brightness/detail to help detect a closed camera shutter. A dark
feed can pass the access check while still being unusable for pose detection.
If opening fails on Windows, try `--api msmf`, another `--camera` index, and Windows
Settings > Privacy & security > Camera > Let desktop apps access your camera.
Close the posture monitor before testing, so both processes do not compete for the camera.

```powershell
uv run --frozen --extra local --extra dev pytest -q
uv run --frozen --extra dev ruff check .
```

Tests cover dwell/cooldown behavior, missing data, calibration, coordinate transforms,
and the real JPEG/HTTP protocol using a fixture inference worker. Fixture protocol
tests do not establish that the physical Edge TPU works.

Windows verification: 49 tests passed, Ruff checks passed, and the CPU model
returned all 17 normalized keypoints on Google's sample pose image. An eight-second
webcam smoke check completed 58 inferences with 119 ms mean result latency and released
the camera cleanly. Subsequent live use detected seated posture and forward leaning.
Physical Coral inference and Mac execution remain unverified.
Both Piper voices generated valid 22,050 Hz audio; Hindi speaker playback completed
successfully on this Windows laptop. Speech tests cover cancellation, duplicate
suppression, muting, mode selection, recovery from missing audio devices, persistent
audio reuse without a neural runtime, saved profile reloads, and both screen positions.
Feedback tests cover FN/FP button hit detection, append-only label storage, matching
prediction/metric snapshots, bounded history, and rejection of stale/untracked examples.

To check an existing still image without opening the webcam:

```powershell
uv run --frozen --extra local posture-lab check --image path/to/person.jpg
```

Reference APIs: [MediaPipe Pose Landmarker](https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker/python),
[Coral TensorFlow Lite inference](https://www.coral.withgoogle.com/docs/edgetpu/tflite-python/),
[Coral MoveNet example](https://github.com/google-coral/pycoral/blob/master/examples/movenet_pose_estimation.py).
