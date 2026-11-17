# Darvin #314 CPU action candidate

This fork uses PhanthyMotus upstream commit `7b16c433463dcb7017eb95b5bb40ef99297d3d34` and preserves its perception MCP server and ROS2 entry point. The reference module `perception/plugins/pose.py` selects the task-specific CPU backend. The former implementation remains in `pose_reference_upstream.py`.

## Environment for the submission form

Use this repository and the full immutable commit SHA returned by `git rev-parse HEAD` after the tested version is pushed:

```yaml
env:
  - name: PHANTHYMOTUS_COMMIT_ID
    value: "<full tested commit SHA>"
  - name: PHANTHYMOTUS_REPO
    value: "https://github.com/kevin6kg123/phanthymotus-dev.git"
  - name: JP_VERSION
    value: "6"
```

JP5 has not been tested. `MCP_PORT` and `WS_PORT` override the YAML defaults as requested by the leaderboard. The default task profile enables only pose/action recognition, avoiding unrelated model downloads.

## Required model asset

Weights are not stored in Git. A public Release asset must be available **before submitting**:

`https://github.com/kevin6kg123/phanthymotus-dev/releases/download/action314-cpu-v1/action314-models-v1.tar.gz`

The archive contains only `pose.onnx`, `joint.onnx`, `bone.onnx` and `runtime.json`. Its expected SHA256 is:

`a981210b62ba1efd467d0882976a8d0722cff661c0cea08f342283693e9db6d1`

`plugins/action314/asset.json` also pins every extracted file. The runtime downloads into `/models/action314-cpu-v1`, verifies the archive and each member, and reuses a verified cache. No private registry, third-party credentials, or unverified fallback models are used. `ACTION314_MODEL_DIR` may select a pre-provisioned model directory; `ACTION314_OFFLINE=1` requires that cache to be complete.

The three ONNX files total **23,208,501 bytes**, with **5,686,424 parameters**. The compressed asset is 20,709,852 bytes. Video samples and training data are not published.

## Actual interface

Start the existing `perception/main.py` in the ROS2 environment supplied by deployment. It exposes MCP JSON-RPC at `/mcp` and keeps the upstream `/file/upload` intake. An upload requires the deployment access token when one is configured; its returned `path` is then passed directly to the tool.

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "pose",
    "arguments": {
      "action": "recognize_by_video",
      "video_path": "/models/uploads/example.mp4"
    }
  }
}
```

The MCP result contains the action JSON in its standard text content. The object has `timestamp`, `count`, `persons`, and `latency_ms`; each person has `id`, `position`, `activity`, `activity_zh`, and `activity_score`. Position is the fixed `[0,0]` placeholder; metric ground-plane coordinates are not estimated. All English/Chinese labels follow the fixed 126-label registry, including `Attention`, `Rally`, `MoveInReverse` and the valid `none` class.

Compatibility calls `pose_recognize_by_photo` with `image_path=<MP4 path>` and `pose` / `recognize_by_url` with an HTTP(S) MP4 URL use the same full-video pipeline. `list_actions`, `info`, `start`, `stop`, `reset` and prefixed aliases are supported. A single image can be classified, but its repeated static pose is not equivalent to a temporal video observation.

One-shot results are also published on ROS2 `/perception/pose`. `start` with `input_topic` subscribes to JPEG `CompressedImage` and publishes action JSON on `<input_topic>/poses`. Stream state is cleared on explicit reset and timestamp discontinuities. Streaming uses only observed history; the offline full-video accuracy below does **not** establish streaming accuracy.

Corrupt or missing files return an explicit error with `count=0` and an empty persons list. Failures are not relabeled as `none` or filled with fabricated predictions. Valid no-detection clips still execute the classifier on the resulting pose sequence.

## Tests and practical limits

- All 246 fixed development MP4s through the full CPU pipeline: fixed126 Macro-F1 **0.6061854371**, supported123 Macro-F1 **0.6209704478**, zero request failures. The automatic split has not established person/scene independence; these are not official leaderboard scores. Three classes have no validation support.
- Both exported skeleton classifiers: all 246 fixed inputs passed PyTorch CPU / ONNX numerical comparison with no argmax disagreements.
- Jetson JP6-family device: the real `perception/main.py` passed initialization, exact126 label catalogue, file upload, 12 sequential video requests across three entry forms, two concurrent requests, and ROS2 one-shot/video plus camera publication. Missing/corrupt inputs were explicitly rejected. The WebSocket port completed its handshake; ASR is disabled in this action-only task profile.
- CPU-only ONNX Runtime **1.18.0** was tested in an isolated Jetson dependency directory, matching the official Dockerfile's ORT version. CPU 1.23.2 was also tested. ONNX IR8 / opset17 models are used. All sessions explicitly select `CPUExecutionProvider`.
- Earlier four-video/two-repeat CPU candidate test: 29.97 original-video frames/s, p95 3.07s per video, 0% GPU samples, zero failures. During the later complete-service tests the shared device had other GPU activity (also observed after this service stopped); those aggregate GPU samples reached99% and cannot certify an isolated resource benchmark. Official measurement scope and minimum throughput remain unconfirmed.
- No official evaluation container or Darvin run has been executed. Passing these tests does not guarantee a score or zero failures in an unseen harness. JP5 compatibility is not claimed.

Reproduce the service test on a fresh output directory with ROS2, OpenCV, NumPy, PyYAML, websockets and ONNX Runtime available:

```bash
python3 perception/tools/test_action314_service.py --samples /path/to/private/test-samples --out /path/to/fresh-output
```

The sample manifest is a local test fixture and is not part of the published repository. Unit checks are in `perception/tests/test_action314_store.py` and `test_action314_contract.py`. Startup does not use a random classifier, a network API for inference, or a synthetic fallback.
