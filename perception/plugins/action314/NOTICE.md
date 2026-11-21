# Model and source provenance

- Pose: Ultralytics YOLO11n-pose, using Ultralytics 8.3.200. The public source checkpoint SHA256 is `869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0`. Public model source: https://huggingface.co/Ultralytics/YOLO11 . Corresponding source version: https://github.com/ultralytics/ultralytics/tree/v8.3.200 . The supplied AGPL-3.0 license is retained in this directory.
- Action: adapted ST-GCN++ joint and bone classifiers, initialized from public PYSKL NTU120 COCO17 models. Source and model zoo: https://github.com/kennymckormick/pyskl . The supplied Apache-2.0 code license is retained in this directory. These are ST-GCN++ models, not PoseGraphLite.
- The task classifier heads have126 outputs. Deployment model hashes are pinned in `asset.json` and `runtime.json`. No training videos, contact sheets, dataset archive, credentials or machine environment files are distributed with the model asset.
- Parent PhanthyMotus source remains under its original license. Third-party source and model notices are retained rather than replaced with a blanket claim about all weights.

- hand landmarks: mp_palmdet.py / mp_handpose.py vendored from the OpenCV model zoo (Apache License 2.0, LICENSE-opencv-zoo); MediaPipe palm-detection and hand-landmark ONNX models from the same zoo.
