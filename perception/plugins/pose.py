"""Darvin #314 compatibility entry at the official reference module path.

Original rules/TensorRT implementation is preserved in pose_reference_upstream.py.
The task profile always selects the validated CPU action backend.
"""
from plugins.action314_pose import PosePerceptionPlugin, TOOLS
DEFAULT_OUTPUT_TOPIC = "/perception/pose"
