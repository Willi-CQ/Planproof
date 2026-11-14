Planproof detector — validated global-view release

detector.pt: inference-only PyTorch checkpoint; optimizer removed, weights verified identical.
detector.onnx: all five output heads. wall.onnx: one-channel wall logits.
metadata.json documents normalization, tensor activation and coordinate transforms.
calibration.json is bound to detector.pt and this inference mode.
validation.json gives source-specific validation scores and cohort fingerprints.
real-input-parity.json verifies every ONNX head against PyTorch on an actual drawing.

From the installed Planproof ML environment:
planproof-ml infer --checkpoint detector.pt --calibration calibration.json --global-view --image PAGE.png --out .artifacts/inference --device cpu

Use --init-weights for a fresh training run; optimizer resume requires the original training checkpoint.
Stairs remain weak. Treat detections as evidence to verify against the original drawing.
No Stoa questions or final test data selected this model. No training images are included here.
