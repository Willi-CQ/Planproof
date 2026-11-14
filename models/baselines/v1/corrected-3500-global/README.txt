Planproof selected global-view detector — step 3500

detector.pt: inference-only PyTorch checkpoint, optimizer removed; parameters verified identical.
detector.onnx: all five heads. wall.onnx: one-channel wall logits.
metadata.json documents normalization, transforms, activation and hashes.
calibration.json is bound to detector.pt and GLOBAL1024 inference.
validation.json: fixed-threshold confirmation on 128 CubiCasa + 128 ResPlan validation drawings.
calibration-validation.json: smaller 32-per-source calibration confirmation, not the headline result.
selection.json and selection-lock.json: validation-only decision locked before test results.
real-input-parity.json: every ONNX head versus PyTorch on an actual drawing.
integration-check.json: actual CLI, original-coordinate JSON, overlay, probability output and input preservation checks.

planproof-ml infer --checkpoint detector.pt --calibration calibration.json --global-view --image PAGE.png --document-id drawing-123 --page-index 0 --out .artifacts/inference --device cpu
Use --device cuda on the GPU. Render PDF pages first. Keep this bundle outside Git.
Do not reuse global calibration for tiled inference.

Room/wall minimum component area is 2048 working-image pixels (long side capped at 4096).
This suppresses fragments but can remove legitimate tiny features. Review raw maps when needed.
Stairs remain weak. Room counts/categories, metric units and question answering require independent verification.
Use --init-weights for fresh training; exact optimizer resume requires the original training checkpoint.
No training images are included. test.json reports all 403 CubiCasa + 1648 ResPlan held-out drawings after the selection lock.
Test macro: wall IoU .713359, object mAP50 .586993, room-instance F1 .739116.
Stair AP50: CubiCasa .155862 (377 instances), ResPlan .007426 (79 instances).
These are project component metrics, not Stoa QA or a publisher benchmark.
