Planproof CAD object specialist ? step 3000

Supported predictions: door, window and stairs. The released PyTorch checkpoint declares this scope;
the Planproof CLI automatically filters JSON and overlays to these classes.
detector.onnx emits only centers (4 channels; channel0 reserved, use1-3) and boxes (12 channels).
No wall.onnx or segmentation tensors are exported. The main corrected-3500-global release provides structural/room evidence.

planproof-ml infer --checkpoint detector.pt --calibration calibration.json --global-view --image PAGE.png --out .artifacts/cad-inference --device cpu

calibration.json binds a fixed validated profile, not calibrated correctness probabilities.
validation.json reports128CAD validation drawings. selection.json compares1500/3000updates.
calibration-validation.json records the fixed profile on the full mixed-source manifest.
real-input-parity.json verifies every published ONNX output on a realCADdrawing at the unchanged strict tolerance.
integration-check.json verifies actualCLI object-only JSON, overlay, original coordinates and input preservation.

The earlier full-head CAD ONNX export was rejected for numerical differences in unsupported segmentation heads.
This artifact publishes the validated center/box task only. Main model artifacts and held-out results are unchanged.
No Stoa package/questions or publisherCADtestfamilies selected these weights. No dataset images are included.
Keep models outside Git. Stairs remain weak; verify detections against the original drawing.
