# Bilingual floor-plan question router

This optional CLI proposes a room-reader operation and copies its arguments from a German or English question. The output contains the operation, arguments and weight digest. It can be inspected independently of a floor plan. The normal drawing CLI uses its rule-based compiler.

```sh
python models/room-intent-v1/suggest.py "Welche Tür verbindet R12 mit R18?"
```

Inference requires Python and NumPy. The included 77,026-byte model uses a linear classifier with character and word features. Training requires NumPy and scikit-learn. Provide a new output directory to reproduce the fit.

```sh
python models/room-intent-v1/train.py output/room-intent-retrain
```

The training file contains 100 author-written language examples, including unsupported requests. Room and door identifiers are normalized during training and copied from the original question at inference. On twelve independently authored questions the model produced 10/10 correct supported programs and rejected 1/2 unsupported questions. A question about widening a door and its supporting wall was incorrectly interpreted as a request to locate the door. Automatic fallback is disabled in the drawing reader.

The packaged CLI reproduced all twelve closed program outputs on a private CPU worker. Training from the included examples reproduced the weight archive byte for byte. [Release verification](release-check.json)

Source code uses the [project's MIT license](LICENSE-CODE.txt). The newly authored questions and fitted coefficients use the same CC BY 4.0 terms as the project's open research models, with attribution “PlanProof contributors, Bilingual Room Intent, 2026”. The training set contains no drawings or third-party model weights. [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) · [Original training receipt](fit-receipt.json)
