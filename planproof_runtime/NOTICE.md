# Bundled reader assets

English OCR uses the Tesseract project's fast English model from release 4.1.0, distributed under Apache-2.0. Its SHA-256 is `7d4322bd2a7749724879683fc3912cb542f19906c83bcc1a52132556427170b2`. The license accompanies the model in `LICENSE_TESSDATA`. [Upstream model](https://github.com/tesseract-ocr/tessdata_fast/blob/4.1.0/eng.traineddata)

Raster word verification uses DejaVu Sans distributed in Matplotlib 3.10.7. Its SHA-256 is `3fdf69cabf06049ea70a00b5919340e2ce1e6d02b0cc3c4b44fb6801bd1e0d22`. Font attribution and redistribution terms accompany the font in `LICENSE_DEJAVU`. [Upstream font](https://github.com/matplotlib/matplotlib/blob/v3.10.7/lib/matplotlib/mpl-data/fonts/ttf/DejaVuSans.ttf)

The compact FP32 detector reads the existing `eteks-door.npz` parameters and uses the released configuration, tile positions and suppression implementation. See `models/sheet-doors-v1/NOTICE.md` for detector provenance.
