# PlanProof MVP

One PDF page and one question: **Does this exact reference appear on the page?** The API returns the text match and a link to the rendered source page. PDF text confirms the presence of a reference. It does not establish that the object has a particular function or that the drawing meets a code requirement.

## Run locally

From the repo root with Python 3.11 or newer:

```powershell
python -m pip install -r backend/requirements.txt
python -m uvicorn app:app --app-dir backend
```

Open http://127.0.0.1:8000/docs and run `POST /api/demo`. Copy the returned `document_id`, then call `POST /api/analyze`:

```json
{"document_id":"<id from /api/demo>","label":"FD-12"}
```

Expect `supported`, a quote, and `evidence.image_url` for page 1. Retry with `FD-1` to exercise `insufficient_evidence`. The synthetic page also contains `FD-120`, so the demo exercises exact matching. The demo runs locally without credentials.

For a real sample, send `file` and optional `page` as multipart fields to `POST /api/documents`. Pages are numbered from 1. The API keeps up to eight rendered pages in memory for this local prototype. Restarting the server clears them. PDFs with no searchable text can be previewed, but the API cannot confirm a reference from the text.

## Integration for the four of us

| Owner | Next integration |
| --- | --- |
| Norlan | Call the endpoints from the viewer. Show the returned answer and status, open `evidence.image_url` relative to the API, and label the demo as synthetic. |
| Johan | Check the exact match on an authorized STOA sheet, then connect Willi's results to source validation once the output format is known. |
| Willi | Provide one reproducible inference command, one permitted sample and the raw result. Confirm whether the model reads text, detects symbols or only produces embeddings. |
| Imad | Confirm which sheets can be sent to Gemini, set the model and key, and verify one live result against the page. |

## Gemini comparison

Set `GEMINI_API_KEY` and `GEMINI_MODEL` in the server process. Send `"use_gemini":true` with the analysis request to send the selected page image to Gemini. Use the synthetic page or a sheet authorized for external processing. Gemini is asked whether it can read the exact same label. An unconfirmed Gemini match gives `disputed` and no evidence link. A local text match remains visible when Gemini is unavailable. The response reports `willi_ml: "not_connected"` until the actual model is integrated.
