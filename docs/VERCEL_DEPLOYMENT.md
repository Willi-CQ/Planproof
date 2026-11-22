# Vercel deployment

[PlanProof](https://planproof-silk.vercel.app/) hosts the web app and the CPU
floor-plan reader in the same Vercel project, `willi-cq/planproof`. The
production request handler is `api/reader.py`. It reads the selected page,
answers the question and returns evidence coordinates. It uses bundled models
and runs without a GPU, a separate inference server or provider API keys.

## Publish from a committed revision

Run the commands from the repository root. Choose an empty staging directory
for each release. The packager reads application files and dependencies from
Git at the requested revision.

```sh
python tools/stage_vercel.py --revision HEAD --output .artifacts/vercel/release
cd .artifacts/vercel/release
vercel link --project planproof --scope willi-cq --yes
vercel deploy --prod --skip-domain --scope willi-cq --yes
```

Check the deployment URL returned by the CLI. Once the candidate passes,
run `vercel promote <deployment-url> --scope willi-cq`. Promotion assigns
the public production domain to that exact build. Use a verified CPU-reader
release for rollbacks.

## Verify the reader

1. Open the web app. Select a public example or open a PDF, PNG or JPEG.
2. Choose the intended page and ask a question. Select an evidence reference
   to inspect where the answer comes from on the page.
3. An API client can request `GET /api/reader` to check availability. Send
   `POST /api/reader` with multipart fields `file`, `page` (one-based) and
   `question`. Compare `source_sha256` with the actual submitted bytes.
4. Exercise an unsupported question as well as a readable one. The reader
   reports `answered`, `partial` or `unresolved`. The HTTP status describes
   the request. The reader status describes the evidence for its answer.

Files sent directly to the function are limited to 4,000,000 bytes, image
inputs to 16 megapixels and questions to 2,000 characters. A browser may send
a single page extracted from a larger PDF. Hashes identify the submitted file.
For coordinate conventions, supported questions and interpretation see the
[reader guide](../tools/free_agent_room_qa/README.md).

The [November 15 GPU deployment notes](https://github.com/Willi-CQ/Planproof/blob/430b9c92b74ef7bbf2145878a4b4c1ae83daa44a/docs/VERCEL_DEPLOYMENT.md)
document the historical Junction system.
