# Vercel deployment

The frozen React interface runs on Vercel. Its existing relative `/api` URLs
are forwarded over HTTPS to the persistent Planproof worker on the existing
Verda server. Gemini, Nemotron, ML, geometry and their routing code are unchanged.
The interface continues to submit a background run and poll for activity.

## Why the agent uses a persistent worker

The current API retains canonical pages in memory, starts background threads,
writes run records and invokes CUDA/geometry subprocesses. Moving that API
unchanged into independent functions would break those assumptions. It also
accepts original drawings up to 10 MiB, exceeding Vercel Functions' 4.5 MB request
limit. External rewrites let the existing API handle these requests without
putting a function between Vercel and the worker. Interactive inference runs
outside the HTTP request; short submit/poll calls avoid the external rewrite's
120-second timeout.

## Source and secrets

- `deploy/Dockerfile` layers the current agent and algorithms on the server's
  verified `planproof-inference:e761b84` CUDA image. Building elsewhere requires
  that image, or an explicitly selected compatible `INFERENCE_IMAGE` build arg.
- `deploy/compose.yaml` mounts the already backed-up model bundle read-only.
  Training datasets are not mounted or uploaded. Model inference uses the A100.
- Runtime artifacts and provider secrets live under
  `/srv/planproof-data/product/`, outside the image and source repository.
- `providers.env` contains the configured Gemini/Nemotron credentials and model.
  It is mounted read-only; it is never uploaded to Vercel or put in browser code.
- `origin.env` holds a random `PLANPROOF_ORIGIN_SECRET`. The same value is a
  sensitive Vercel production/preview environment variable. Vercel replaces the
  origin header on outbound API requests; Caddy rejects requests without it.
- This origin gate prevents direct bypass of Vercel. It is not user login or a
  multi-tenant authorization system. Do not describe the demo as such.
- API responses are never CDN-cached. TLS certificates and run records persist
  on the existing attached disk. One API worker is intentional because page
  identity and the executor currently live in process memory.

## Deployment sequence

1. Commit the intended source revision. Archive only `backend/`, `feedback/`,
   `deterministic/`, `ml/` and `deploy/` from Git into a new server release folder.
   Do not archive the private parent workspace or local `.artifacts`.
2. Place provider/origin files in the private server secrets directory with
   permissions `0600`. Create persistent artifact and Caddy data directories.
3. Set `PLANPROOF_RELEASE` to the source commit and `PLANPROOF_API_HOST` to the
   server's public DNS hostname. From that release folder, run:

   ```sh
   docker compose -f deploy/compose.yaml build api
   docker compose -f deploy/compose.yaml up -d
   ```

4. Create/link the `planproof` Vercel project. Add the origin secret via stdin
   using `vercel env add PLANPROOF_ORIGIN_SECRET --environment production,preview
   --sensitive --project planproof`; do not pass it on a command line.
5. Prepare a fresh ignored staging directory containing only committed frontend
   build inputs and generated routing configuration:

   ```sh
   python tools/stage_vercel.py --output .artifacts/vercel/source \
     --api-origin https://YOUR_PUBLIC_API_HOST
   cd .artifacts/vercel/source
   vercel link --project planproof --yes
   vercel deploy --prod --yes
   ```

   The staging tool records the exact source commit and frontend tree. It never
   uploads local drawings, test artifacts, Python environments or weights.
6. Verify HTTPS, agent configuration, original image upload, source frame,
   submission/polling, citations and report download through the Vercel URL.
   Use generated test material for deployment checks; never publish private
   evaluation drawings or claim this smoke test measures construction accuracy.

## Operations

The Vercel site needs the existing Verda instance and volume to remain running.
Stopping the laptop does not stop the cloud app. Restarting the API preserves
saved run records but drops its in-memory page cache; the current browser library
retains drawing files for re-upload. Existing localhost browser data does not
automatically appear under the new Vercel origin.

Use `docker compose -f deploy/compose.yaml ps` and container logs on the server
to inspect health. Preserve the prior release/image and mounted data for rollback.
The recording checkpoint `demo-video-2026-11-15` is unchanged; deployment adds no
visual changes. Do not overwrite that tag during later agent improvements.

Official references checked 15 November 2026:
[Vercel CLI](https://vercel.com/docs/cli),
[external rewrites and secret request headers](https://vercel.com/docs/routing/rewrites),
[function limits](https://vercel.com/docs/functions/limitations),
[external request timeout](https://vercel.com/docs/limits),
[Caddy automatic HTTPS](https://caddyserver.com/docs/automatic-https).
