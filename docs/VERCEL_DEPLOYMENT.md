# Vercel deployment

The frozen React interface runs on Vercel. Its existing relative `/api` URLs
are forwarded over HTTPS to the persistent Planproof worker on the existing
Verda server. Gemini, Nemotron, ML and geometry use the same persistent API
architecture; subsequent backend revisions are recorded below.
The interface continues to submit a background run and poll for activity.

## Qualified backend update — 15 November 2026, 05:20 UTC

The live API now uses evaluated backend `9909cb3`, image
`sha256:e1826b9c803e1ff116d641108897c7aa1fd06747a522f26a6a98717b7bb098b4`.
Source and evaluation tooling were promoted to main in `6391ca8` after the
fixed regression and both held-out comparisons passed their release gates.
See [AGENT_LAB_RESULTS.md](AGENT_LAB_RESULTS.md) for scores and remaining failures.

Only the API container was replaced. Public JavaScript and CSS hashes match the
previous deployment; all four existing saved run files are byte-identical.
The public agent endpoint returns HTTP200 and the container is healthy. No extra
provider inference was used for activation checks. Model weights, Vercel routing
and credentials are unchanged. The prior `fad04ec` image/release remains available
for rollback. Restart cleared the in-memory page cache; cached browser drawings
may need to be selected/uploaded again. Saved conversations remain on disk.

## Initial hosting deployment verified 15 November 2026

- App: **https://planproof-silk.vercel.app/**, Vercel project `willi-cq/planproof`.
- Deployment source: `fad04ec`; frontend tree
  `77ec18603f3ec800b42a5bcd556a9e412689fd5b`, unchanged from the recording tag.
- The A100 worker completed a real generated-plan ML/geometry check in 6.33s
  using the original selected checkpoint SHA256
  `c1861511904636f16433d5a0ec7b925c53a323c975a10df7e1886666a6e1bc54`.
- Through the public Vercel URL: a 6,761,151-byte generated PNG uploaded intact;
  the synthetic OFFICE question returned the printed 24 m2 in 26.48s with one
  Nemotron call and three citations. The 92,561-byte PDF report downloaded;
  browser refresh restored the conversation; desktop/mobile checks passed with
  no JavaScript errors. These are deployment checks, not accuracy benchmarks.
- An earlier synthetic fixture accidentally encoded the area unit as `m?`.
  The agent correctly abstained; that original run was retained before checking
  a new fixture spelling out `square metres`. No agent behavior was changed.
- Deployed JavaScript bytes match the local filming preview. The cloud CSS
  omits only an unused `ml-1` utility and its spacing variable, generated locally
  from a test fixture identifier; no application class or style changed.
- Direct unauthenticated requests to the API origin return 404. The Vercel API
  route succeeds with `Cache-Control: private, no-store`. Only the separate
  origin secret is configured in Vercel; provider credentials remain on Verda.

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
