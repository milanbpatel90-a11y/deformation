# Repository Audit Report — 2026-11-11

## Scope and safety

- Repository: `milanbpatel90-a11y/deformation`
- Working branch: `audit/fix-2026-11-11`
- Base: existing `main`
- `main` was not modified.
- No pull request was opened, no merge was performed, and no production deployment was requested or performed.

This report records the fixes committed to the isolated audit branch and the remaining release blockers. It is not a production-readiness certification.

## Fixes committed

### Deformation and GLB correctness

- Synchronize deformed world-space mesh copies back into the source GLB scene before export, reversing node transforms correctly.
- Reject geometry instances that share a source mesh but use different transforms.
- Reject descriptors that map Frame, Bridge, left/right rims, left/right lenses, or left/right temples to the same source mesh.
- Disable automatic vertex-count/centroid mesh guessing. Generic GLB mesh names now require explicit, reviewed `mesh_aliases`.
- Remove unreferenced alias-only geometry before GLB export to prevent duplicate overlapping render meshes.
- Convert millimetre geometry and VTO anchors to metres at export, as required by glTF 2.0.
- Validate the GLB header, declared byte length, and reloaded mesh buffers before reporting export success.
- Include coordinate units and scale provenance in GLB/sidecar metadata.
- Scale browser try-on from the actual GLB frame width and align the nose-bridge anchor with the detected face nose.

### Physical measurement and image workflow

- Add broad physical bounds and finite-number validation to eyewear measurements.
- Record whether scale came from image estimates, user-supplied dimensions, or manufacturer specifications.
- Reject uncalibrated image-only generation in production by default.
- Update the 4–5 image viewer workflow to require the five physical dimensions: frame width, lens width, lens height, bridge width, and temple length.
- Require front and side/perspective image evidence and reject empty or implausible segmentation masks.
- Keep image-derived estimates available for diagnostics, but use supplied physical dimensions as the production scale source.

### API, security, and runtime behavior

- Offload CPU-heavy reconstruction and rim-detection work from async request handlers.
- Restrict generated GLB downloads to the creator or the Owner account.
- Revoke old sessions on logout and password changes using a database-backed session version.
- Validate template names before filesystem path resolution.
- Add separate liveness and readiness endpoints; readiness checks validated templates, production segmentation model availability, and writable artifact storage.
- Restrict the experimental tuning API to local development, loopback binding, local CORS origins, and bounded uploads; it refuses to start when `APP_ENV=production`.

### Tests and documentation

- Add regression tests for mesh independence, world/local mesh synchronization, metre conversion, anchor metadata, calibration gating, mask quality, session revocation, and path traversal.
- Add an isolated-branch GitHub Actions workflow with Git LFS assets enabled.
- Add `TEMPLATE_ASSET_REQUIREMENTS.md` and update authentication/deployment guidance.

## Remaining blockers — do not release yet

1. **Production template assets are not yet certified.** Every selected GLB must be inspected and mapped to eight genuinely independent components. Add reviewed aliases for generic source mesh names. A duplicated combined mesh is not an acceptable substitute.
2. **Real six-class segmentation is not proven.** A human-reviewed six-part dataset, trained checkpoint, held-out evaluation, and real-product acceptance results are still required. Do not claim six-class accuracy based on AI prelabels or synthetic data.
3. **Physical accuracy is not proven.** Run the 4–5-view pipeline on a representative set of real products with authoritative dimensions and record measurement errors.
4. **Browser acceptance is outstanding.** Verify the exact exported GLB in the browser for geometry, material, scale, nose anchor, and face placement.
5. **Vercel checks are blocked by the team's build-rate limit.** The latest reported Vercel checks failed with the build-rate-limit response. Vercel returned zero deployment records for the audit commit; no deployment was found.
6. **Vercel/Railway origin routing must be confirmed.** The current Strict SameSite session and CSRF cookies expect same-origin browser requests. If the frontend is on Vercel and FastAPI is on Railway, use a Vercel same-origin rewrite/proxy for `/api/*`; CORS alone is insufficient.
7. **Automated test execution is not certified by this report.** The test suite and workflow were added, but a successful GitHub Actions run or local full-suite run still needs to be observed. Do not treat the existence of a workflow as a passing test result.

## Release gate

Do not merge or deploy until:

- the audit branch CI is green;
- at least one production template passes the independent-mesh and descriptor checks;
- a trained segmentation model and reviewed annotations are available;
- real-product measurement and GLB acceptance tests pass;
- browser VTO checks pass against the exact exported assets; and
- Vercel/Railway same-origin routing and persistent storage are verified.
