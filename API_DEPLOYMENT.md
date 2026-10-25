# Deformation API quick start

The API is served by FastAPI at `backend.api.main:app`.

## Run locally

```bash
python -m pip install -r requirements.txt
uvicorn backend.api.main:app --host 0.0.0.0 --port 8000
```

Open `/docs` for the interactive OpenAPI/Swagger UI. Check `GET /api/health` for service health and `GET /api/templates` for templates detected by the runtime.

## Endpoints

- `GET /` — service and endpoint index
- `GET /api/health` — health and template readiness summary
- `GET /api/templates` — available templates
- `POST /api/deform` — front image, optional side/top images, color and template
- `POST /api/deform/multi-view` — 2–8 images; 4–5 varied product views recommended
- `POST /api/deform/measurements` — deform from manufacturer measurements
- `GET /api/output/{filename}` — download a generated GLB

## Production environment

Set `CORS_ORIGINS` to a comma-separated allowlist of exact frontend origins, for example:

```text
CORS_ORIGINS=https://your-frontend.example,https://www.your-frontend.example
MAX_UPLOAD_BYTES=15728640
```

The default upload limit is 15 MiB per image. Configure the production frontend origin before deployment; do not use a wildcard CORS origin for a public service.

## Example: multi-view upload

```bash
curl -X POST http://localhost:8000/api/deform/multi-view \
  -F "images=@front.jpg" \
  -F "images=@left.jpg" \
  -F "images=@right.jpg" \
  -F "images=@three-quarter.jpg" \
  -F "color=#363533"
```

The JSON response includes a `job_id` and a `download_url` when processing succeeds. Download the GLB from that URL.

## Production caveat

A healthy HTTP endpoint does not prove product-level 3D accuracy. Before customer release, test with independently measured real glasses, validate GLB dimensions/topology/materials, and run the complete pipeline on the deployment host. The current synchronous processing endpoint may time out for large or slow jobs; a persistent job queue and object storage are recommended before scaling traffic.
