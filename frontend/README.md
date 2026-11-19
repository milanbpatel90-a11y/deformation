# Defirmation Vercel frontend

This directory is the static browser frontend for Vercel. The FastAPI deformation API remains on Railway.

## Routing
- Browser pages: Vercel static files.
- `/api/*`: transparently rewritten to the Railway API at `https://deformation-production.up.railway.app`.
- API calls remain relative/same-origin in the browser. This keeps the existing HttpOnly session cookie and readable CSRF cookie on the Vercel host while Vercel proxies requests to Railway.

## Vercel project
Set the project Root Directory to `frontend`. No build command is required; serve the static files directly.

## Important
- Do not add Railway secrets to Vercel.
- The Railway API must remain healthy and its domain must remain active.
- Authentication/session behavior must be tested through the Vercel domain after deployment.
