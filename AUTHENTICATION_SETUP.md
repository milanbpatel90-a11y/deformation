# Authentication and user access setup

This application uses server-side role checks for the FastAPI endpoints and the browser viewer. It does not allow public registration.

## Roles

| Role | Access |
| --- | --- |
| **Owner** | One protected full-access account. Can generate models, use rim calibration, and create/manage all other accounts. |
| **Editor** | Can upload images, generate models, and use rim-detection/calibration endpoints. Cannot manage accounts. |
| **Operator** | Can upload images, generate models, and use the 3D viewer. Cannot manage accounts or use rim calibration. |
| **Viewer** | Read-only access to the 3D viewer and only their own generated GLB outputs. Cannot submit generation jobs. |

Non-owner users can download only the GLBs they generated. The Owner can access all generated GLBs.

The Owner role is created once during initial setup. The user-management API does not permit an Owner to be demoted, disabled, or replaced through the UI. Only the Owner can create, change, or deactivate other accounts.

## Local setup

1. Copy `.env.example` to `.env` in the repository root.
2. Generate a one-time setup token:

   ```powershell
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

3. Put the generated value in `.env` as `AUTH_BOOTSTRAP_TOKEN=<generated-value>`. Do not commit `.env`.
4. Start the API:

   ```powershell
   uvicorn backend.api.main:app --host 127.0.0.1 --port 8000
   ```

5. Open `http://127.0.0.1:8000/login`. The first-run screen should offer to create the Owner account. The username is prefilled as `milanbpatel90-a11y`; change it if you prefer. Set a strong password of at least 12 characters and enter the one-time setup token.
6. After setup, the Owner is signed in and redirected to the 3D workspace. Open **Manage users** in the workspace header to create Editor, Operator, and Viewer accounts.

The application creates a local `.auth_secret` automatically for development and stores user accounts in `data/auth.sqlite3`. Both are excluded from Git. Keep the database and signing secret backed up and private.

## Production deployment

Set these server-side environment variables before starting the application:

- `APP_ENV=production`
- `AUTH_SECRET_KEY`: a unique random secret with at least 32 characters. Generate with `python -c "import secrets; print(secrets.token_urlsafe(48))"`.
- `AUTH_BOOTSTRAP_TOKEN`: a separate, random one-time setup token. Remove it from the deployment environment after the Owner account has been created.
- `AUTH_COOKIE_SECURE=true`: required when the application is served over HTTPS.
- `ALLOW_UNCALIBRATED_GENERATION=false`: keep false in production; image-only scale estimates are not accepted for production GLB generation.
- `AUTH_DB_PATH`: point this to a persistent writable volume, for example `/var/data/auth.sqlite3` if the host mounts a persistent disk at `/var/data`.

**Important:** the user database must live on persistent storage. Do not use an ephemeral container filesystem for production accounts; a redeploy or restart that discards the SQLite file would discard all user accounts. Back up the database regularly. For multiple independent application instances, use a shared production database such as PostgreSQL rather than separate local SQLite files.

Use HTTPS in production. Do not expose `.env`, `.auth_secret`, or the SQLite database through static file hosting. Do not put the Owner password, signing secret, or bootstrap token in source code, GitHub Actions logs, screenshots, or client-side JavaScript.

## Session and security behavior

- Passwords are stored as salted PBKDF2-HMAC-SHA256 hashes (not plaintext).
- Sessions use a signed, HttpOnly cookie with an eight-hour lifetime.
- Logout and password changes increment the user's session version, invalidating previously issued cookies. Password changes keep only the current session active.
- Cookies use `SameSite=Strict`; production cookies also use the Secure flag.
- Authenticated state-changing requests require the CSRF token header.
- Deformation, measurement, template, output, and rim-detection endpoints enforce access on the server. Hiding a UI control is not treated as authorization.
- `GET /api/health` is a public liveness check. `GET /api/ready` verifies production-ready templates, segmentation model availability, and writable artifact storage.
- The experimental `deformation_tuning_api.py` is local-development-only, binds to loopback, and refuses to start when `APP_ENV=production`.
- The browser viewer and API should be same-origin. For a Vercel frontend with a Railway API, configure a Vercel rewrite/proxy for `/api/*` to the Railway service. The current Strict SameSite session and CSRF cookies are not designed for direct cross-origin browser calls; CORS alone does not solve that.
- There is no self-service public sign-up. The Owner creates accounts from `/admin/`.
- Users can change their own password from the workspace header after signing in.

## Main routes

| Route | Access |
| --- | --- |
| `/`, `/login` | Public login entry |
| `/viewer/` | Any signed-in user |
| `/admin/` | Owner only |
| `GET /api/auth/status` | Public setup status |
| `POST /api/auth/setup` | One-time Owner setup; requires bootstrap token |
| `POST /api/auth/login`, `POST /api/auth/logout` | Authentication |
| `GET /api/auth/me` | Any signed-in user |
| `GET/POST/PATCH /api/admin/users` | Owner only |
| `POST /api/deform` | Owner, Editor, Operator; legacy image-only route is blocked in production unless explicitly enabled |
| `POST /api/deform/multi-view` | Owner, Editor, Operator; requires 4–5 images plus physical dimensions in millimetres |
| `POST /api/deform/measurements` | Owner, Editor, Operator; requires explicit physical dimensions |
| `/api/rim-detection/*` | Owner, Editor |
| `GET /api/templates` | Any signed-in user; only validated templates are listed |
| `GET /api/output/{filename}` | Owner can access all; other signed-in users can access only their own outputs |

