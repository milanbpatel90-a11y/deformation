# Supabase Google authentication and access policy

The application uses Supabase Auth with Google OAuth. FastAPI verifies each browser session against the Supabase Auth user endpoint and enforces access rules on the server.

## Access policy

| Capability | Owner (configured admin email) | Google user |
| --- | --- | --- |
| Sign in with Google | Yes | Yes |
| Upload eyewear photos | Yes | Yes |
| Generate 3D models | Yes | Yes |
| Use deformation pipeline | Yes | Yes |
| Rim detection and calibration | Yes | Yes |
| View and try on generated models | Yes | Yes |
| Load a model in the browser | Yes | Yes |
| Download GLB using the download endpoint | Yes | **No** |
| Owner-only administration | Yes | No |

Every authenticated non-owner is assigned the `editor` role, which permits the generation and rim-calibration workflows. The one exact email configured in `SUPABASE_ADMIN_EMAIL` is assigned `owner` and is the only role allowed to call `GET /api/output/{filename}`.

## 1. Configure Supabase

1. Create or select a project at https://supabase.com/dashboard.
2. Open **Authentication -> Sign In / Providers -> Google** and enable Google.
3. In Google Cloud Console, create a **Web application** OAuth client.
4. Add the app's origin (for example `https://YOUR-APP-DOMAIN`) to Google Authorized JavaScript origins.
5. Add the Supabase callback URL shown on the Google provider settings page (normally `https://YOUR_PROJECT_REF.supabase.co/auth/v1/callback`) to Google Authorized redirect URIs.
6. Paste the Google OAuth Client ID and Client Secret into Supabase's Google provider settings and save.
7. In **Authentication -> URL Configuration**, set the Site URL to the app's public origin and add these Redirect URLs:
   - `https://YOUR-APP-DOMAIN/login`
   - `http://localhost:8000/login` (local development only)

The application uses the PKCE OAuth flow and exchanges the callback code in the browser. The resulting access and refresh tokens are posted to FastAPI, validated against Supabase, and stored in HttpOnly cookies.

## 2. Configure backend environment

Set these variables in Railway (or your local `.env`):

| Variable | Value |
| --- | --- |
| `SUPABASE_URL` | Project URL from Supabase Project Settings -> API |
| `SUPABASE_PUBLISHABLE_KEY` | Supabase publishable key from Project Settings -> API |
| `SUPABASE_ADMIN_EMAIL` | The exact Google email address for the Owner/admin |
| `AUTH_COOKIE_SECURE` | `true` in production over HTTPS; `false` only for local HTTP |

A legacy `SUPABASE_ANON_KEY` can be used instead of `SUPABASE_PUBLISHABLE_KEY`. Never place a Supabase secret/service-role key in frontend code or commit it to Git.

Copy `.env.example` to `.env` for local development. Do not commit `.env`.

## 3. Server-side routes and enforcement

- `GET /api/auth/config` returns only the public Supabase URL and publishable key needed by the Google login page.
- `POST /api/auth/session` verifies the Supabase access token before issuing HttpOnly application cookies.
- `GET /api/auth/me` validates the session and returns the user's role and permissions.
- `POST /api/auth/logout` clears application session cookies.
- `GET /api/model/{filename}` serves an authenticated GLB inline for Three.js so users can view and try it on.
- `GET /api/output/{filename}` is the explicit download endpoint and is Owner-only.

The viewer hides the Download GLB control for regular users, but the FastAPI endpoint also rejects non-owner downloads. UI hiding is not the security boundary.

**Model confidentiality limitation:** browser-based 3D viewing requires the browser to receive the GLB data. A user with browser developer tools may be able to capture the inline model response. This implementation blocks the explicit download endpoint and UI control; it is not DRM and cannot make a model impossible to extract after it has been rendered in the browser.

## 4. Deployment checklist

1. Set all four environment variables in Railway.
2. Ensure Railway serves the application over HTTPS and `AUTH_COOKIE_SECURE=true`.
3. Add the production `/login` URL to Supabase's redirect allowlist.
4. Confirm the configured admin email matches the Google account used by the Owner.
5. Restart/redeploy the backend after changing environment variables.
6. Test with two accounts: the configured admin account should see Download GLB; another Google account should be able to generate and view models but receive HTTP 403 from `/api/output/{filename}`.

No database migration is required for this Supabase Auth integration; Supabase manages Google identities and sessions.
