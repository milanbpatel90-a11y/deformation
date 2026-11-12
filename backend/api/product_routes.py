"""Private eyewear product catalogue backed by Supabase Postgres and Storage.

All requests use the signed-in user's JWT, so Supabase RLS is the security boundary.
No service-role key is required.
"""
from __future__ import annotations
import re
import uuid
from pathlib import Path
from typing import Any
import requests
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, ConfigDict, Field
from backend.supabase_auth import SESSION_COOKIE_NAME, SUPABASE_KEY, SUPABASE_URL, get_current_user

router = APIRouter(prefix="/api/products", tags=["products"])
BUCKET = "eyewear-products"
OUTPUT_DIR = Path(__file__).resolve().parents[2] / "output"
MODEL_NAME = re.compile(r"^[a-f0-9]{12}\.glb$", re.IGNORECASE)

class ProductInput(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    glb_filename: str = Field(min_length=16, max_length=64)
    item_name: str = Field(min_length=1, max_length=160)
    price: float | None = Field(default=None, ge=0, le=100000000)
    currency: str = Field(default="INR", min_length=3, max_length=3)
    description: str | None = Field(default=None, max_length=5000)
    weight_grams: float | None = Field(default=None, ge=0, le=100000)
    size: str | None = Field(default=None, max_length=160)
    model: str | None = Field(default=None, max_length=120)
    style: str | None = Field(default=None, max_length=100)
    colour: str | None = Field(default=None, max_length=100)
    logo_url: str | None = Field(default=None, max_length=2048)
    brand: str | None = Field(default=None, max_length=160)
    frame_material: str | None = Field(default=None, max_length=100)
    status: str = Field(default="draft", pattern="^(draft|active|inactive)$")

class ProductUpdate(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    item_name: str | None = Field(default=None, min_length=1, max_length=160)
    price: float | None = Field(default=None, ge=0, le=100000000)
    currency: str | None = Field(default=None, min_length=3, max_length=3)
    description: str | None = Field(default=None, max_length=5000)
    weight_grams: float | None = Field(default=None, ge=0, le=100000)
    size: str | None = Field(default=None, max_length=160)
    model: str | None = Field(default=None, max_length=120)
    style: str | None = Field(default=None, max_length=100)
    colour: str | None = Field(default=None, max_length=100)
    logo_url: str | None = Field(default=None, max_length=2048)
    brand: str | None = Field(default=None, max_length=160)
    frame_material: str | None = Field(default=None, max_length=100)
    status: str | None = Field(default=None, pattern="^(draft|active|inactive)$")

def _token(request: Request) -> str:
    refreshed = getattr(request.state, "supabase_refreshed_session", None)
    token = (refreshed or {}).get("access_token") or request.cookies.get(SESSION_COOKIE_NAME, "")
    if not token:
        raise HTTPException(status_code=401, detail="Sign in again to manage products.")
    return token

def _headers(token: str, prefer: str | None = None) -> dict[str, str]:
    if not SUPABASE_URL or not SUPABASE_KEY:
        raise HTTPException(status_code=503, detail="Supabase is not configured.")
    h = {"apikey": SUPABASE_KEY, "Authorization": f"Bearer {token}", "Content-Type": "application/json"}
    if prefer:
        h["Prefer"] = prefer
    return h

def _check(response: requests.Response) -> None:
    if response.ok:
        return
    if response.status_code in (401, 403):
        raise HTTPException(status_code=response.status_code, detail="Supabase denied access. Check the product-table and storage RLS migration.")
    raise HTTPException(status_code=502, detail=f"Supabase request failed ({response.status_code}): {response.text[:400]}")

def _db(request: Request, method: str, path: str, json: Any = None, prefer: str | None = None):
    try:
        r = requests.request(method, f"{SUPABASE_URL}/rest/v1/{path}", headers=_headers(_token(request), prefer), json=json, timeout=30)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail="Could not reach Supabase database.") from exc
    _check(r)
    return r.json() if r.content else None

def _storage(request: Request, method: str, path: str, data: bytes | None = None):
    h = _headers(_token(request))
    h.pop("Content-Type", None)
    if method == "POST":
        h["Content-Type"] = "model/gltf-binary"
        h["x-upsert"] = "false"
    try:
        r = requests.request(method, f"{SUPABASE_URL}/storage/v1/object/{BUCKET}/{path}", headers=h, data=data, timeout=60)
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail="Could not reach private model storage.") from exc
    _check(r)
    return r

def _allowed(row: dict, user: dict) -> bool:
    return user["role"] == "owner" or str(row.get("user_id")) == str(user["id"])

def _valid_product_id(product_id: str) -> str:
    try:
        return str(uuid.UUID(product_id))
    except (ValueError, TypeError, AttributeError):
        raise HTTPException(status_code=404, detail="Product not found.")

@router.get("")
def list_products(request: Request, user: dict = Depends(get_current_user)):
    return {"products": _db(request, "GET", "eyewear_products?select=*&order=created_at.desc") or []}

@router.post("", status_code=201)
def create_product(body: ProductInput, request: Request, user: dict = Depends(get_current_user)):
    if not MODEL_NAME.fullmatch(body.glb_filename):
        raise HTTPException(status_code=400, detail="Invalid generated GLB filename.")
    source = OUTPUT_DIR / body.glb_filename
    ownership_file = OUTPUT_DIR / f"{source.stem}.ownership.json"
    if not source.is_file() or not ownership_file.is_file():
        raise HTTPException(status_code=404, detail="Generated model is missing or ownership could not be verified.")
    import json
    try:
        ownership = json.loads(ownership_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        raise HTTPException(status_code=404, detail="Model ownership could not be verified.")
    if ownership.get("user_id") != user["id"]:
        raise HTTPException(status_code=403, detail="You can only save a model generated in your own account.")
    product_id = str(uuid.uuid4())
    object_path = f"{user['id']}/{product_id}.glb"
    _storage(request, "POST", object_path, source.read_bytes())
    record = body.model_dump(exclude={"glb_filename"})
    record.update({"id": product_id, "user_id": user["id"], "owner_email": user["email"], "glb_path": object_path})
    try:
        rows = _db(request, "POST", "eyewear_products?select=*", record, "return=representation")
        if not rows:
            raise HTTPException(status_code=502, detail="Product row was not returned by Supabase.")
    except Exception:
        try:
            _storage(request, "DELETE", object_path)
        except Exception:
            pass
        raise
    return {"product": rows[0]}

@router.patch("/{product_id}")
def update_product(product_id: str, body: ProductUpdate, request: Request, user: dict = Depends(get_current_user)):
    product_id = _valid_product_id(product_id)
    changes = body.model_dump(exclude_unset=True)
    if not changes:
        raise HTTPException(status_code=400, detail="No product fields supplied.")
    rows = _db(request, "PATCH", f"eyewear_products?id=eq.{product_id}&select=*", changes, "return=representation")
    if not rows:
        raise HTTPException(status_code=404, detail="Product not found or you do not have permission to edit it.")
    return {"product": rows[0]}

@router.delete("/{product_id}", status_code=204)
def delete_product(product_id: str, request: Request, user: dict = Depends(get_current_user)):
    product_id = _valid_product_id(product_id)
    rows = _db(request, "GET", f"eyewear_products?id=eq.{product_id}&select=id,user_id,glb_path")
    if not rows or not _allowed(rows[0], user):
        raise HTTPException(status_code=404, detail="Product not found.")
    _db(request, "DELETE", f"eyewear_products?id=eq.{product_id}", prefer="return=minimal")
    _storage(request, "DELETE", rows[0]["glb_path"])
    return None

@router.get("/{product_id}/model")
def view_product_model(product_id: str, request: Request, user: dict = Depends(get_current_user)):
    product_id = _valid_product_id(product_id)
    rows = _db(request, "GET", f"eyewear_products?id=eq.{product_id}&select=id,user_id,glb_path")
    if not rows or not _allowed(rows[0], user):
        raise HTTPException(status_code=404, detail="Product not found.")
    from fastapi.responses import Response
    r = _storage(request, "GET", rows[0]["glb_path"])
    return Response(content=r.content, media_type="model/gltf-binary", headers={"Content-Disposition": "inline", "Cache-Control": "private, no-store"})

@router.get("/{product_id}/download")
def download_product_model(product_id: str, request: Request, user: dict = Depends(get_current_user)):
    product_id = _valid_product_id(product_id)
    if user["role"] != "owner":
        raise HTTPException(status_code=403, detail="Only administrators can download saved GLB files.")
    rows = _db(request, "GET", f"eyewear_products?id=eq.{product_id}&select=id,glb_path,item_name")
    if not rows:
        raise HTTPException(status_code=404, detail="Product not found.")
    from fastapi.responses import Response
    r = _storage(request, "GET", rows[0]["glb_path"])
    filename = re.sub(r"[^A-Za-z0-9._-]+", "-", rows[0]["item_name"]).strip("-") or "eyewear"
    return Response(content=r.content, media_type="model/gltf-binary", headers={"Content-Disposition": f'attachment; filename="{filename}.glb"', "Cache-Control": "private, no-store"})
