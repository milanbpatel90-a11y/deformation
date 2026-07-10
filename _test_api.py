"""Send a real multipart POST to /api/deform and print the full response."""
import urllib.request
import urllib.error

img_path = "output/test_front.jpg"
with open(img_path, "rb") as f:
    img_bytes = f.read()

boundary = b"----FormBoundary1234567890"
body = b""
body += b"--" + boundary + b"\r\n"
body += b'Content-Disposition: form-data; name="front"; filename="front.jpg"\r\n'
body += b"Content-Type: image/jpeg\r\n\r\n"
body += img_bytes + b"\r\n"
body += b"--" + boundary + b"--\r\n"

req = urllib.request.Request(
    "http://localhost:8000/api/deform",
    data=body,
    headers={"Content-Type": "multipart/form-data; boundary=----FormBoundary1234567890"},
)
try:
    resp = urllib.request.urlopen(req, timeout=30)
    print("SUCCESS:", resp.read().decode()[:1000])
except urllib.error.HTTPError as e:
    err_body = e.read().decode()
    print(f"HTTP {e.code}:")
    print(err_body[:4000])
except Exception as e:
    print("ERROR:", e)
