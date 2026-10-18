#!/usr/bin/env python3
"""Post a progress update (text + optional images/videos) to the project's Discord channel.

The webhook URL is read from $DISCORD_WEBHOOK or the gitignored file `.discord_webhook`
at the repo root. Stdlib only, so it runs from any env (login node, SLURM job, ...).

    python3 tools/notify.py "VAE training started" [plot.png rollout.gif ...]
    python3 tools/notify.py --title "VAE done" --color green "msg" recon.png

Never raises on network failure: a failed notification must not kill a training job.
"""

import argparse
import json
import mimetypes
import os
import sys
import urllib.request
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MAX_FILE_BYTES = 9_500_000  # Discord webhook limit is 10 MB per file
COLORS = {"blue": 0x3B82F6, "green": 0x22C55E, "red": 0xEF4444, "yellow": 0xEAB308, "grey": 0x6B7280}


def webhook_url():
    url = os.environ.get("DISCORD_WEBHOOK")
    if not url:
        f = REPO / ".discord_webhook"
        if f.exists():
            url = f.read_text().strip()
    return url or None


def post(text, files=(), title=None, color="blue"):
    url = webhook_url()
    if not url:
        print("[notify] no webhook configured; skipping", file=sys.stderr)
        return False

    host = os.uname().nodename.split(".")[0]
    job = os.environ.get("SLURM_JOB_ID")
    footer = host + (f" · job {job}" if job else "")
    embed = {"description": text[:4000], "color": COLORS.get(color, COLORS["blue"]),
             "footer": {"text": footer}}
    if title:
        embed["title"] = title[:256]

    attach, skipped = [], []
    for p in map(Path, files):
        if not p.is_file():
            skipped.append(f"{p.name} (missing)")
        elif p.stat().st_size > MAX_FILE_BYTES:
            skipped.append(f"{p.name} ({p.stat().st_size / 1e6:.1f} MB > limit)")
        else:
            attach.append(p)
    if skipped:
        embed["description"] += "\n_skipped attachments: " + ", ".join(skipped) + "_"
    for p in attach:  # show the first image inline
        if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".gif", ".webp"):
            embed["image"] = {"url": f"attachment://{p.name}"}
            break

    payload = {"embeds": [embed], "username": "OccWorld bot"}
    boundary = uuid.uuid4().hex
    body = bytearray()

    def part(headers, data):
        body.extend(f"--{boundary}\r\n{headers}\r\n\r\n".encode())
        body.extend(data)
        body.extend(b"\r\n")

    part('Content-Disposition: form-data; name="payload_json"\r\nContent-Type: application/json',
         json.dumps(payload).encode())
    for i, p in enumerate(attach):
        ctype = mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        part(f'Content-Disposition: form-data; name="files[{i}]"; filename="{p.name}"\r\n'
             f"Content-Type: {ctype}", p.read_bytes())
    body.extend(f"--{boundary}--\r\n".encode())

    req = urllib.request.Request(
        url, data=bytes(body), method="POST",
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}",
                 "User-Agent": "occwm-notify/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            ok = 200 <= r.status < 300
    except Exception as e:  # noqa: BLE001
        print(f"[notify] post failed: {e}", file=sys.stderr)
        return False
    print(f"[notify] posted ({len(attach)} attachment(s))")
    return ok


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("text")
    ap.add_argument("files", nargs="*")
    ap.add_argument("--title")
    ap.add_argument("--color", default="blue", choices=sorted(COLORS))
    a = ap.parse_args()
    post(a.text, a.files, a.title, a.color)
