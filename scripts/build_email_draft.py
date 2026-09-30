"""Rebuild submission/email_draft.eml (unsent, both .pck attached) from submission/email_draft.txt.

    python scripts/build_email_draft.py

The .txt has the header lines `To:`, `Subject:`, `Attachments:` followed by an empty line and the body. The .eml carries `X-Unsent: 1` so that mail clients open it as a
draft. Nothing is sent by this script. Attachment SHA-256 checksums in the body are recomputed from the files and must match the text (checked here).
"""
from __future__ import annotations

import hashlib
import re
from email.message import EmailMessage
from email.policy import SMTP
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    txt = (ROOT / "submission" / "email_draft.txt").read_text(encoding="utf-8")
    head, body = txt.split("\n\n", 1)
    hdr = dict(line.split(": ", 1) for line in head.splitlines() if ": " in line)
    files = [f.strip() for f in hdr["Attachments"].split(",")]
    for f in files:  # the checksums quoted in the body must match the attached files
        name = Path(f).name
        sha = hashlib.sha256((ROOT / f).read_bytes()).hexdigest()
        m = re.search(re.escape(name) + r" ([0-9a-f]{64})", body)
        if not m or m.group(1) != sha:
            raise SystemExit(f"SHA-256 in the email body does not match {f} (expected {sha})")
    m = EmailMessage(policy=SMTP)
    m["To"] = hdr["To"]
    m["Subject"] = hdr["Subject"]
    m["X-Unsent"] = "1"
    m.set_content(body, charset="utf-8")
    for f in files:
        m.add_attachment((ROOT / f).read_bytes(), maintype="text", subtype="plain", filename=Path(f).name)
    (ROOT / "submission" / "email_draft.eml").write_bytes(bytes(m))
    print("wrote submission/email_draft.eml with", len(files), "attachments; body checksums verified; NOT sent")


if __name__ == "__main__":
    main()
