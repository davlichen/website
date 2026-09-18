"""Local preview: run `python3 preview.py`, then open http://localhost:8000."""

import argparse
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit


ROOT = Path(__file__).resolve().parent


class PreviewHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path):
        relative = unquote(urlsplit(path).path).lstrip("/")
        candidates = [(ROOT / "html", relative)]
        for folder in ("css", "files", "js"):
            candidates.append((ROOT / folder, relative))
            candidates.append((ROOT / folder, Path(relative).name))
            if relative.startswith(folder + "/"):
                candidates.append((ROOT / folder, relative[len(folder) + 1:]))
        if relative.startswith("wiki/"):
            candidates.append((ROOT / "wiki/export", relative[5:]))
        for base, suffix in candidates:
            target = (base / suffix).resolve()
            if target.is_relative_to(base.resolve()) and target.exists():
                return str(target)
        return str(ROOT / "html" / "__preview_missing__")

    def end_headers(self):
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    with ThreadingHTTPServer(("127.0.0.1", args.port), PreviewHandler) as server:
        print(f"Preview: http://localhost:{args.port} (Ctrl+C to stop)", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
