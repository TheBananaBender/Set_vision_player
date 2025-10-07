import os, sys
import uvicorn

BACKEND_PKG = "backend"   # or "backened" — match your folder name
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

BACKEND_DIR = os.path.join(ROOT, BACKEND_PKG)

if __name__ == "__main__":
    uvicorn.run(
        f"{BACKEND_PKG}.main:app",
        host="0.0.0.0",      # <— accept 127.0.0.1 AND localhost
        port=8000,
        reload=True,
        reload_dirs=[BACKEND_DIR],
        reload_excludes=["__pycache__", "*.pyc", "node_modules", ".git", ".venv"],
    )
