import webbrowser
import subprocess
import time
from pathlib import Path
import sys

def main():
    BASE_DIR = Path(__file__).resolve().parent
    frontend_path = BASE_DIR / "client" / "index.html"
    backend_path = "backend.entrypoint:app"

    # Start uvicorn 
    process = subprocess.Popen([
        sys.executable, "-m", "uvicorn", backend_path, "--reload", "--port", "8000"
    ])

    # Wait for server
    time.sleep(2)

    # Open browser 
    webbrowser.open(frontend_path.as_uri())

if __name__ == "__main__":
    main()

