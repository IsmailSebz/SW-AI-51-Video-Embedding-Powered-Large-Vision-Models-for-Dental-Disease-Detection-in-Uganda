## Running backend (uvicorn) and frontend (Streamlit) together with Docker

This repository contains a Dockerfile and a docker-compose configuration to run both the FastAPI backend and the Streamlit frontend in the same container. The container starts:

- FastAPI via `uvicorn` on port 8000
- Streamlit on port 8501

Build the image (from repo root):

```bash
docker build -t dental-app:latest .
```

Run with Docker:

```bash
docker run --rm -p 8000:8000 -p 8501:8501 dental-app:latest
```

Or using docker-compose (recommended for local development):

```bash
docker-compose up --build
```

Notes and caveats:

- The image installs some common system libraries required by OpenCV and related packages (libgl1, ffmpeg, etc.). If `pip install -r requirements.txt` fails on some native dependency, check the build logs and add the relevant `apt-get` packages.
- Note: `torch`/`torchvision` and other ML packages are large and may require substantial disk/memory during image build. On macOS Docker Desktop you may need to increase the VM memory (Preferences → Resources → Memory) to 6–8 GB and retry the build. Alternatively, build on a machine/CI runner with more memory or install platform-specific prebuilt wheels (see PyTorch installation docs).
- The container mounts the repository into `/app` when using `docker-compose` (see `docker-compose.yml`).
- The `start.sh` script is the entrypoint and starts both services. It will forward SIGTERM to child processes (so `docker stop` works).
- If you want to serve a production React build (if you use one), build it separately and serve the static files with nginx or integrate into the backend; this Dockerfile does not build Node/React assets.

Verification:

- Backend: visit http://localhost:8000/docs to see FastAPI docs (if enabled)
- Frontend: visit http://localhost:8501 to see the Streamlit UI
