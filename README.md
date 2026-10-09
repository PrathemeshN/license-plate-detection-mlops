# License Plate Detection MLOps Platform

This repository contains a FastAPI application for license plate detection using the tracked YOLO11n V3 model at `models/yolo11n-v3/best.pt`.

## Implemented

- FastAPI serves the browser application and inference API.
- `POST /api/v1/predict` runs inference on one uploaded image.
- `POST /api/v1/bulk-predict` runs inference on supported images in an uploaded ZIP archive.
- The browser UI includes image upload, bulk ZIP upload, and browser camera capture using `getUserMedia`. Camera frames are sent to the single-image endpoint; camera capture is handled in the browser.
- `GET /health`, `GET /ready`, and `/openapi.json` expose health, readiness, and the API schema.
- The Dockerfile packages the application, frontend, preprocessing and evaluation code, and tracked model.
- GitHub Actions runs Ruff, Black, Mypy, unit tests with a 70% coverage threshold, Bandit, pip-audit, and a Docker build/API smoke test.

The application flow is browser frontend → FastAPI routes → inference service and YOLO engine → YOLO11n V3 → API response.

## Run locally

Use Python 3.13, install `requirements.txt` and `requirements-dev.txt`, then start the app with:

```powershell
python -m uvicorn app.main:app --reload
```

The browser application is available at `http://localhost:8000/`. The model path can be overridden with `MODEL_PATH`; by default it is `models/yolo11n-v3/best.pt`.

To build the container:

```powershell
docker build -t license-plate-detector:v3 .
```

## Model and dataset

The tracked model is `models/yolo11n-v3/best.pt` (YOLO11n V3). Its training arguments are recorded in `models/yolo11n-v3/args.yaml`.

Training and evaluation dataset files are external and are not stored in this repository. The model arguments and evaluation script contain paths referring to the original dataset environment; run dataset-dependent evaluation only when the actual dataset is available at the configured paths. The independent evaluation report is in `evaluation/results/v3_independent_test.md`.

## Not implemented

Azure deployment is not implemented. No Azure resources, credentials, or deployment configuration are included. Azure deployment remains a later project stage.
