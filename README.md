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

Use Python 3.13. Install the pinned runtime and development requirements, then start the app with:

```powershell
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m uvicorn app.main:app --reload
```

The browser application is available at `http://localhost:8000/`. The model path can be overridden with `MODEL_PATH`; by default it is `models/yolo11n-v3/best.pt`.

To build the container:

```powershell
docker build -t license-plate-detector:v3 .
```

## Model and dataset

The tracked model is `models/yolo11n-v3/best.pt` (YOLO11n V3). Its training arguments are recorded in `models/yolo11n-v3/args.yaml`.

### Reproduce the holdout evaluation

The evaluation script expects a YOLO dataset at `testing_dataset/License-Plate-Data/test/` with paired `images/` and `labels/` directories. Images may be JPG, JPEG, PNG, or WEBP; each image must have a same-basename YOLO text label file. Labels use `class_id x_center y_center width height` normalized to the image dimensions. The current local workspace was verified to contain 87 test images and 87 label files. The model file exists at `models/yolo11n-v3/best.pt`. The dataset's original source, license, and acquisition history are not recorded here, so this local copy's provenance is unknown and it is not included in Git.

Create the Python 3.13 environment using `requirements.txt` and `requirements-dev.txt`, then run:

```powershell
python evaluation/evaluate_v3.py --output-dir evaluation/results
```

Settings in `evaluation/evaluate_v3.py`: model path `models/yolo11n-v3/best.pt`, confidence threshold `0.38`, prediction NMS IoU `0.50`, image size `640`, and one-to-one ground-truth match IoU `0.50`. It writes `v3_holdout_metrics.json` and `v3_per_image.json` under the selected output directory. Pass another `--output-dir` to preserve an earlier result. Reported precision, recall, and F1 are the script's box-matching metrics; average confidence is a prediction score, not an accuracy metric. A successful run is required before treating a metrics file as current.

The candidate workspace evaluation completed against that local split with Ultralytics 8.4.173 and the tracked weights (SHA-256 `19e423f2deb7e4952c6cec32b5ae667a6748055fccb539ba649175e4ed0e4ba6`). It counted 87 ground-truth boxes and 92 predictions: 75 true positives, 17 false positives, and 12 false negatives; precision 0.8152, recall 0.8621, F1 0.8380; mean inference 153.58 ms/image. This run's JSON outputs were written to a temporary directory. A temporary SHA-256 manifest covered 174 image and label files; the manifest SHA-256 was `2425a911a2df217f0b841dc2a74e1acdc892c757561d721a2c54c02c7f93b391`. `constraints-runtime.txt` pins the transitive runtime set observed in the successful Docker CI build; direct runtime and development dependencies are pinned in their respective requirement files. The dataset files remain ignored local data and are not committed; no source/license provenance was available to verify.

## AWS deployment

AWS ECS with Fargate is the intended deployment platform. The repository contains an AWS deployment guide at [AWS ECS Fargate Deployment Guide](docs/deployment/aws-ecs-fargate.md).

Automated deployment behavior and required GitHub OIDC setup are documented in the [GitHub Actions ECS deployment guide](docs/deployment/github-actions-cd.md).

The guide records a previous deployment and test run, but current live AWS service health, the running image digest, task-definition revision, and security-group configuration must be checked against AWS before claiming a release is currently active or up to date. Follow the guide's verification steps before a new deployment. The current documented testing architecture uses a public task IP restricted to an authorized tester IP and HTTP without TLS termination; it is not production-ready for unrestricted or sensitive traffic.

## CI and release process

Automated CI is defined in `.github/workflows/ci.yml`. It runs code-quality checks, unit tests with a 70% coverage threshold, security checks, a Docker build, health/readiness checks, OpenAPI validation, and single/bulk inference smoke tests.

Merge release changes through a pull request and confirm checks for the exact commit. A green CI run validates the tested commit and container workflow; it does not by itself prove that the live AWS service has been updated or that production security controls are complete.

## Current limitations

- Live AWS deployment status and deployed image identity need verification before each release.
- The documented direct-public-IP test setup does not provide HTTPS termination or application authentication.
- Browser camera behavior has not been manually verified on physical hardware.
- Training and evaluation datasets are external and unavailable in this repository; dataset-dependent evaluation requires those datasets at the expected paths.
- Precision, recall, false positives, false negatives, and mAP must be established using annotated ground-truth data. Inference counts and confidence scores alone are not model-accuracy measurements.
