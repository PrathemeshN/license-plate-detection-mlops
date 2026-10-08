# YOLO11n V3 Independent Test Evaluation

## Dataset

- Dataset: Testing dataset yolo
- Test images: 87
- Test annotations: 87
- Background images: 0
- Corrupt images: 0
- Model: YOLO11n V3
- Confidence threshold: 0.38
- IoU threshold: 0.50

## Results

| Metric | Result |
|---|---:|
| Precision | 0.8670 |
| Recall | 0.8391 |
| mAP@50 | 0.7984 |
| mAP@50-95 | 0.3428 |
| Preprocess | 9.2 ms/image |
| Inference | 190.2 ms/image |
| Postprocess | 6.8 ms/image |

## Environment

- Container: license-plate-detector:v3
- Runtime: CPU
- Ultralytics: 8.4.174
- Python: 3.13.16
- PyTorch: 2.14.1+cpu

## API Verification

- GET /health: PASS
- GET /ready: PASS
- POST /api/v1/predict: PASS
