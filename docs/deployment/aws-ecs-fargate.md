# AWS ECS Fargate Deployment Guide

> **Status note:** This guide contains observations from a previous deployment and test session. The live AWS service, running image digest, task-definition revision, task count, logs, and security-group rules have not been verified as of this documentation revision. Run the verification commands below before relying on these details for a current release.

## 1. Previously recorded deployment snapshot

Validated components:
- Private Amazon ECR repository for the container image.
- ECS cluster: `license-plate-cluster`.
- ECS service: `license-plate-api`.
- Task definition: `license-plate-detector:1`.
- Container port: `8000`.
- CloudWatch log group: `/ecs/license-plate-detector`.
- Health endpoint: `/health`.
- Readiness endpoint: `/ready`.
- Single-image inference: `POST /api/v1/predict`.
- ZIP bulk inference: `POST /api/v1/bulk-predict`.

During the earlier test session, the service reportedly reached ECS steady state with one running task. Health and readiness checks returned successful responses. Single-image inference returned HTTP 200. Bulk inference returned HTTP 200 for nine batches covering 433 images. These are historical results, not a statement of current live status.

These results validate deployment and API execution. They do not independently establish precision, recall, or model accuracy.

## 2. Architecture

1. The container image is stored in private Amazon ECR.
2. Amazon ECS schedules the API container on AWS Fargate.
3. The task uses `awsvpc` networking and a public IP for restricted testing access.
4. A security group permits TCP port 8000 only from the authorized tester's public IPv4 address using a `/32` rule.
5. Container logs are sent to CloudWatch Logs.
6. ECS and the container health check use `/health`; `/ready` is available for readiness verification.

No load balancer or HTTPS termination is configured in the current testing setup.

## 3. Resource configuration recorded during the prior deployment

| Resource | Configuration |
| --- | --- |
| Region | `ap-south-1` |
| ECR repository | `license-plate-detector` |
| Image tag used | `v3` |
| ECS cluster | `license-plate-cluster` |
| ECS service | `license-plate-api` |
| Task definition revision | `license-plate-detector:1` |
| Launch type | Fargate |
| CPU | 1024 units (1 vCPU) |
| Memory | 4096 MiB |
| Container port | 8000 |
| CloudWatch log retention | 7 days |
| Desired task count | 1 |

The earlier deployment used image tag `v3`. The currently running image digest has not been verified. For a future release, record the exact image digest and use immutable image identity for deployment and rollback.

## 4. Access and security

The previously tested service configuration was intended for restricted testing, not unrestricted production access. Verify the live security-group rules before relying on this restriction.

- Keep inbound TCP port 8000 restricted to the authorized tester's current public IP `/32`.
- Update the security-group rule when the tester's public IP changes.
- Do not expose port 8000 to `0.0.0.0/0`.
- Use a dedicated least-privilege IAM identity for routine deployment operations. Do not use AWS root credentials for routine work.
- The task execution role should have only the permissions required to pull images and publish logs.
- Do not commit credentials, local AWS profiles, access keys, or secret environment files.
- The current direct public-IP endpoint uses HTTP. Add HTTPS through an appropriate TLS-terminating architecture before production use.
- Review authentication, request limits, monitoring, and data-retention requirements before public or sensitive workloads.

## 5. Verification commands

Use the configured AWS CLI profile and region.

### Service status

```powershell
aws ecs describe-services `
  --cluster license-plate-cluster `
  --services license-plate-api `
  --profile aws-deploy `
  --region ap-south-1 `
  --query "services[0].{Status:status,Desired:desiredCount,Running:runningCount,Pending:pendingCount,Deployments:deployments[*].rolloutState}" `
  --output json
```

### Health and readiness

Replace `<PUBLIC_IP>` with the current task public IP:

```powershell
Invoke-RestMethod "http://<PUBLIC_IP>:8000/health"
Invoke-RestMethod "http://<PUBLIC_IP>:8000/ready"
```

### OpenAPI

```powershell
Invoke-RestMethod "http://<PUBLIC_IP>:8000/openapi.json"
```

### Inference routes

- `POST /api/v1/predict`: multipart upload with form field `file`.
- `POST /api/v1/bulk-predict`: multipart ZIP upload with form field `file`.

The bulk endpoint enforces a request-size limit. Split large archives into smaller batches that fit the configured application limit. Confirm that all expected images are represented in the response before treating a batch as complete.

## 6. Bulk inference validation

A testing archive containing 433 images was split into nine ZIP batches. All nine requests returned HTTP 200, and the saved responses contained 433 image results in total.

Recorded inference summary:
- Images processed: 433.
- Detection results returned: 473.
- Images with zero detections: 35.
- Images flagged for review: 31.
- Detections flagged for review: 32.
- Mean detection confidence: 71.20%.

These figures describe model outputs only. Precision, recall, false positives, false negatives, and mAP require matching predictions against ground-truth annotations.

## 7. Logs and troubleshooting

Inspect service events and task status before rebuilding the image.

```powershell
aws ecs describe-services `
  --cluster license-plate-cluster `
  --services license-plate-api `
  --profile aws-deploy `
  --region ap-south-1 `
  --output json
```

```powershell
aws ecs list-tasks `
  --cluster license-plate-cluster `
  --service-name license-plate-api `
  --profile aws-deploy `
  --region ap-south-1
```

```powershell
aws logs describe-log-streams `
  --log-group-name /ecs/license-plate-detector `
  --order-by LastEventTime `
  --descending `
  --limit 5 `
  --profile aws-deploy `
  --region ap-south-1
```

If the task fails to start, inspect ECS stopped reasons, service events, and CloudWatch logs. Avoid changing application code until logs identify an application-level problem.

## 8. CI and release workflow

The repository CI workflow runs code-quality checks, unit tests with a coverage threshold, security checks, a Docker build, health/readiness checks, OpenAPI validation, and single/bulk inference smoke tests.

For a controlled release:
1. Run the complete CI workflow and confirm it passes.
2. Review and merge changes through a pull request.
3. Build and scan the release image.
4. Push an immutable release image to ECR.
5. Register a reviewed task-definition revision.
6. Deploy the revision to the ECS service.
7. Verify service steady state, health/readiness, and real inference.
8. Record the deployed image digest and retain a known-good revision for rollback.

A successful local or CI build does not alone prove a successful AWS deployment.

## 9. Cost controls

Fargate charges accrue while tasks run. ECR storage and CloudWatch Logs can also incur charges. Review current AWS billing and service pricing for the selected region and workload.

When the API is not needed, reduce the ECS service desired count to zero or delete the service and related resources after verifying that they are no longer required. Remember that other resources, such as the ECR repository and CloudWatch log group, can continue to incur charges.

## 10. Rollback and teardown

Before a release, record the last known-good task-definition revision and image digest.

If a release fails:
1. Inspect ECS events and container logs.
2. Update the service to the previously validated task-definition revision.
3. Wait for the deployment to complete.
4. Re-test health, readiness, and inference.

Do not delete resources by guessing names. Verify each resource and its dependencies first. For a full teardown, confirm that the service is no longer required, set desired count to zero or delete the service, then remove only the dedicated resources approved for deletion. Preserve evidence and image digests needed for reporting or rollback.
