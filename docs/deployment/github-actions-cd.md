# GitHub Actions deployment to ECS

`.github/workflows/aws-deploy.yml` starts only after the `CI` workflow completes successfully for a push to `main`. It checks out that exact CI commit, builds and pushes an ECR image tagged with the full commit SHA, and deploys a task-definition revision to the existing `license-plate-api` ECS service. The workflow preserves the service's current task definition settings and changes only the first container's image. It waits for ECS stability and verifies the running task-definition revision, container image digest, and ECS container health. It records the previous task definition and image; failed post-update checks trigger a rollback to the previous task definition and wait for stability.

## Required setup

No GitHub repository deployment variable was found. The AWS CLI session expired before IAM role/OIDC inventory could be completed, and its previously selected identity was account root. The workflow intentionally does not create IAM resources or use long-lived AWS keys. Before automatic deployment can succeed, an administrator must confirm or configure:

- Repository variable `AWS_DEPLOY_ROLE_ARN`, set to an existing least-privilege IAM role for GitHub OIDC.
- An OIDC trust relationship limited to audience `sts.amazonaws.com` and subject `repo:PrathemeshN/license-plate-detection-mlops:ref:refs/heads/main`.
- Permissions limited to pushing to ECR repository `license-plate-detector`, describing the existing ECS service/tasks and task definition, registering a new revision, updating `license-plate-api`, and passing only its existing task and execution roles.
- Optional repository variable `DEPLOYMENT_API_URL`, set to the current API base URL (scheme and host, optionally port). When configured and reachable from the GitHub-hosted runner, the workflow also probes `/health`, `/ready`, `/openapi.json`, `/api/v1/predict`, and `/api/v1/bulk-predict`. When absent, it reports the external API checks as skipped; ECS health and immutable image identity are still checked.

The current service ingress is restricted to a tester `/32`, so a GitHub-hosted runner may not be able to reach the API. Do not widen ingress to make those probes pass. Set `DEPLOYMENT_API_URL` only when the runner already has authorized network access. The service currently uses a public IPv4 address with HTTP and has no HTTPS termination; it is a restricted test deployment, not a production public endpoint.

## Rollback and verification

If the new task does not become stable, report the expected SHA image digest, or pass configured API probes, the workflow restores the prior task-definition revision and waits for the ECS service to become stable. It does not delete images or task definitions. Review the workflow run summary for the candidate commit, image tag/digest, current and prior task-definition revisions, and whether external API probes ran. CI's Docker job independently exercises health, readiness, OpenAPI, and both inference routes before the deployment workflow can start.

IAM trust and permission setup requires a least-privilege AWS administrator and is intentionally not performed by this repository workflow or deployment guide.
