# Deployment Notes

## Environments

- **staging** deploys automatically on every merge to `main`. It uses the `staging` Kubernetes namespace and the `acme-staging` cloud project.
- **production** deploys only from a tagged release (`v*`) and requires one approval from the on-call engineer.

## Rollback

Run `acme deploy rollback --env production --to <release-tag>`. Rollbacks take effect within 5 minutes. Database migrations are never rolled back automatically; check `docs/migrations.md` first.

## Configuration

- Secrets are stored in the vault and injected as environment variables at pod startup. Do not commit `.env` files.
- Feature flags are defined in the flags service; defaults live in `config/flags.yaml`.

## Monitoring

- Dashboards: latency (p50/p95), error rate, and queue depth per service.
- Alerts page the on-call engineer when error rate exceeds 2% for 5 minutes.
- Logs are retained for 30 days; traces for 14 days.
