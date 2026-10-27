# Security

This is archived research code. It is not maintained as a service and has no deployment.

## Secrets

The only credential the code uses is `OPENAI_API_KEY`, read from the environment or a local
`.env` file. `.env` is ignored by git. Copy `.env.example` to start. Do not put keys in code,
notebooks or commit messages. A pre-commit hook (`gitleaks`) blocks commits that contain
something that looks like a secret.

## Known trade-offs

- The CSM, PGR and STJ crawlers disable TLS certificate verification (`ssl=False`,
  `verify=False`) because those sites' certificates did not validate at the time of the crawl.
  Only point them at the public sites they were written for.
- `torch.load(..., weights_only=False)` is used for the sharded pool cache, and pickle is used
  for the chunk caches. Only load cache files you produced yourself.

## Reporting

If you find a security problem, open an issue on the repository or contact the author through
GitHub.
