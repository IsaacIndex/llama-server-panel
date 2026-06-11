# Security

## Supported Versions

Security fixes are expected to land on `main` until the project adopts versioned release branches.

## Reporting

Report security issues by emailing `isaacwongnh@gmail.com`.

If GitHub private vulnerability reporting is enabled for this repository, use that channel for sensitive reports.

## Operational Notes

- Do not commit local overrides such as `env.local.*`.
- Do not commit model files, logs, benchmark output, or packaged release output.
- The service gateway does not enforce authentication. Bind it to `127.0.0.1` unless the network is trusted and private.
- Treat external API keys and local model paths as sensitive machine-local configuration.
