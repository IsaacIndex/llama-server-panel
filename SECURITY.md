# Security

## Supported Versions

Security fixes are expected to land on `main` until the project adopts versioned release branches.

## Reporting

No public reporting address has been selected yet. Before publishing the repository, replace this section with the preferred security contact or GitHub private vulnerability reporting instructions.

## Operational Notes

- Do not commit local overrides such as `env.local.*`.
- Do not commit model files, logs, benchmark output, or packaged release output.
- The service gateway does not enforce authentication. Bind it to `127.0.0.1` unless the network is trusted and private.
- Treat external API keys and local model paths as sensitive machine-local configuration.
