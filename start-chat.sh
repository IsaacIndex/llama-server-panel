#!/bin/zsh
set -euo pipefail

SCRIPT_DIR=${0:A:h}
exec "$SCRIPT_DIR/scripts/llama_role_command.sh" exec chat --auto-tune
