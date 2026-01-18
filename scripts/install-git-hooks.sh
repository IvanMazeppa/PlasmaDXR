#!/usr/bin/env bash
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"

if [ ! -d ".githooks" ]; then
  echo "❌ .githooks/ not found"
  exit 1
fi

chmod +x .githooks/pre-commit

# Enable repo-local hooks
git config core.hooksPath .githooks

echo "✅ Enabled repo hooks via: git config core.hooksPath .githooks"
echo "✅ pre-commit hook installed: .githooks/pre-commit"
echo ""
echo "To disable: git config --unset core.hooksPath"

