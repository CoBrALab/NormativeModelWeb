#!/usr/bin/env bash
# Make an upstream-ready neurodesk/webapps tree with this app in it, the way
# the app will be contributed: a real apps/normative-dwi (no symlink), the
# registry patch applied, and pnpm-lock.yaml regenerated. Run the upstream
# gates there, or open the upstream pull request from it.
#
#   bash tools/export-upstream.sh <webapps-checkout>
#
# The checkout should be clean (e.g. `git -C vendor/webapps worktree add <dir>`).
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
target="${1:?usage: export-upstream.sh <webapps-checkout>}"

if ! git -C "$target" apply --check -R "$root/upstream/registry.patch" 2>/dev/null; then
  git -C "$target" apply "$root/upstream/registry.patch"
fi
for app in "$root"/apps/*/; do
  id="$(basename "$app")"
  rm -rf "$target/apps/$id"
  mkdir -p "$target/apps/$id"
  # Never export local data: the parity fixtures hold UK Biobank-derived parameters.
  (cd "$app" && tar --exclude=node_modules --exclude=dist --exclude=test-results \
    --exclude=playwright-report --exclude=test/fixtures --exclude=__pycache__ -cf - .) | tar -xf - -C "$target/apps/$id"
done
(cd "$target" && pnpm install --no-frozen-lockfile)
echo "Exported to $target. Lockfile changes: git -C $target diff --stat pnpm-lock.yaml"
