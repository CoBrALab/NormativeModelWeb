#!/usr/bin/env bash
# Wire this repo's app and package into the vendored neurodesk/webapps checkout
# so the upstream build, theme and audit scripts (which assume
# <repoRoot>/apps/<id>, <repoRoot>/packages/<id> and a registry entry) work
# unchanged. Idempotent.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
upstream="$root/vendor/webapps"

git -C "$root" submodule update --init vendor/webapps
# Already applied when the reverse applies; otherwise apply, and fail loudly on a conflict.
if ! git -C "$upstream" apply --check -R "$root/upstream/registry.patch" 2>/dev/null; then
  git -C "$upstream" apply "$root/upstream/registry.patch"
fi
# Install upstream first: its frozen lockfile must not see our apps or packages.
for dir in "$root"/apps/*/ "$root"/packages/*/; do
  kind="$(basename "$(dirname "$dir")")"
  rm -f "$upstream/$kind/$(basename "$dir")"
done
(cd "$upstream" && pnpm install --frozen-lockfile)
for dir in "$root"/apps/*/ "$root"/packages/*/; do
  kind="$(basename "$(dirname "$dir")")"
  id="$(basename "$dir")"
  ln -s "../../../$kind/$id" "$upstream/$kind/$id"
done
(cd "$root" && pnpm install)
