#!/usr/bin/env bash
# Wire this repo's app into the vendored neurodesk/webapps checkout so the
# upstream build, theme and audit scripts (which assume <repoRoot>/apps/<id>
# and a registry entry) work unchanged. Idempotent.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
upstream="$root/vendor/webapps"

git -C "$root" submodule update --init vendor/webapps
if git -C "$upstream" apply --check "$root/upstream/registry.patch" 2>/dev/null; then
  git -C "$upstream" apply "$root/upstream/registry.patch"
fi
# Install upstream first: its frozen lockfile must not see our apps.
for app in "$root"/apps/*/; do rm -f "$upstream/apps/$(basename "$app")"; done
(cd "$upstream" && pnpm install --frozen-lockfile)
for app in "$root"/apps/*/; do
  id="$(basename "$app")"
  ln -s "../../../apps/$id" "$upstream/apps/$id"
done
(cd "$root" && pnpm install)
