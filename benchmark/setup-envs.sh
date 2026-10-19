#!/usr/bin/env bash
# Create the two Python environments the benchmark needs, next to this script:
#   .venv-pcn13  PCNtoolkit 1.3.0 (Python 3.11)
#   .venv-pcn14  PCNtoolkit 1.4.0 fork + nmzarr's packages (Python 3.12)
# Uses uv when it is on PATH, otherwise python3.11 / python3.12 -m venv.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

make_env() {
  local dir=$1 python=$2 reqs=$3
  if [[ -x "${dir}/bin/python" ]]; then
    echo "${dir} exists; skipping"
    return
  fi
  if command -v uv >/dev/null; then
    uv venv --python "${python}" "${dir}"
    uv pip install --python "${dir}/bin/python" -r "${reqs}"
  else
    "python${python}" -m venv "${dir}"
    "${dir}/bin/python" -m pip install -r "${reqs}"
  fi
}

make_env "${HERE}/.venv-pcn13" 3.11 "${HERE}/requirements-pcn13.txt"
make_env "${HERE}/.venv-pcn14" 3.12 "${HERE}/requirements-pcn14.txt"
