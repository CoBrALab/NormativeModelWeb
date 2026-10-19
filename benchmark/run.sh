#!/usr/bin/env bash
# Reproduce the benchmark: synthetic models -> archives -> nmzarr store -> timings, sizes, summary.
#
# Environment (defaults in brackets):
#   PCN13_PYTHON   Python with PCNtoolkit 1.3.0         [benchmark/.venv-pcn13/bin/python]
#   PCN14_PYTHON   Python with the 1.4.0 fork + zarr    [benchmark/.venv-pcn14/bin/python]
#   NMZARR_DIR     proc_nm's opt_inference/ (the converter)   [../proc_nm/opt_inference next to this repo]
#   NODE           Node >= 22.12                         [node]
#   WORK           archives and store, ~6 GB            [benchmark/work]
#   RESULTS        result JSONs and summary.md          [benchmark/results]
#   N_VOX, PROTOTYPES, SAMPLE_PER_CHUNK, SUBJECTS   sizes (see make_models.py, bench_*.py)
#
# Steps whose output exists are skipped; delete WORK to rebuild the models.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "${HERE}")"
PCN13_PYTHON=${PCN13_PYTHON:-${HERE}/.venv-pcn13/bin/python}
PCN14_PYTHON=${PCN14_PYTHON:-${HERE}/.venv-pcn14/bin/python}
NMZARR_DIR=${NMZARR_DIR:-$(dirname "${REPO}")/proc_nm/opt_inference}
NODE=${NODE:-node}
WORK=${WORK:-${HERE}/work}
RESULTS=${RESULTS:-${HERE}/results}
N_VOX=${N_VOX:-304327}
PROTOTYPES=${PROTOTYPES:-512}
SAMPLE_PER_CHUNK=${SAMPLE_PER_CHUNK:-20}
SUBJECTS=${SUBJECTS:-1,100}

for exe in "${PCN13_PYTHON}" "${PCN14_PYTHON}"; do
  [[ -x "${exe}" ]] || { echo "Error: ${exe} not found; run benchmark/setup-envs.sh or set PCN13_PYTHON/PCN14_PYTHON" >&2; exit 1; }
done
[[ -f "${NMZARR_DIR}/nmzarr/convert.py" ]] || { echo "Error: no nmzarr converter in ${NMZARR_DIR}; set NMZARR_DIR" >&2; exit 1; }
"${NODE}" -e 'const [a,b]=process.versions.node.split(".").map(Number); process.exit(a>22||(a===22&&b>=12)?0:1)' \
  || { echo "Error: ${NODE} must be Node >= 22.12 (it runs the app's TypeScript directly)" >&2; exit 1; }
[[ -d "${REPO}/apps/normative-dwi/node_modules/zarrita" ]] \
  || { echo "Error: the app's dependencies are not installed; run tools/link-upstream.sh first" >&2; exit 1; }

# One core everywhere: the browser kernel is single-threaded, and so is n01's per-voxel loop.
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=1
mkdir -p "${WORK}" "${RESULTS}"

for tk in pcn13 pcn14; do
  py=PCN${tk#pcn}_PYTHON
  if [[ ! -f "${WORK}/${tk}/manifest.json" ]]; then
    rm -rf "${WORK:?}/${tk}"
    echo "== models (${tk})"
    "${!py}" -I "${HERE}/make_models.py" --out "${WORK}/${tk}" --n-vox "${N_VOX}" \
      --prototypes "${PROTOTYPES}" --sample-per-chunk "${SAMPLE_PER_CHUNK}"
  fi
  cp "${WORK}/${tk}/manifest.json" "${RESULTS}/manifest-${tk}.json"
done

if [[ ! -f "${WORK}/store.zarr/zarr.json" ]]; then
  rm -rf "${WORK}/store.zarr"
  echo "== nmzarr convert"
  (cd "${NMZARR_DIR}" && "${PCN14_PYTHON}" -m nmzarr convert --models-dir "${WORK}/pcn14" \
    --mask "${WORK}/pcn14/mask.nii.gz" --family synthetic_blr_agesex --out "${WORK}/store.zarr")
fi

echo "== storage"
python3 -I "${HERE}/measure_storage.py" --archives "pcn13=${WORK}/pcn13" --archives "pcn14=${WORK}/pcn14" \
  --store "${WORK}/store.zarr" --out "${RESULTS}/storage.json"

echo "== new: nmzarr store + app kernel"
"${NODE}" "${HERE}/bench_new.mjs" --store "${WORK}/store.zarr" --manifest "${WORK}/pcn14/manifest.json" \
  --subjects "${SUBJECTS}" --out "${RESULTS}/new.json"

for tk in pcn13 pcn14; do
  py=PCN${tk#pcn}_PYTHON
  echo "== old: archives + PCNtoolkit (${tk})"
  "${!py}" -I "${HERE}/bench_old.py" --models "${WORK}/${tk}" --subjects "${SUBJECTS}" --out "${RESULTS}/old-${tk}.json"
done

python3 -I - "${RESULTS}/environment.json" "${REPO}" "${NMZARR_DIR}" "${NODE}" <<'PY'
import datetime, json, os, platform, subprocess, sys
out, repo, nmzarr, node = sys.argv[1:]
def run(*cmd):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None
cpu = next((l.split(":", 1)[1].strip() for l in open("/proc/cpuinfo") if l.startswith("model name")), platform.processor())
json.dump({
    "date": datetime.date.today().isoformat(),
    "host": platform.node(),
    "cpu": cpu,
    "cores": os.cpu_count(),
    "load_average": os.getloadavg(),
    "os": f"{platform.system()} {platform.release()}",
    "node": run(node, "--version"),
    "repo_commit": run("git", "-C", repo, "rev-parse", "HEAD"),
    "nmzarr_commit": run("git", "-C", nmzarr, "rev-parse", "HEAD"),
}, open(out, "w"), indent=1)
PY

python3 -I "${HERE}/report.py" --results "${RESULTS}"
