// Node side of the command line: a model store in a local folder, maps and
// z-maps as files. Scoring itself is the web app's (score.js).
import { existsSync, openAsBlob, statSync } from "node:fs";
import { mkdir, mkdtemp, readFile, rm, rmdir, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { basename, dirname, join, resolve } from "node:path";
import { guessMarker, parseMarker, zmapFileName } from "./config.js";
import { MARKERS } from "./model.js";
import { directoryStore } from "./model-store/directory-store.js";
import { openModelStore } from "./model-store/index.js";
import { cpuPredictor } from "./predictor/index.js";
import { zmapCpu } from "./predictor/kernel.js";
import { scoreMap } from "./score.js";
import { gridMapping, loadSubjectMap } from "./subject-image/index.js";
import { syntheticMap, syntheticModel, syntheticParams, writeSyntheticStore } from "./synthetic-store.js";

const notFound = () => Object.assign(new Error("not found"), { name: "NotFoundError" });

/** A DirectoryLike (the File System Access API subset the store reader uses) over a local folder. */
export function folderDirectory(path, name = basename(path)) {
  return {
    name,
    async getDirectoryHandle(child) {
      const sub = join(path, child);
      if (!statSync(sub, { throwIfNoEntry: false })?.isDirectory()) throw notFound();
      return folderDirectory(sub);
    },
    async getFileHandle(child) {
      const file = join(path, child);
      if (!statSync(file, { throwIfNoEntry: false })?.isFile()) throw notFound();
      return { getFile: () => openAsBlob(file) };
    },
  };
}

/** Open the nmzarr model store in a local folder. */
export async function openFolderStore(path) {
  if (!statSync(path, { throwIfNoEntry: false })?.isDirectory()) throw new Error(`${path}: not a folder (--store takes a model store folder)`);
  return openModelStore({ kind: "store", store: directoryStore(folderDirectory(resolve(path))), label: path });
}

/**
 * The marker of each map: from `markers` (keyed by the path as given, or its
 * file name; a marker or one of its aliases, as in file names), else from the file name.
 */
export function mapMarkers(maps, markers = {}) {
  const chosen = maps.map((path) => {
    const named = markers[path] ?? markers[basename(path)];
    if (named !== undefined && !parseMarker(named)) throw new Error(`${path}: ${named} is not a marker (${MARKERS.join(", ")}, or NDI, FWF, FISO, ODI)`);
    const marker = named !== undefined ? parseMarker(named) : guessMarker(basename(path));
    if (!marker) throw new Error(`${path}: the file name names no single marker; add --marker ${basename(path)}=<${MARKERS.join("|")}>`);
    return { path, marker };
  });
  for (const marker of MARKERS) {
    const same = chosen.filter((map) => map.marker === marker);
    if (same.length > 1) throw new Error(`Two maps for ${marker}: ${same.map((map) => map.path).join(", ")}`);
  }
  for (const key of Object.keys(markers)) {
    if (!maps.some((path) => path === key || basename(path) === key)) throw new Error(`--marker ${key}: no such map among the inputs`);
  }
  return chosen;
}

/**
 * Score each map against the store and write OUTPUT/<MARKER>_zscore.nii.gz.
 * Every input is checked (marker, store, grid, existing outputs) before any is
 * scored, and the z-maps are written only once every map is scored: a run that
 * fails leaves nothing behind. An existing file is never overwritten.
 */
export async function scoreFiles({ store: storePath, maps, markers, covariates, output, onStatus = () => {} }) {
  if (!maps.length) throw new Error("Give at least one map");
  const chosen = mapMarkers(maps, markers);
  const store = await openFolderStore(storePath);
  onStatus(`Model store ${store.label}: ${store.family}, ${store.markers.join(", ")}, ${store.grid.nVox} voxels`);
  const inputs = [];
  for (const { path, marker } of chosen) {
    if (!store.markers.includes(marker)) throw new Error(`${path}: the store has no ${marker} models`);
    const map = await loadSubjectMap(basename(path), await readFile(path));
    gridMapping(map, store.grid);
    inputs.push({ path, marker, map, out: join(output, zmapFileName(marker)) });
  }
  const existing = inputs.filter((input) => existsSync(input.out)).map((input) => input.out);
  if (existing.length) throw new Error(`Not overwriting ${existing.join(", ")}`);

  const predictor = cpuPredictor();
  const scored = [];
  for (const { path, marker, map, out } of inputs) {
    const result = await scoreMap({ store, map, marker, covariates, predictor, onStatus });
    onStatus(`${marker}: ${result.scored} voxels scored`);
    if (result.notice) onStatus(`${marker}: ${result.notice}`);
    scored.push({ input: path, marker, output: out, scored: result.scored, clamped: result.clamped, nifti: result.nifti });
  }
  await writeAll(output, scored);
  return { store: store.label, family: store.family, covariates, files: scored.map(({ nifti, ...file }) => file) };
}

/** Write every z-map or none: "wx" refuses an existing path (or a link) at write time, and a failure removes what this run wrote. */
async function writeAll(output, files) {
  const created = await mkdir(output, { recursive: true });
  const written = [];
  try {
    for (const file of files) {
      await writeFile(file.output, file.nifti, { flag: "wx" }).catch((error) => {
        throw error.code === "EEXIST" ? new Error(`Not overwriting ${file.output}`) : error;
      });
      written.push(file.output);
    }
  } catch (error) {
    await Promise.all(written.map((path) => rm(path, { force: true })));
    // The folders this run created, innermost first; rmdir leaves any that something else wrote into.
    if (created) {
      for (let dir = resolve(output); dir.startsWith(resolve(created)); dir = dirname(dir)) if (!(await rmdir(dir).then(() => true, () => false))) break;
    }
    throw error;
  }
}

/** Score a generated store and map in a temporary folder and check the written z-map against the float64 kernel. */
export async function checkInstallation() {
  const folder = await mkdtemp(join(tmpdir(), "normative-dwi-check-"));
  try {
    const model = syntheticModel({ markers: ["FA"] });
    await writeSyntheticStore(join(folder, "store"), model);
    const { bytes, y } = syntheticMap(model, "FA");
    await writeFile(join(folder, "sub-01_FA.nii"), bytes);
    const covariates = { age: 66, sex: "female" };
    const { files } = await scoreFiles({ store: join(folder, "store"), maps: [join(folder, "sub-01_FA.nii")], covariates, output: join(folder, "out") });
    const written = await loadSubjectMap("out", await readFile(files[0].output));
    const expected = zmapCpu(syntheticParams(model, "FA"), covariates, y);
    const mapping = gridMapping(written, { ...model, affine: Float64Array.from(model.affine) });
    let mismatched = 0;
    for (let v = 0; v < model.nVox; v++) if (written.data[mapping[v]] !== (expected.valid[v] ? Math.fround(expected.z[v]) : 0)) mismatched++;
    if (mismatched || !files[0].scored) throw new Error(`self-check failed: ${mismatched} of ${model.nVox} voxels differ from the float64 kernel`);
    return { ok: true, node: process.version, voxels: model.nVox, scored: files[0].scored };
  } finally {
    await rm(folder, { recursive: true, force: true });
  }
}
