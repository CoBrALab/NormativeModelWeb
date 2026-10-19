// Normative DWI: maps in, z-maps out. The interface comes from the shared
// workspace vocabulary; the science is in model-store/, predictor/ and
// subject-image/ (see docs/GLOSSARY.md and docs/adr/).
import "@neurodesk/webapp-components/styles/imaging-workspace.css";
import { mountImagingWorkspace } from "@neurodesk/webapp-components/core/mount-imaging-workspace";
import {
  bindFileDrop,
  bindInfoTooltips,
  createConsole,
  createExampleSelector,
  createInfoDialog,
  createResultList,
  createViewerToolbar,
  ProgressManager,
} from "@neurodesk/webapp-components/ui";
import { downloadFile } from "@neurodesk/webapp-components/file-io/download";
import { registerAppAutomation } from "@neurodesk/webapp-components/automation";
import { APP, DEFAULT_STORE_URL, Z_DISPLAY_LIMIT, clampingNotice, guessMarker, isMarker, parseCovariates, trainingRange, zmapFileName } from "./config.ts";
import { MARKERS, type Covariates, type Marker } from "./model.ts";
import { openModelStore, type ModelStore, type StoreSource } from "./model-store/index.ts";
import { filesDirectory } from "./model-store/directory-store.ts";
import { clearModelCache, pruneOldCaches } from "./model-store/persistent-cache.ts";
import { createPredictor, type Predictor, type PredictorKind } from "./predictor/index.ts";
import { displayNifti, fieldOfViewOverlap, gatherToStore, gridMapping, loadSubjectMap, readBackground, scatterToNifti, voxelAtMm, type BackgroundImage, type SubjectMap } from "./subject-image/index.ts";
import { createZViewer } from "./viewer.ts";
import examples from "../examples.json";

const $ = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;

mountImagingWorkspace({
  controls: "#controls",
  viewer: "#viewer",
  status: "#status",
  title: APP.title,
  subtitle: "Voxelwise normative z-scores for diffusion microstructure",
  mark: "N",
  controlsContract: { about: "#aboutBtn", privacy: "#privacyBtn" },
});

const viewer = createZViewer("gl1", (text) => ($("location").textContent = text));
const toolbar = createViewerToolbar({
  window: false, overlay: false, colormap: false, download: false, screenshot: false,
  views: (["multiplanar", "axial", "coronal", "sagittal"] as const).map((id) => ({
    id,
    label: id === "multiplanar" ? "3-Plane" : id[0].toUpperCase() + id.slice(1),
    active: id === "multiplanar",
    onClick: () => { viewer.setView(id); toolbar.setActive(id); },
  })),
});
$("viewer").prepend(toolbar);
const log = createConsole({ id: "technicalLog", resizable: true });
$("viewer").append(log);
const progress = new ProgressManager();
bindInfoTooltips(document);
const info = createInfoDialog();
$("aboutBtn").onclick = () => info.open(`About ${APP.title}`, $("aboutContent"));
$("privacyBtn").onclick = () => info.open("Privacy", $("privacyContent"));

interface InputMap {
  file: File;
  map: SubjectMap;
  marker: Marker | null;
  /** Why the map cannot be scored against the open store, if it cannot. */
  problem?: string;
}
interface ZmapOutput {
  [key: string]: unknown;
  description: string;
  file: File;
  display: File;
  /** Display copy for drawing over a background (unscored voxels 0), and its |z| (the opacity). */
  overlay: File;
  magnitude: File;
  /** The map's header: the grid the z-map is on. */
  header: SubjectMap["header"];
  /** True z on the map's grid (NaN where not scored), for the viewer's readout. */
  values: Float32Array;
  /** World position (mm) → the map voxel there (index into `values`, centre in mm), or null. */
  locate: ReturnType<typeof voxelAtMm>;
}

let inputs: InputMap[] = [];
let outputs: Record<string, ZmapOutput> = {};
let store: ModelStore | null = null;
let storeSource: StoreSource | null = DEFAULT_STORE_URL ? { kind: "url", url: DEFAULT_STORE_URL } : null;
let running: AbortController | null = null;
let background: (BackgroundImage & { file: File }) | null = null;
let shownMarker: string | null = null;

function status(message: string, error = false) {
  $("statusText").textContent = message;
  $("statusText").classList.toggle("error", error);
  log.log(message, error ? "error" : "info");
}

const uiCovariates = () => parseCovariates(($("age") as HTMLInputElement).value, ($("sex") as HTMLSelectElement).value);

function refreshActions() {
  const ready = !running && storeSource && uiCovariates() && inputs.some((i) => i.marker);
  ($("runButton") as HTMLButtonElement).disabled = !ready;
}

/** While scoring, nothing that feeds the run may change. */
function setBusy(busy: boolean) {
  for (const id of ["imageInput", "backgroundInput", "clearBackgroundButton", "storeFolder", "storeUrl", "age", "sex", "compute", "clearCacheButton"]) ($(id) as HTMLInputElement).disabled = busy;
  for (const select of $("mapList").querySelectorAll("select")) select.disabled = busy;
  exampleControl.setDisabled(busy);
  refreshActions();
}

const results = createResultList<ZmapOutput>({
  element: $("resultList"),
  onView: (stage) => void showOutput(stage),
  onDownload: (_stage, result) => result && downloadFile(result.file),
});

async function showOutput(marker: string) {
  const output = outputs[marker];
  if (!output) return;
  const mismatch = background && backgroundProblem(background, output.header);
  if (mismatch) {
    setBackground(null);
    status(mismatch, true);
  }
  shownMarker = marker;
  await viewer.show({ zmap: output.display, overlay: output.overlay, magnitude: output.magnitude, background: background?.file }, (mm) => {
    const voxel = output.locate(mm);
    if (!voxel) return "";
    const value = output.values[voxel.index];
    const where = `${voxel.centre.map((v) => Number(v.toFixed(1))).join(", ")} mm`;
    return `${where} · ${Number.isFinite(value) ? `${marker} z = ${value.toFixed(3)}` : `${marker} not scored`}`;
  });
  $("emptyState").hidden = true;
  if (mismatch) return;
  status(background
    ? `Showing ${marker} z-map over ${background.name} · opacity grows with |z|, opaque and colour saturated at |z| = ${Z_DISPLAY_LIMIT}`
    : `Showing ${marker} z-map · colour saturates at |z| = ${Z_DISPLAY_LIMIT}`);
}

/** Least share of the z-map's field of view a background must cover. */
const MIN_BACKGROUND_OVERLAP = 0.5;

/** Why a background cannot sit under a z-map on `header`'s grid, if it cannot. */
function backgroundProblem(image: BackgroundImage, header: SubjectMap["header"]): string | undefined {
  const overlap = fieldOfViewOverlap(image.header, header);
  if (overlap >= MIN_BACKGROUND_OVERLAP) return undefined;
  return `${image.name}: its field of view covers only ${Math.round(100 * overlap)}% of the z-map grid; is it in UK Biobank space?`;
}

function setBackground(image: (BackgroundImage & { file: File }) | null) {
  background = image;
  $("backgroundInfo").hidden = !image;
  $("backgroundInfo").textContent = image ? `${image.name} · ${image.header.nx}×${image.header.ny}×${image.header.nz} at ${image.header.voxelSize.map((v) => Number(v.toFixed(2))).join("×")} mm` : "";
  $("clearBackgroundButton").hidden = !image;
}

/** Show the background change: under the shown z-map, or alone before any run. */
async function redisplay() {
  if (shownMarker && outputs[shownMarker]) return showOutput(shownMarker);
  if (!background) {
    await viewer.show({}, null);
    $("emptyState").hidden = false;
    return;
  }
  await viewer.show({ background: background.file }, null);
  $("emptyState").hidden = true;
  status(`Background ${background.name} loaded · compute z-maps to draw them over it`);
}

/** Read a background image; the first file is used. */
async function loadBackground(files: File[]) {
  if (running) throw new Error("Wait for the current run to finish, or cancel it.");
  const [file] = files;
  const image = { ...(await readBackground(file.name, await file.arrayBuffer())), file };
  const reference = (shownMarker && outputs[shownMarker]?.header) || inputs[0]?.map.header;
  const problem = reference && backgroundProblem(image, reference);
  if (problem) throw new Error(problem);
  setBackground(image);
  await redisplay();
  if (files.length > 1) status(`Background: using ${file.name}; ignored ${files.slice(1).map((f) => f.name).join(", ")}`, true);
}

function renderMapList() {
  const list = $("mapList");
  list.replaceChildren(
    ...inputs.map((input, index) => {
      const field = document.createElement("div");
      field.className = "nd-field";
      const label = document.createElement("label");
      label.htmlFor = `marker-${index}`;
      label.textContent = input.file.name;
      const select = document.createElement("select");
      select.id = `marker-${index}`;
      select.append(new Option("Not used", ""), ...MARKERS.map((m) => new Option(m, m)));
      select.value = input.marker ?? "";
      select.disabled = Boolean(running);
      select.onchange = () => {
        const marker = isMarker(select.value) ? select.value : null;
        // One map per marker: choosing a marker takes it from any other map.
        for (const other of inputs) if (other !== input && marker && other.marker === marker) other.marker = null;
        input.marker = marker;
        renderMapList();
        refreshActions();
      };
      field.append(label, select);
      if (input.problem) {
        const message = document.createElement("p");
        message.className = "nd-message warning";
        message.id = `marker-${index}-problem`;
        message.textContent = input.problem;
        select.setAttribute("aria-describedby", message.id);
        field.append(message);
      }
      return field;
    }),
  );
  $("dropZone").classList.toggle("has-files", inputs.length > 0);
}

/** Tell the page drop which files the list holds now, so Undo restores it as it was. */
function reportFiles() {
  $("imageInput").dispatchEvent(new CustomEvent("nd-files", { bubbles: true, detail: { files: inputs.map((i) => i.file) } }));
}

function resetOutputs() {
  outputs = {};
  shownMarker = null;
  results.render({});
}

/** Bumped by clearMaps: a load that started before a clear must not add to the new list. */
let loadGeneration = 0;

/** Read maps and add them to the list; a newer map of a marker replaces the older one. Returns the maps read. */
async function loadMaps(files: File[], signal?: AbortSignal): Promise<InputMap[]> {
  if (running) throw new Error("Wait for the current run to finish, or cancel it.");
  const generation = loadGeneration;
  const loaded: InputMap[] = [];
  for (const file of files) {
    signal?.throwIfAborted();
    if (generation !== loadGeneration) return [];
    try {
      const map = await loadSubjectMap(file.name, await file.arrayBuffer());
      loaded.push({ file, map, marker: guessMarker(file.name), problem: store ? gridProblem(map, store) : undefined });
    } catch (error) {
      status((error as Error).message, true);
    }
  }
  signal?.throwIfAborted();
  if (generation !== loadGeneration || running) return [];
  // One map per marker: within this batch the last file wins, and it replaces an earlier map.
  const seen = new Set<Marker>();
  const duplicates: string[] = [];
  for (const input of [...loaded].reverse()) {
    if (!input.marker) continue;
    if (seen.has(input.marker)) {
      duplicates.push(input.file.name);
      input.marker = null;
    } else seen.add(input.marker);
  }
  const replaced = inputs.filter((i) => i.marker && seen.has(i.marker));
  inputs = [...inputs.filter((i) => !replaced.includes(i)), ...loaded];
  resetOutputs();
  renderMapList();
  // The page drop records what an input holds; tell it after every change
  // (drop-zone adds never fire `change`, and failed files are not in the list).
  reportFiles();
  const unnamed = loaded.filter((i) => !i.marker).length;
  if (duplicates.length) status(`Same marker twice; set to "Not used": ${duplicates.join(", ")}`, true);
  else if (loaded.length) status(unnamed ? `${loaded.length} map(s) loaded · choose the marker for ${unnamed}` : `${loaded.length} map(s) loaded`);
  refreshActions();
  return loaded;
}

function clearMaps() {
  if (running) return; // e.g. the status-bar Undo after a page drop, during a run
  loadGeneration++;
  inputs = [];
  setBackground(null); // it belongs to the subject being replaced
  resetOutputs();
  renderMapList();
  refreshActions();
}

/** Why a map cannot be scored against the store (without its file name, shown beside it), if it cannot. */
function gridProblem(map: SubjectMap, modelStore: ModelStore): string | undefined {
  try {
    gridMapping(map, modelStore.grid);
    return undefined;
  } catch (error) {
    const message = (error as Error).message;
    return message.startsWith(`${map.name}: `) ? message.slice(map.name.length + 2) : message;
  }
}

/** Open the chosen store once; a store chosen meanwhile wins over a slow open. */
async function ensureStore(): Promise<ModelStore> {
  const source = storeSource;
  if (!source) throw new Error("Choose a model store under Advanced settings");
  if (store) return store;
  status("Opening model store…");
  const opened = await openModelStore(source);
  if (storeSource !== source) throw new Error("The model store changed while it was opening; run again.");
  store = opened;
  log.log(`Model store ${opened.label}: ${opened.family}, ${opened.markers.join(", ")}, ${opened.grid.nVox} voxels`, "info");
  for (const input of inputs) input.problem = gridProblem(input.map, opened);
  renderMapList();
  return opened;
}

function setStoreSource(source: StoreSource | null) {
  storeSource = source;
  store = null;
  for (const input of inputs) input.problem = undefined;
  renderMapList();
  refreshActions();
}

let predictor: { prefer: PredictorKind; instance: Predictor } | null = null;
async function ensurePredictor(prefer: PredictorKind): Promise<Predictor> {
  if (predictor?.prefer === prefer) return predictor.instance;
  predictor?.instance.dispose();
  const { predictor: instance, reason } = await createPredictor(prefer, (why) => status(`GPU failed (${why}); continuing on the CPU`, true));
  if (reason) log.log(`WebGPU unavailable (${reason}); using the CPU`, "warning");
  predictor = { prefer, instance };
  return instance;
}

interface RunResult {
  artifacts: { role: string; file: File }[];
  provenance: Record<string, unknown>;
}

async function run(cov: Covariates, prefer: PredictorKind, signal: AbortSignal): Promise<RunResult> {
  const modelStore = await ensureStore();
  signal.throwIfAborted();
  const chosen = inputs.filter((i): i is InputMap & { marker: Marker } => i.marker !== null);
  if (!chosen.length) throw new Error("Load at least one map and choose its marker");
  const usable = chosen.filter((i) => !i.problem && modelStore.markers.includes(i.marker));
  const skipped = chosen.filter((i) => !usable.includes(i));
  for (const input of skipped) log.log(input.problem ?? `${input.file.name}: the store has no ${input.marker} models`, "warning");
  if (!usable.length) throw new Error(skipped[0]?.problem ?? "No map can be scored against this store");

  const compute = await ensurePredictor(prefer);
  resetOutputs();
  const produced: Record<string, ZmapOutput> = {};
  const notices: string[] = [];
  const kinds = new Set<PredictorKind>();
  const started = performance.now();
  for (const [k, input] of usable.entries()) {
    const share = (f: number) => (k + f) / usable.length;
    const mapping = gridMapping(input.map, modelStore.grid);
    status(`Loading ${input.marker} models…`);
    const params = await modelStore.loadMarker(input.marker, { signal, onProgress: (f) => progress.setProgress(share(0.9 * f)) });
    status(`Scoring ${input.marker}…`);
    const { z, valid, clamped, kind } = await compute.zmap(params, cov, gatherToStore(input.map, mapping), signal);
    kinds.add(kind);
    const notice = clampingNotice(cov.age, clamped, trainingRange(params.covRanges, valid));
    if (notice) notices.push(notice);
    const scored = valid.reduce((n, x) => n + x, 0);
    const name = zmapFileName(input.marker);
    const { nx, ny, nz } = input.map.header;
    const values = new Float32Array(nx * ny * nz).fill(Number.NaN);
    for (let v = 0; v < mapping.length; v++) if (valid[v]) values[mapping[v]] = z[v];
    const description = `normative z (${modelStore.family} ${input.marker}, age ${cov.age}, ${cov.sex})`;
    produced[input.marker] = {
      description: `${input.marker} z-map · ${scored.toLocaleString()} voxels`,
      file: new File([scatterToNifti(z, valid, mapping, input.map, description) as Uint8Array<ArrayBuffer>], name, { type: "application/gzip" }),
      display: new File([displayNifti(z, valid, mapping, input.map, Z_DISPLAY_LIMIT) as Uint8Array<ArrayBuffer>], name.replace(/\.gz$/, "")),
      overlay: new File([displayNifti(z, valid, mapping, input.map, Z_DISPLAY_LIMIT, 0) as Uint8Array<ArrayBuffer>], name.replace(/\.nii\.gz$/, "_overlay.nii")),
      magnitude: new File([displayNifti(z.map(Math.abs), valid, mapping, input.map, Z_DISPLAY_LIMIT, 0) as Uint8Array<ArrayBuffer>], name.replace(/\.nii\.gz$/, "_abs.nii")),
      header: input.map.header,
      values,
      locate: voxelAtMm(input.map.header),
    };
    log.log(`${input.marker}: ${scored} voxels scored on ${kind.toUpperCase()}`, "info");
    progress.setProgress(share(1));
  }
  outputs = produced;
  results.render(outputs, usable.map((i) => i.marker));
  ($("outputSection") as HTMLDetailsElement).open = true;
  await showOutput(usable[0].marker);
  const seconds = ((performance.now() - started) / 1000).toFixed(1);
  const engines = [...kinds].map((kind) => kind.toUpperCase()).join("+");
  progress.end(`${usable.length} z-map(s) ready in ${seconds} s`);
  const skippedNote = skipped.length ? ` · skipped ${skipped.map((i) => i.marker).join(", ")} (see log)` : "";
  status(`${notices[0] ?? `${usable.length} z-map(s) ready · ${engines} · ${seconds} s`}${skippedNote}`, skipped.length > 0);
  return {
    artifacts: Object.values(outputs).map((o) => ({ role: "zmap", file: o.file })),
    provenance: {
      method: "pcntoolkit-blr-zscore",
      store: modelStore.label,
      family: modelStore.family,
      compute: [...kinds],
      covariates: cov,
      markers: usable.map((i) => i.marker),
      skipped: skipped.map((i) => ({ file: i.file.name, marker: i.marker, reason: i.problem ?? "no models in store" })),
      ageClamped: notices.length > 0,
    },
  };
}

/** One run at a time, from the UI or automation; the footer × cancels it. */
async function guardedRun(cov: Covariates, prefer: PredictorKind, signal?: AbortSignal): Promise<RunResult> {
  if (running) throw new Error("A run is already in progress.");
  const controller = new AbortController();
  const abort = () => controller.abort();
  signal?.addEventListener("abort", abort, { once: true });
  running = controller;
  setBusy(true);
  progress.begin("Computing z-maps…", { cancellable: true });
  try {
    return await run(cov, prefer, controller.signal);
  } catch (error) {
    const aborted = (error as Error).name === "AbortError";
    progress.end(aborted ? "Cancelled" : "Failed", { success: false });
    status(aborted ? "Cancelled" : (error as Error).message, !aborted);
    throw error;
  } finally {
    signal?.removeEventListener("abort", abort);
    running = null;
    progress.setCancellable(false);
    setBusy(false);
  }
}

$("runButton").addEventListener("click", () => {
  const cov = uiCovariates();
  if (cov) void guardedRun(cov, ($("compute") as HTMLSelectElement).value as PredictorKind).catch(() => {});
});
$("cancelButton").onclick = () => {
  exampleControl.cancel();
  running?.abort();
};
for (const id of ["age", "sex"]) $(id).addEventListener("input", refreshActions);
/** Use a store URL and show it in the Advanced settings. */
function useStoreUrl(url: string) {
  ($("storeUrl") as HTMLInputElement).value = url;
  $("storeInfo").hidden = true;
  setStoreSource({ kind: "url", url });
}

$("storeUrl").addEventListener("change", () => {
  const url = ($("storeUrl") as HTMLInputElement).value.trim();
  setStoreSource(url ? { kind: "url", url } : DEFAULT_STORE_URL ? { kind: "url", url: DEFAULT_STORE_URL } : null);
});
$("storeFolder").addEventListener("change", (event) => {
  const files = Array.from((event.target as HTMLInputElement).files ?? []);
  if (!files.length) return;
  setStoreSource({ kind: "directory", handle: filesDirectory(files) });
  ($("storeUrl") as HTMLInputElement).value = "";
  $("storeInfo").hidden = false;
  $("storeInfo").textContent = `${files[0].webkitRelativePath.split("/")[0]} · ${files.length} files`;
});
$("clearCacheButton").addEventListener("click", () => {
  void clearModelCache().then((cleared) => status(cleared ? "Cached models cleared" : "No cached models"));
});

// The map list appends (data-appends). A whole-window drop replaces the
// session: the shell fires nd-replace first, then change with the new files.
$("imageInput").addEventListener("nd-replace", clearMaps);
$("imageInput").addEventListener("change", (event) => {
  const target = event.target as HTMLInputElement;
  const files = Array.from(target.files ?? []);
  target.value = "";
  exampleControl.cancel();
  if (files.length) void loadMaps(files).catch((error) => status((error as Error).message, true));
});
$("backgroundInput").addEventListener("change", (event) => {
  const target = event.target as HTMLInputElement;
  const files = Array.from(target.files ?? []);
  target.value = "";
  if (files.length) void loadBackground(files).catch((error) => status((error as Error).message, true));
});
// No page-level target: a whole-window drop still goes to the maps.
bindFileDrop($("backgroundDrop"), (files) => {
  void files.then((f) => (f.length ? loadBackground(f) : undefined)).catch((error) => status((error as Error).message, true));
}, undefined, () => Boolean(running));
$("clearBackgroundButton").addEventListener("click", () => {
  setBackground(null);
  void redisplay();
});
bindFileDrop($("dropZone"), (files) => {
  exampleControl.cancel();
  void files.then((f) => loadMaps(f)).catch((error) => status((error as Error).message, true));
}, document, () => Boolean(running));

const exampleControl = createExampleSelector({
  examples,
  onStatus: status,
  onLoad: async (_example, { fetchFiles, assertCurrent, signal }) => {
    const files = await fetchFiles();
    assertCurrent();
    clearMaps();
    const loaded = await loadMaps(files, signal);
    if (!loaded.some((i) => i.marker)) throw new Error("The example has no usable maps");
  },
});
$("exampleControl").append(exampleControl);

registerAppAutomation({
  app: APP.id,
  operations: {
    zmap: async ({ inputs: given, parameters, signal }) => {
      const cov = parseCovariates(String(parameters.age ?? ""), String(parameters.sex ?? ""));
      if (!cov) throw new Error("zmap needs age (years, > 0) and sex (female or male)");
      if (running) throw new Error("A run is already in progress.");
      // storeUrl, else the store already chosen in the page (URL or folder), else the default.
      const url = typeof parameters.storeUrl === "string" && parameters.storeUrl ? parameters.storeUrl : null;
      if (url) {
        if (storeSource?.kind !== "url" || storeSource.url !== url) useStoreUrl(url);
      } else if (!storeSource) {
        throw new Error("zmap needs storeUrl until a default model store is published");
      }
      clearMaps();
      await loadMaps(given.maps as File[], signal);
      const unnamed = inputs.filter((i) => !i.marker).map((i) => i.file.name);
      if (unnamed.length) throw new Error(`Cannot tell the marker of ${unnamed.join(", ")} from its name`);
      const compute = parameters.compute === "gpu" ? "gpu" : "cpu";
      return guardedRun(cov, compute, signal);
    },
  },
});

window.addEventListener("pagehide", () => {
  exampleControl.destroy();
  running?.abort();
  predictor?.instance.dispose();
});

void pruneOldCaches();
await viewer.attach().catch((error) => status(`Viewer unavailable: ${(error as Error).message}`, true));
refreshActions();
