// Normative DWI: maps in, z-maps out. The interface comes from the shared
// workspace vocabulary; the science is in model-store/, predictor/ and
// subject-image/ (see GLOSSARY.md and docs/adr/).
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
import { APP, DEFAULT_STORE_URL, Z_DISPLAY_LIMIT, clampingNotice, guessMarker, isMarker, trainingRange, zmapFileName } from "./config.ts";
import { MARKERS, type Covariates, type Marker } from "./model.ts";
import { openModelStore, type ModelStore, type StoreSource } from "./model-store/index.ts";
import { filesDirectory } from "./model-store/directory-store.ts";
import { createPredictor, type Predictor, type PredictorKind } from "./predictor/index.ts";
import { displayNifti, gatherToStore, gridMapping, loadSubjectMap, scatterToNifti, type SubjectMap } from "./subject-image/index.ts";
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
}
interface ZmapOutput {
  [key: string]: unknown;
  description: string;
  file: File;
  display: File;
}

let inputs: InputMap[] = [];
let outputs: Record<string, ZmapOutput> = {};
let store: ModelStore | null = null;
let storeSource: StoreSource | null = DEFAULT_STORE_URL ? { kind: "url", url: DEFAULT_STORE_URL } : null;
let running: AbortController | null = null;

function status(message: string, error = false) {
  $("statusText").textContent = message.length <= 90 ? message : `${message.slice(0, 88).trimEnd()}…`;
  $("statusText").classList.toggle("error", error);
  log.log(message, error ? "error" : "info");
}

function covariates(): Covariates | null {
  const age = Number.parseFloat(($("age") as HTMLInputElement).value);
  const sex = ($("sex") as HTMLSelectElement).value;
  return Number.isFinite(age) && age > 0 && (sex === "female" || sex === "male") ? { age, sex } : null;
}

function refreshActions() {
  const ready = !running && storeSource && covariates() && inputs.some((i) => i.marker);
  ($("runButton") as HTMLButtonElement).disabled = !ready;
}

const results = createResultList<ZmapOutput>({
  element: $("resultList"),
  onView: (stage) => void showOutput(stage),
  onDownload: (_stage, result) => result && downloadFile(result.file),
});

async function showOutput(marker: string) {
  const output = outputs[marker];
  if (!output) return;
  await viewer.show(output.display);
  $("emptyState").hidden = true;
  status(`Showing ${marker} z-map · colour saturates at |z| = 3`);
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
      select.onchange = () => {
        input.marker = isMarker(select.value) ? select.value : null;
        refreshActions();
      };
      field.append(label, select);
      return field;
    }),
  );
  $("dropZone").classList.toggle("has-files", inputs.length > 0);
}

async function loadMaps(files: File[], signal?: AbortSignal) {
  const loaded: InputMap[] = [];
  for (const file of files) {
    signal?.throwIfAborted();
    try {
      const map = await loadSubjectMap(file.name, await file.arrayBuffer());
      loaded.push({ file, map, marker: guessMarker(file.name) });
    } catch (error) {
      status(`${file.name}: ${(error as Error).message}`, true);
    }
  }
  // A marker can come from one map only: the newest file wins.
  const taken = new Set(loaded.map((i) => i.marker).filter(Boolean));
  inputs = [...inputs.filter((i) => !taken.has(i.marker)), ...loaded];
  outputs = {};
  results.render({});
  renderMapList();
  const unnamed = loaded.filter((i) => !i.marker).length;
  if (loaded.length) status(unnamed ? `${loaded.length} map(s) loaded · choose the marker for ${unnamed}` : `${loaded.length} map(s) loaded`);
  refreshActions();
}

async function ensureStore(): Promise<ModelStore> {
  if (store) return store;
  if (!storeSource) throw new Error("Choose a model store under Advanced settings");
  status("Opening model store…");
  store = await openModelStore(storeSource);
  log.log(`Model store ${store.label}: ${store.family}, ${store.markers.join(", ")}, ${store.grid.nVox} voxels`, "info");
  return store;
}

let predictor: { kind: PredictorKind; instance: Predictor } | null = null;
async function ensurePredictor(prefer: PredictorKind): Promise<Predictor> {
  if (predictor?.kind === prefer) return predictor.instance;
  predictor?.instance.dispose();
  const { predictor: instance, reason } = await createPredictor(prefer);
  if (reason) log.log(`WebGPU unavailable (${reason}); using the CPU`, "warning");
  predictor = { kind: prefer, instance };
  return instance;
}

interface RunResult {
  artifacts: { role: string; file: File }[];
  provenance: Record<string, unknown>;
}

async function run(cov: Covariates, prefer: PredictorKind, signal: AbortSignal): Promise<RunResult> {
  const chosen = inputs.filter((i): i is InputMap & { marker: Marker } => i.marker !== null);
  if (!chosen.length) throw new Error("Load at least one map and choose its marker");
  const modelStore = await ensureStore();
  const compute = await ensurePredictor(prefer);
  outputs = {};
  const notices: string[] = [];
  const started = performance.now();
  for (const [k, input] of chosen.entries()) {
    const share = (f: number) => (k + f) / chosen.length;
    const mapping = gridMapping(input.map, modelStore.grid);
    status(`Loading ${input.marker} models…`);
    const params = await modelStore.loadMarker(input.marker, { signal, onProgress: (f) => progress.setProgress(share(0.9 * f)) });
    signal.throwIfAborted();
    status(`Scoring ${input.marker}…`);
    const { z, valid, clamped } = await compute.zmap(params, cov, gatherToStore(input.map, mapping));
    const notice = clampingNotice(cov.age, clamped, trainingRange(params.covRanges, valid));
    if (notice) notices.push(notice);
    const scored = valid.reduce((n, x) => n + x, 0);
    const name = zmapFileName(input.marker);
    outputs[input.marker] = {
      description: `${input.marker} z-map · ${scored.toLocaleString()} voxels`,
      file: new File([scatterToNifti(z, valid, mapping, input.map) as Uint8Array<ArrayBuffer>], name, { type: "application/gzip" }),
      display: new File([displayNifti(z, valid, mapping, input.map, Z_DISPLAY_LIMIT) as Uint8Array<ArrayBuffer>], name.replace(/\.gz$/, "")),
    };
    log.log(`${input.marker}: ${scored} voxels scored on ${compute.kind.toUpperCase()}`, "info");
    progress.setProgress(share(1));
  }
  results.render(outputs, chosen.map((i) => i.marker));
  ($("outputSection") as HTMLDetailsElement).open = true;
  await showOutput(chosen[0].marker);
  const seconds = ((performance.now() - started) / 1000).toFixed(1);
  progress.end(`${chosen.length} z-map(s) ready in ${seconds} s`);
  status(notices[0] ?? `${chosen.length} z-map(s) ready · ${compute.kind.toUpperCase()} · ${seconds} s`, false);
  return {
    artifacts: Object.values(outputs).map((o) => ({ role: "zmap", file: o.file })),
    provenance: {
      method: "pcntoolkit-blr-zscore",
      store: modelStore.label,
      family: modelStore.family,
      compute: compute.kind,
      covariates: cov,
      markers: chosen.map((i) => i.marker),
      ageClamped: notices.length > 0,
    },
  };
}

async function runFromUi(): Promise<RunResult | null> {
  const cov = covariates();
  if (!cov || running) return null;
  running = new AbortController();
  refreshActions();
  progress.begin("Computing z-maps…", { cancellable: true });
  try {
    return await run(cov, ($("compute") as HTMLSelectElement).value as PredictorKind, running.signal);
  } catch (error) {
    const aborted = (error as Error).name === "AbortError";
    progress.end(aborted ? "Cancelled" : "Failed", { success: false });
    status(aborted ? "Cancelled" : (error as Error).message, !aborted);
    if (!aborted) throw error;
    return null;
  } finally {
    running = null;
    progress.setCancellable(false);
    refreshActions();
  }
}

$("runButton").addEventListener("click", () => void runFromUi().catch(() => {}));
$("cancelButton").onclick = () => {
  exampleControl.cancel();
  running?.abort();
};
for (const id of ["age", "sex"]) $(id).addEventListener("input", refreshActions);
$("storeUrl").addEventListener("change", () => {
  const url = ($("storeUrl") as HTMLInputElement).value.trim();
  storeSource = url ? { kind: "url", url } : DEFAULT_STORE_URL ? { kind: "url", url: DEFAULT_STORE_URL } : null;
  store = null;
  refreshActions();
});
$("storeFolder").addEventListener("change", (event) => {
  const files = Array.from((event.target as HTMLInputElement).files ?? []);
  if (!files.length) return;
  storeSource = { kind: "directory", handle: filesDirectory(files) };
  store = null;
  $("storeInfo").hidden = false;
  $("storeInfo").textContent = `${files[0].webkitRelativePath.split("/")[0]} · ${files.length} files`;
  refreshActions();
});
$("imageInput").addEventListener("change", (event) => {
  const target = event.target as HTMLInputElement;
  const files = Array.from(target.files ?? []);
  target.value = "";
  if (files.length) void loadMaps(files);
});
bindFileDrop($("dropZone"), (files) => void files.then((f) => loadMaps(f)));

const exampleControl = createExampleSelector({
  examples,
  onStatus: status,
  onLoad: async (_example, { fetchFiles, assertCurrent, signal }) => {
    const files = await fetchFiles();
    assertCurrent();
    await loadMaps(files, signal);
  },
});
$("exampleControl").append(exampleControl);

registerAppAutomation({
  app: APP.id,
  operations: {
    zmap: async ({ inputs: given, parameters, signal }) => {
      inputs = [];
      await loadMaps(given.maps as File[], signal);
      const unnamed = inputs.filter((i) => !i.marker).map((i) => i.file.name);
      if (unnamed.length) throw new Error(`Cannot tell the marker of ${unnamed.join(", ")} from its name`);
      const cov = { age: Number(parameters.age), sex: parameters.sex as Covariates["sex"] };
      return run(cov, (parameters.compute as PredictorKind) ?? "gpu", signal);
    },
  },
});

window.addEventListener("pagehide", () => {
  exampleControl.destroy();
  running?.abort();
  predictor?.instance.dispose();
});

await viewer.attach().catch((error) => status(`Viewer unavailable: ${(error as Error).message}`, true));
refreshActions();
