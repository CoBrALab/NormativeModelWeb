// The command line as a user runs it: a generated store and maps in, z-map
// files out, compared with the float64 kernel. The last test repeats the
// whole-brain check of end-to-end.test.js through the command line; it needs
// local data (NMZARR_STORE and NORMATIVE_E2E_DIR) and skips without it.
import assert from "node:assert/strict";
import { execFile } from "node:child_process";
import { existsSync, readFileSync } from "node:fs";
import { mkdir, mkdtemp, readdir, readFile, rm, symlink, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { promisify } from "node:util";
import { MARKERS } from "../src/model.js";
import { zmapCpu } from "../src/predictor/kernel.js";
import { gridMapping, loadSubjectMap } from "../src/subject-image/index.js";
import { syntheticMap, syntheticModel, syntheticParams, writeSyntheticStore } from "../src/synthetic-store.js";

const BIN = new URL("../bin/normative-dwi.js", import.meta.url).pathname;
const model = syntheticModel({ markers: ["FA", "MD", "OD"] });
const grid = { ...model, affine: Float64Array.from(model.affine) };

/** Run the command line; resolves with { code, stdout, stderr } whatever the exit code. */
async function cli(args, options = {}) {
  try {
    const { stdout, stderr } = await promisify(execFile)(process.execPath, [BIN, ...args], options);
    return { code: 0, stdout, stderr };
  } catch (error) {
    return { code: error.code, stdout: error.stdout, stderr: error.stderr };
  }
}

let folder;
test.before(async () => {
  folder = await mkdtemp(join(tmpdir(), "normative-dwi-cli-"));
  await writeSyntheticStore(join(folder, "store"), model);
  await writeFile(join(folder, "sub-01_FA.nii"), syntheticMap(model, "FA", 11, "ras").bytes);
  await writeFile(join(folder, "sub-01_MD.nii"), syntheticMap(model, "MD", 12, "minc").bytes);
  await writeFile(join(folder, "subject.nii"), syntheticMap(model, "MD", 12, "las").bytes);
  await writeFile(join(folder, "subject-b.nii"), syntheticMap(model, "OD", 13, "ras").bytes);
  // A store whose MD models were fitted with another warp: refused only when MD is read, after FA is scored.
  await writeSyntheticStore(join(folder, "store-bad-md"), model);
  const root = JSON.parse(await readFile(join(folder, "store-bad-md", "zarr.json"), "utf8"));
  root.consolidated_metadata.metadata.MD.attributes.regression.warp_name = "warpboxcox";
  await writeFile(join(folder, "store-bad-md", "zarr.json"), JSON.stringify(root));
});
test.after(() => rm(folder, { recursive: true, force: true }));

/** Every voxel of a written z-map against the float64 kernel: z (as float32) where scored, 0 elsewhere. */
async function assertMatchesKernel(path, marker, seed, layout, covariates) {
  const written = await loadSubjectMap(path, await readFile(path));
  const { y } = syntheticMap(model, marker, seed, layout);
  const expected = zmapCpu(syntheticParams(model, marker), covariates, y);
  const mapping = gridMapping(written, grid);
  let scored = 0;
  for (let v = 0; v < model.nVox; v++) {
    assert.equal(written.data[mapping[v]], expected.valid[v] ? Math.fround(expected.z[v]) : 0, `voxel ${v}`);
    scored += expected.valid[v];
  }
  assert.equal(written.data.filter((value) => value !== 0).length, scored, "nothing written off the modelled grid");
  return scored;
}

test("scores each map against its marker's models and writes the web app's z-maps", async () => {
  const out = join(folder, "out-both");
  const result = await cli(["--store", join(folder, "store"), "--age", "66", "--sex", "female", "-o", out, join(folder, "sub-01_FA.nii"), join(folder, "sub-01_MD.nii")]);
  assert.equal(result.code, 0, result.stderr);
  assert.deepEqual(result.stdout.trim().split("\n"), [join(out, "FA_zscore.nii.gz"), join(out, "MD_zscore.nii.gz")]);
  const covariates = { age: 66, sex: "female" };
  const fa = await assertMatchesKernel(join(out, "FA_zscore.nii.gz"), "FA", 11, "ras", covariates);
  await assertMatchesKernel(join(out, "MD_zscore.nii.gz"), "MD", 12, "minc", covariates);
  assert.match(result.stderr, new RegExp(`^FA: ${fa} voxels scored$`, "m"));
  assert.doesNotMatch(result.stderr, /outside the model/);
});

test("a map whose name gives no marker is scored with --marker, which takes the file-name aliases too", async () => {
  const out = join(folder, "out-marker");
  const result = await cli(["--store", join(folder, "store"), "--age", "71.5", "--sex", "male", "-o", out, "--marker", "subject.nii=md", "--marker", "subject-b.nii=ODI", join(folder, "subject.nii"), join(folder, "subject-b.nii")]);
  assert.equal(result.code, 0, result.stderr);
  await assertMatchesKernel(join(out, "MD_zscore.nii.gz"), "MD", 12, "las", { age: 71.5, sex: "male" });
  await assertMatchesKernel(join(out, "OD_zscore.nii.gz"), "OD", 13, "ras", { age: 71.5, sex: "male" });
});

test("an age outside the training range is scored at its nearest end, with a warning", async () => {
  const out = join(folder, "out-young");
  const result = await cli(["--store", join(folder, "store"), "--age", "30", "--sex", "female", "-o", out, join(folder, "sub-01_FA.nii")]);
  assert.equal(result.code, 0, result.stderr);
  assert.match(result.stderr, /^FA: Age 30 is outside the model's 45–81 range; scored at the nearest age$/m);
  await assertMatchesKernel(join(out, "FA_zscore.nii.gz"), "FA", 11, "ras", { age: 30, sex: "female" });
});

test("existing z-maps are not overwritten", async () => {
  const out = join(folder, "out-again");
  const args = ["--store", join(folder, "store"), "--age", "66", "--sex", "female", "-o", out, join(folder, "sub-01_FA.nii")];
  assert.equal((await cli(args)).code, 0);
  const again = await cli(args);
  assert.equal(again.code, 1);
  assert.match(again.stderr, /Not overwriting .*FA_zscore\.nii\.gz/);
});

test("a map that fails while scoring leaves nothing behind, not even the maps scored before it", async () => {
  const out = join(folder, "out-partial", "nested");
  const result = await cli(["--store", join(folder, "store-bad-md"), "--age", "66", "--sex", "female", "-o", out, join(folder, "sub-01_FA.nii"), join(folder, "sub-01_MD.nii")]);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /^FA: \d+ voxels scored$/m, "FA was scored");
  assert.match(result.stderr, /MD: unsupported model design \(regression\.warp_name is "warpboxcox"/);
  assert.equal(result.stdout, "");
  assert.ok(!existsSync(join(folder, "out-partial")), "no z-map and no folder left");
});

test("a path that appears at write time is not overwritten, and the run's other z-maps are removed", async () => {
  const out = join(folder, "out-link");
  await mkdir(out);
  // A dangling link passes the existence check before scoring; the exclusive write still refuses it.
  await symlink(join(folder, "elsewhere.nii.gz"), join(out, "MD_zscore.nii.gz"));
  const result = await cli(["--store", join(folder, "store"), "--age", "66", "--sex", "female", "-o", out, join(folder, "sub-01_FA.nii"), join(folder, "sub-01_MD.nii")]);
  assert.equal(result.code, 1);
  assert.match(result.stderr, /Not overwriting .*MD_zscore\.nii\.gz/);
  assert.deepEqual(await readdir(out), ["MD_zscore.nii.gz"], "FA's z-map removed, the link left alone");
  assert.ok(!existsSync(join(folder, "elsewhere.nii.gz")), "nothing written through the link");
});

test("bad input fails before anything is written, and says why", async () => {
  const store = join(folder, "store");
  const fa = join(folder, "sub-01_FA.nii");
  const offGrid = join(folder, "sub-02_FA.nii");
  const { bytes } = syntheticMap(syntheticModel({ shape: [10, 14, 12], markers: ["FA"] }), "FA");
  await writeFile(offGrid, bytes);
  const out = join(folder, "out-bad");
  // Usage errors exit with 2, errors found in the inputs with 1.
  for (const [code, args, message] of [
    [2, ["--age", "66", "--sex", "female", fa], /--store/],
    [2, ["--store", store, "--sex", "female", fa], /--age .* --sex/],
    [2, ["--store", store, "--age", "66", "--sex", "F", fa], /--age .* --sex/],
    [2, ["--store", store, "--age", "66,5", "--sex", "female", fa], /--age .* --sex/],
    [2, ["--store", store, "--age", "72y", "--sex", "female", fa], /--age .* --sex/],
    [2, ["--store", store, "--age", "0x42", "--sex", "female", fa], /--age .* --sex/],
    [2, ["--store", store, "--age", "66", "--age", "70", "--sex", "female", fa], /--age is given 2 times/],
    [2, ["--store", store, "--store", store, "--age", "66", "--sex", "female", fa], /--store is given 2 times/],
    [2, ["--store", store, "--age", "66", "--sex", "female"], /Give at least one map/],
    [2, ["--store", store, "--age", "66", "--sex", "female", "--marker", "FA", fa], /expected FILE=MARKER/],
    [2, ["--store", store, "--age", "66", "--sex", "female", "--unknown", fa], /Unknown option '--unknown'/],
    [1, ["--store", join(folder, "nowhere"), "--age", "66", "--sex", "female", fa], /not a folder/],
    [1, ["--store", folder, "--age", "66", "--sex", "female", fa], /is not a readable model store/],
    [1, ["--store", store, "--age", "66", "--sex", "female", join(folder, "subject.nii")], /names no single marker; add --marker subject\.nii=/],
    [1, ["--store", store, "--age", "66", "--sex", "female", "--marker", "subject.nii=XY", join(folder, "subject.nii")], /XY is not a marker/],
    [1, ["--store", store, "--age", "66", "--sex", "female", "--marker", "other.nii=FA", fa], /--marker other\.nii: no such map/],
    [1, ["--store", store, "--age", "66", "--sex", "female", fa, offGrid], /Two maps for FA/],
    [1, ["--store", store, "--age", "66", "--sex", "female", "--marker", "subject.nii=NDI", join(folder, "subject.nii")], /the store has no ICVF models/],
    [1, ["--store", store, "--age", "66", "--sex", "female", offGrid], /sub-02_FA\.nii: not on the 12×14×12 template grid/],
  ]) {
    const result = await cli([...args, "-o", out]);
    assert.equal(result.code, code, `${args.join(" ")} should exit with ${code}`);
    assert.match(result.stderr, message, args.join(" "));
    assert.equal(result.stdout, "");
    assert.ok(!existsSync(out), "nothing written");
  }
});

test("help and self-check", async () => {
  for (const args of [["--help"], []]) {
    const help = await cli(args);
    assert.equal(help.code, 0);
    assert.match(help.stdout, /^Usage: normative-dwi --store STORE/);
  }
  const check = await cli(["self-check"]);
  assert.equal(check.code, 0, check.stderr);
  const report = JSON.parse(check.stdout);
  assert.equal(report.ok, true);
  assert.ok(report.scored > 0);
  assert.equal((await cli(["self-check", "--age", "66"])).code, 2);
});

const STORE = process.env.NMZARR_STORE;
const DIR = process.env.NORMATIVE_E2E_DIR;
test("whole brain through the command line matches the reference z-maps", { skip: STORE && DIR ? false : "set NMZARR_STORE and NORMATIVE_E2E_DIR", timeout: 900000 }, async () => {
  const { age, sex } = JSON.parse(readFileSync(join(DIR, "reference.json"), "utf8"));
  const markers = MARKERS.filter((marker) => existsSync(join(DIR, `input_${marker}.nii.gz`)));
  assert.ok(markers.length, `no input_<MARKER>.nii.gz in ${DIR}`);
  const out = join(folder, "out-real");
  const result = await cli(["--store", STORE, "--age", String(age), "--sex", sex, "-o", out, ...markers.map((marker) => join(DIR, `input_${marker}.nii.gz`))]);
  assert.equal(result.code, 0, result.stderr);
  for (const marker of markers) {
    const got = await loadSubjectMap("out", await readFile(join(out, `${marker}_zscore.nii.gz`)));
    const ref = await loadSubjectMap("ref", readFileSync(join(DIR, `reference_${marker}.nii.gz`)));
    assert.equal(got.data.length, ref.data.length);
    let worst = 0, zeroMismatch = 0;
    for (let i = 0; i < ref.data.length; i++) {
      if ((ref.data[i] === 0) !== (got.data[i] === 0)) zeroMismatch++;
      worst = Math.max(worst, Math.abs(got.data[i] - ref.data[i]) / Math.max(1, Math.abs(ref.data[i])));
    }
    assert.equal(zeroMismatch, 0, marker);
    assert.ok(worst < 1e-6, `${marker}: max relative |Δz| ${worst}`);
  }
});
