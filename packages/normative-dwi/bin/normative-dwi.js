#!/usr/bin/env node
import { parseArgs } from "node:util";
import { parseCovariates } from "../src/config.js";
import { MARKERS } from "../src/model.js";
import { checkInstallation, scoreFiles } from "../src/node.js";

const HELP = `Usage: normative-dwi --store STORE --age YEARS --sex female|male [-o OUTPUT_DIR]
                     [--marker FILE=MARKER ...] MAP.nii[.gz] ...
       normative-dwi self-check

Scores one person's diffusion microstructure maps against the voxelwise
normative models of the UK Biobank population in STORE, as the Normative DWI
web app does on its CPU, and writes a z-score map for each:

  OUTPUT_DIR/<MARKER>_zscore.nii.gz   z on the map's own grid, 0 where unscored

Each map is one of ${MARKERS.join(", ")}, named in its file name (NDI, FWF/FISO and
ODI are also recognised) or given with --marker. Maps must already be
registered to the UK Biobank 2 mm template and harmonised to UK Biobank.

Options:
  --store STORE         nmzarr model store folder (required)
  --age YEARS           age at scan (required)
  --sex female|male     sex (required)
  -o, --output DIR      where to write the z-maps (default: current folder)
  --marker FILE=MARKER  marker of a map whose file name does not say it
                        (a marker or one of the aliases above)
  -h, --help            show this help

The z-maps are written only when every map has been scored: a run that fails
writes nothing, and an existing file is never overwritten. Progress and
warnings (such as an age outside a model's training range, which is scored at
the nearest age) go to standard error; the written files go to standard
output. self-check scores a generated store and checks the result.

Exit status: 0 on success, 1 when scoring fails, 2 on a usage error.`;

/** A mistake in the arguments: exit status 2. */
class UsageError extends Error {}

function markerOptions(values = []) {
  return Object.fromEntries(values.map((value) => {
    const at = value.lastIndexOf("=");
    if (at <= 0) throw new UsageError(`--marker ${value}: expected FILE=MARKER`);
    return [value.slice(0, at), value.slice(at + 1)];
  }));
}

/** The one value of an option given at most once. */
function single(values, name) {
  if (values[name] && values[name].length > 1) throw new UsageError(`--${name} is given ${values[name].length} times; give it once.`);
  return values[name]?.[0];
}

function parse(args) {
  try {
    return parseArgs({
      args,
      allowPositionals: true,
      options: {
        help: { type: "boolean", short: "h" },
        store: { type: "string", multiple: true },
        age: { type: "string", multiple: true },
        sex: { type: "string", multiple: true },
        output: { type: "string", short: "o", multiple: true },
        marker: { type: "string", multiple: true },
      },
    });
  } catch (error) {
    throw new UsageError(error.message);
  }
}

try {
  const args = process.argv.slice(2);
  const { values, positionals } = parse(args);
  if (values.help || !args.length) {
    console.log(HELP);
  } else if (positionals[0] === "self-check") {
    if (positionals.length !== 1 || Object.keys(values).length) throw new UsageError("self-check does not accept arguments.");
    console.log(JSON.stringify(await checkInstallation()));
  } else {
    const store = single(values, "store");
    const output = single(values, "output") ?? ".";
    if (!store) throw new UsageError("Give the model store folder with --store.");
    const covariates = parseCovariates(single(values, "age") ?? "", single(values, "sex") ?? "");
    if (!covariates) throw new UsageError("Give --age (a number of years, above 0 and up to 120) and --sex (female or male).");
    if (!positionals.length) throw new UsageError("Give at least one map.");
    const result = await scoreFiles({
      store,
      maps: positionals,
      markers: markerOptions(values.marker),
      covariates,
      output,
      onStatus: (message) => process.stderr.write(`${message}\n`),
    });
    for (const file of result.files) console.log(file.output);
  }
} catch (error) {
  if (error instanceof UsageError) {
    console.error(`${error.message} Use --help for details.`);
    process.exitCode = 2;
  } else {
    console.error(error.message);
    process.exitCode = 1;
  }
}
