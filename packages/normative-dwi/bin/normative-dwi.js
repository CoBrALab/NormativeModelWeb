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
  -o, --output DIR      where to write the z-maps (default: current folder);
                        existing z-maps are not overwritten
  --marker FILE=MARKER  marker of a map whose file name does not say it
  -h, --help            show this help

Progress and warnings (such as an age outside a model's training range, which
is scored at the nearest age) go to standard error; the written files go to
standard output. self-check scores a generated store and checks the result.`;

function markerOptions(values = []) {
  return Object.fromEntries(values.map((value) => {
    const at = value.lastIndexOf("=");
    if (at <= 0) throw new Error(`--marker ${value}: expected FILE=MARKER`);
    return [value.slice(0, at), value.slice(at + 1).toUpperCase()];
  }));
}

try {
  const { values, positionals } = parseArgs({
    allowPositionals: true,
    options: {
      help: { type: "boolean", short: "h" },
      store: { type: "string" },
      age: { type: "string" },
      sex: { type: "string" },
      output: { type: "string", short: "o" },
      marker: { type: "string", multiple: true },
    },
  });
  if (values.help || !positionals.length) {
    console.log(HELP);
  } else if (positionals[0] === "self-check") {
    if (positionals.length !== 1 || Object.keys(values).length) throw new Error("self-check does not accept arguments.");
    console.log(JSON.stringify(await checkInstallation()));
  } else {
    if (!values.store) throw new Error("Give the model store folder with --store. Use --help for details.");
    const covariates = parseCovariates(values.age ?? "", values.sex ?? "");
    if (!covariates) throw new Error("Give --age (years, above 0 and up to 120) and --sex (female or male).");
    const result = await scoreFiles({
      store: values.store,
      maps: positionals,
      markers: markerOptions(values.marker),
      covariates,
      output: values.output ?? ".",
      onStatus: (message) => process.stderr.write(`${message}\n`),
    });
    for (const file of result.files) console.log(file.output);
  }
} catch (error) {
  console.error(error.message);
  process.exitCode = 1;
}
