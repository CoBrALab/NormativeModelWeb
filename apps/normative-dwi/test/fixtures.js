// Loads the PCNtoolkit parity fixtures (tools/make_fixtures.py) as MarkerParams.
import { existsSync, readFileSync } from "node:fs";

export { fixtureParams, sexOf } from "./fixture-params.js";

export const FIXTURE_MARKERS = ["FA", "MD", "ICVF", "ISOVF", "OD"];

/** Fixtures are made locally by tools/make_fixtures.py and never committed. */
export function fixturesAvailable() {
  return FIXTURE_MARKERS.every((marker) => existsSync(new URL(`./fixtures/predict-${marker}.json`, import.meta.url)));
}

export function loadFixture(marker) {
  return JSON.parse(readFileSync(new URL(`./fixtures/predict-${marker}.json`, import.meta.url), "utf8"));
}
