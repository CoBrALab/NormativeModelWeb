// Loads the PCNtoolkit parity fixtures (tools/make_fixtures.py) as MarkerParams.
import { readFileSync } from "node:fs";

export { fixtureParams, sexOf } from "./fixture-params.js";

export const FIXTURE_MARKERS = ["FA", "MD", "ICVF", "ISOVF", "OD"];

export function loadFixture(marker) {
  return JSON.parse(readFileSync(new URL(`./fixtures/predict-${marker}.json`, import.meta.url), "utf8"));
}
