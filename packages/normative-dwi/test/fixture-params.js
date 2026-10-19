// Fixture voxels → MarkerParams. No Node imports: also loaded in the browser by e2e/gpu-parity.spec.js.
/** Pack fixture voxels into one MarkerParams, row v = fixture.voxels[v]. */
export function fixtureParams(fixture, records = fixture.voxels) {
  const flat = (key, Type) => Type.from(records.flatMap((r) => [r.params[key]].flat(2)));
  return {
    marker: fixture.marker,
    nVox: records.length,
    nKnots: records[0].params.mean_knots.length,
    beLevelColumns: fixture.be_level_columns,
    hasModel: Uint8Array.from(records.map((r) => (r.params.has_model ? 1 : 0))),
    m: flat("m", Float64Array),
    aTriu: flat("A_triu", Float64Array),
    hyp: flat("hyp", Float64Array),
    meanKnots: flat("mean_knots", Float64Array),
    inscaler: flat("inscaler", Float64Array),
    outscaler: flat("outscaler", Float64Array),
    beMap: flat("be_map", Int16Array),
    covRanges: flat("cov_ranges", Float64Array),
  };
}

export const sexOf = (code) => (code === "0" ? "female" : "male");
