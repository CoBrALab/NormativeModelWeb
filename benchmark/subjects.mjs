// The synthetic subjects both benchmarks score. Keep in step with subjects.py.

export const age = (s) => 50 + ((s * 7.919) % 25);

/** The app's sex for subject s; "female" is PCNtoolkit's Sex level "0". */
export const sex = (s) => (s % 2 === 0 ? "female" : "male");

export const response = (v, s, outM, outS) => outM + outS * 1.5 * Math.sin(0.618 * v + 1.3 * s + 0.1);
