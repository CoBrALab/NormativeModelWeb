// WebGPU float32 z-score adapter (docs/adr/0001). One invocation per voxel,
// the same steps in the same order as the float64 kernel in kernel.ts, except
// that the Cholesky factor of A is computed on the CPU in float64 and uploaded:
// the design is collinear (spline basis, intercept and sex columns each sum to
// one), cond(A) reaches ~1e8, and a float32 factorisation breaks down.
// Validity (modelled, nonzero, finite) and age clamping counts are decided on
// the CPU, so both adapters agree exactly on which voxels are scored.
import { designSize, type Covariates, type MarkerParams, type ZmapResult } from "../model.ts";
import { choleskyLower, sexColumn } from "./kernel.ts";

const WORKGROUP = 64;
// The shader is written for the ukb_blr_agesex design: 10 knots (6 cubic
// basis functions), intercept, two batch-effect columns.
const KNOTS = 10;
const BE_LEVELS = 2;
const D = designSize({ nKnots: KNOTS, beLevelColumns: Array(BE_LEVELS) });

export const SHADER = /* wgsl */ `
const D: u32 = ${D}u;
const NK: u32 = ${KNOTS}u;
const NB: u32 = ${KNOTS - 4}u;
const NT: u32 = ${(D * (D + 1)) / 2}u;
const DEG: u32 = 3u;

struct Uniforms { age: f32, sexCol: u32, nVox: u32, pad: u32 }

@group(0) @binding(0) var<uniform> u: Uniforms;
@group(0) @binding(1) var<storage, read> m: array<f32>;
// Cholesky factor of A, lower triangle row-major (row r starts at r(r+1)/2).
@group(0) @binding(2) var<storage, read> lTri: array<f32>;
@group(0) @binding(3) var<storage, read> hyp: array<f32>;
@group(0) @binding(4) var<storage, read> knots: array<f32>;
// in_m, in_s, out_m, out_s, age_lo, age_hi
@group(0) @binding(5) var<storage, read> scal: array<f32>;
@group(0) @binding(6) var<storage, read> beMap: array<i32>;
@group(0) @binding(7) var<storage, read> y: array<f32>;
@group(0) @binding(8) var<storage, read_write> z: array<f32>;

@compute @workgroup_size(${WORKGROUP})
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let v = gid.x;
  if (v >= u.nVox) { return; }
  let value = y[v];
  if (value == 0.0) { z[v] = 0.0; return; }

  let age = clamp(u.age, scal[6u * v + 4u], scal[6u * v + 5u]);
  let x = (age - scal[6u * v]) / scal[6u * v + 1u];

  // Cubic B-spline basis (de Boor), end pieces extended.
  let ko = v * NK;
  var i = DEG;
  loop {
    if (i >= NB - 1u || knots[ko + i + 1u] > x) { break; }
    i = i + 1u;
  }
  var basis: array<f32, 4>;
  var left: array<f32, 4>;
  var right: array<f32, 4>;
  basis[0] = 1.0;
  for (var j = 1u; j <= DEG; j = j + 1u) {
    left[j] = x - knots[ko + i + 1u - j];
    right[j] = knots[ko + i + j] - x;
    var saved = 0.0;
    for (var r = 0u; r < j; r = r + 1u) {
      let temp = basis[r] / (right[r + 1u] + left[j - r]);
      basis[r] = saved + right[r + 1u] * temp;
      saved = left[j - r] * temp;
    }
    basis[j] = saved;
  }

  var phi: array<f32, ${D}>;
  for (var r = 0u; r <= DEG; r = r + 1u) { phi[i - DEG + r] = basis[r]; }
  phi[NB] = 1.0;
  phi[NB + 1u + u32(beMap[2u * v + u.sexCol])] = 1.0;

  var mu = 0.0;
  for (var d = 0u; d < D; d = d + 1u) { mu = mu + phi[d] * m[v * D + d]; }

  // Φ A⁻¹ Φᵀ by forward substitution L w = Φ.
  let lo = v * NT;
  var quad = 0.0;
  for (var r = 0u; r < D; r = r + 1u) {
    let row = lo + (r * (r + 1u)) / 2u;
    var s = phi[r];
    for (var k = 0u; k < r; k = k + 1u) { s = s - lTri[row + k] * phi[k]; }
    let w = s / lTri[row + r];
    phi[r] = w;
    quad = quad + w * w;
  }

  let h = 5u * v;
  let b = exp(hyp[h + 3u]);
  let a = -hyp[h + 2u] * b;
  let beta = exp(x * hyp[h] + hyp[h + 1u]) / (b * b);
  let s2 = 1.0 / beta + quad;
  let ys = (value - scal[6u * v + 2u]) / scal[6u * v + 3u];
  z[v] = (sinh(b * asinh(ys) - a) - mu) / sqrt(s2);
}
`;

/** Whether the shader handles this marker's design; other stores fall back to the CPU. */
export function supportsDesign(params: MarkerParams): boolean {
  return params.nKnots === KNOTS && params.beLevelColumns.length === BE_LEVELS;
}

/** Float64 Cholesky factors of every voxel's A, packed lower-triangular as float32. */
export function packCholesky(params: MarkerParams): Float32Array {
  const nTri = (D * (D + 1)) / 2;
  const out = new Float32Array(params.nVox * nTri);
  const L = new Float64Array(D * D);
  for (let v = 0; v < params.nVox; v++) {
    if (!params.hasModel[v]) continue;
    choleskyLower(params.aTriu, v * nTri, D, L);
    for (let r = 0, t = v * nTri; r < D; r++) for (let c = 0; c <= r; c++, t++) out[t] = L[r * D + c];
  }
  return out;
}

function packScalers(params: MarkerParams): Float32Array {
  const out = new Float32Array(6 * params.nVox);
  for (let v = 0; v < params.nVox; v++) {
    out[6 * v] = params.inscaler[4 * v];
    out[6 * v + 1] = params.inscaler[4 * v + 1];
    out[6 * v + 2] = params.outscaler[4 * v];
    out[6 * v + 3] = params.outscaler[4 * v + 1];
    out[6 * v + 4] = params.covRanges[2 * v];
    out[6 * v + 5] = params.covRanges[2 * v + 1];
  }
  return out;
}

export interface GpuZmap {
  zmap(params: MarkerParams, covariates: Covariates, y: ArrayLike<number>, signal?: AbortSignal): Promise<ZmapResult>;
  /** Why the device is unusable (lost), or null while it works. */
  lostReason(): string | null;
  dispose(): void;
}

export async function createGpuZmap(gpu: GPU | undefined = globalThis.navigator?.gpu): Promise<GpuZmap> {
  if (!gpu) throw new Error("WebGPU is not available in this browser");
  const adapter = await gpu.requestAdapter();
  if (!adapter) throw new Error("No WebGPU adapter");
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
      maxBufferSize: adapter.limits.maxBufferSize,
    },
  });
  let lost: string | null = null;
  void device.lost.then((info) => {
    lost = info.message || `WebGPU device lost (${info.reason})`;
  });
  const module = device.createShaderModule({ code: SHADER });
  const pipeline = await device.createComputePipelineAsync({ layout: "auto", compute: { module, entryPoint: "main" } });

  function storage(data: Float32Array | Int32Array, owned: GPUBuffer[]): GPUBuffer {
    const size = Math.max(16, Math.ceil(data.byteLength / 4) * 4);
    if (size > device.limits.maxStorageBufferBindingSize) throw new Error("Marker too large for this GPU's storage buffers");
    const buffer = device.createBuffer({ size, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, mappedAtCreation: true });
    owned.push(buffer);
    new (data.constructor as Float32ArrayConstructor)(buffer.getMappedRange()).set(data as Float32Array);
    buffer.unmap();
    return buffer;
  }

  async function zmap(params: MarkerParams, covariates: Covariates, y: ArrayLike<number>, signal?: AbortSignal): Promise<ZmapResult> {
    if (lost) throw new Error(lost);
    const { nVox } = params;
    if (y.length !== nVox) throw new Error(`Expected ${nVox} values, got ${y.length}`);
    if (!supportsDesign(params)) throw new Error(`${params.marker}: the GPU path does not support this model design`);
    const column = sexColumn(params, covariates);

    // Validity on the CPU (cheap), so both adapters score the same voxels.
    const valid = new Uint8Array(nVox);
    const values = new Float32Array(nVox);
    for (let v = 0; v < nVox; v++) {
      const value = y[v];
      if (!params.hasModel[v] || value === 0 || !Number.isFinite(value)) continue;
      valid[v] = 1;
      values[v] = value;
    }

    // Parameters are uploaded per call and freed after it: each run scores a
    // marker once, and a cache would pin ~60 MB of GPU memory per marker.
    const owned: GPUBuffer[] = [];
    device.pushErrorScope("out-of-memory");
    device.pushErrorScope("validation");
    let scopes = 2;
    const popScope = () => {
      scopes--;
      return device.popErrorScope();
    };
    try {
      const paramBuffers = [
        storage(Float32Array.from(params.m), owned),
        storage(packCholesky(params), owned),
        storage(Float32Array.from(params.hyp), owned),
        storage(Float32Array.from(params.meanKnots), owned),
        storage(packScalers(params), owned),
        storage(Int32Array.from(params.beMap), owned),
      ];
      signal?.throwIfAborted();
      const uniforms = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
      owned.push(uniforms);
      const uniformData = new ArrayBuffer(16);
      new Float32Array(uniformData, 0, 1)[0] = covariates.age;
      new Uint32Array(uniformData, 4, 2).set([column, nVox]);
      device.queue.writeBuffer(uniforms, 0, uniformData);
      const yBuffer = storage(values, owned);
      const zBuffer = device.createBuffer({ size: values.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC });
      owned.push(zBuffer);
      const readback = device.createBuffer({ size: values.byteLength, usage: GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST });
      owned.push(readback);
      const bindGroup = device.createBindGroup({
        layout: pipeline.getBindGroupLayout(0),
        entries: [
          { binding: 0, resource: { buffer: uniforms } },
          ...paramBuffers.map((buffer, k) => ({ binding: k + 1, resource: { buffer } })),
          { binding: 7, resource: { buffer: yBuffer } },
          { binding: 8, resource: { buffer: zBuffer } },
        ],
      });
      const encoder = device.createCommandEncoder();
      const pass = encoder.beginComputePass();
      pass.setPipeline(pipeline);
      pass.setBindGroup(0, bindGroup);
      pass.dispatchWorkgroups(Math.ceil(nVox / WORKGROUP));
      pass.end();
      encoder.copyBufferToBuffer(zBuffer, 0, readback, 0, values.byteLength);
      device.queue.submit([encoder.finish()]);
      const validation = await popScope();
      const memory = await popScope();
      const error = validation ?? memory;
      if (error) throw new Error(`WebGPU error: ${error.message}`);
      await readback.mapAsync(GPUMapMode.READ);
      const z32 = new Float32Array(readback.getMappedRange().slice(0));
      readback.unmap();
      if (lost) throw new Error(lost);
      signal?.throwIfAborted();

      // Same rules as the CPU kernel: non-finite z is not scored; clamping
      // counts scored voxels only.
      const z = new Float64Array(nVox);
      let clamped = 0;
      for (let v = 0; v < nVox; v++) {
        if (!valid[v]) continue;
        if (!Number.isFinite(z32[v])) {
          valid[v] = 0;
          continue;
        }
        z[v] = z32[v];
        if (covariates.age < params.covRanges[2 * v] || covariates.age > params.covRanges[2 * v + 1]) clamped++;
      }
      return { z, valid, clamped };
    } catch (error) {
      // Leave the error-scope stack balanced if we threw before popping.
      while (scopes > 0) await popScope().catch(() => null);
      throw error;
    } finally {
      for (const buffer of owned) buffer.destroy();
    }
  }

  return {
    zmap,
    lostReason: () => lost,
    dispose: () => device.destroy(),
  };
}
