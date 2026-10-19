// The z-map viewer, in the CET-D1 diverging colormap at ±3 (user-set), saturating, as in
// proc_nm's figures (voxel_nm/Analyses/norm_zscores_common/norm_zscores_common_6_viz.R,
// zscore_png). Alone, the z-map is drawn opaque on black (proc_nm draws on
// white; on black, CET-D1's near-white middle stays distinct from unscored
// voxels). Over a structural background image, each voxel's opacity is
// min(1, |z|/limit) by default, so near-normal voxels let the anatomy through, or
// one uniform opacity chosen by the user.
import NiiVue, { MULTIPLANAR_TYPE, SHOW_RENDER, SLICE_TYPE } from "@niivue/niivue";
import { Z_DISPLAY_LIMIT } from "./config.ts";
// CET-D1 (Kovesi's diverging_bwr_40-95_c42), 256 entries, from holoviz/colorcet
// colorcet/__init__.py `diverging_bwr_40_95_c42`; cetcolor's cet_pal(256, "d1") in R.
import cetD1 from "./colormaps/cet-d1.json";

export const ZMAP_COLORMAP = "cet-d1";

export interface ZLayers {
  /** Display copy with unscored voxels below the range (displayNifti's default), for the z-map alone. */
  zmap?: File;
  /** Display copy with unscored voxels at 0, for the z-map over `background`. */
  overlay?: File;
  /** |z| of `overlay` (saturated, unscored 0): its opacity over `background`. */
  magnitude?: File;
  /** A structural image in the same world space, at any resolution. */
  background?: File;
  /** |z| where the colour saturates (and, from |z|, opacity is full); the copies must be saturated at it. Default Z_DISPLAY_LIMIT. */
  limit?: number;
}

/** How the z-map blends over a background: opacity from |z|, or `opacity` (0–1) everywhere. */
export interface OverlayOpacity {
  fromZ: boolean;
  opacity: number;
}

export interface ZViewer {
  attach(): Promise<void>;
  /**
   * Show a z-map alone (`zmap`), over a background (`background` and
   * `overlay`), the background alone, or (no layers) nothing. `readout` describes the crosshair's
   * world position (mm) from the true values (the display copies are
   * saturated); its text is the location line.
   */
  show(layers: ZLayers, readout: ((mm: number[]) => string) | null): Promise<void>;
  setView(view: "multiplanar" | "axial" | "coronal" | "sagittal"): void;
  /** Set how the z-map blends over a background; kept for later views. */
  setOverlayOpacity(style: OverlayOpacity): Promise<void>;
}

export function createZViewer(canvasId: string, onLocation: (text: string) => void): ZViewer {
  // WebGL2: slices of one small volume need nothing more, and WebGPU stays free for the predictor.
  const nv = new NiiVue({ backend: "webgl2", isDragDropEnabled: false, backgroundColor: [0, 0, 0, 1] });
  let attached = false;
  let readout: ((mm: number[]) => string) | null = null;
  let style: OverlayOpacity = { fromZ: true, opacity: 1 };
  /** Whether the z-map is shown over a background (volumes: background, z-map, |z|). */
  let layered = false;

  // NiiVue makes an overlay's colormap alpha binary, so the |z| opacity comes
  // from alpha modulation instead: the hidden |z| layer, windowed to
  // [0, limit] with exponent 1, scales each voxel's alpha by min(1, |z|/limit).
  // Unscored voxels (0 in both) are transparent. Without it, the layer's own
  // opacity applies to every scored voxel alike.
  async function applyStyle() {
    if (!layered) return;
    const [, target, modulator] = nv.volumes;
    if (!target?.id || !modulator?.id) throw new Error("the viewer did not load the z-map layers");
    await nv.setModulationImage(target.id, style.fromZ ? modulator.id : "", style.fromZ ? 1 : 0);
    await nv.setVolume(1, { opacity: style.fromZ ? 1 : style.opacity });
  }

  return {
    async attach() {
      if (attached) return;
      await nv.attachTo(canvasId);
      // Entry 0 is transparent and the range starts one LUT step below −limit:
      // unscored voxels (far below, see displayNifti) vanish while z ≤ −limit
      // still draws in CET-D1's first blue. NiiVue's isTransparentBelowCalMin
      // alone leaves them blue.
      nv.addColormap(ZMAP_COLORMAP, { ...cetD1, A: cetD1.A.map((a, i) => (i === 0 ? 0 : a)) });
      nv.multiplanarType = MULTIPLANAR_TYPE.GRID;
      nv.sliceType = SLICE_TYPE.MULTIPLANAR;
      nv.showRender = SHOW_RENDER.NEVER;
      nv.isColorbarVisible = true;
      nv.createExtensionContext().on("locationChange", (event: { detail: { mm?: number[] } }) => {
        const { mm } = event.detail;
        if (!mm || !readout) return onLocation("");
        onLocation(readout(mm.slice(0, 3)));
      });
      attached = true;
    },
    async show({ zmap, overlay, magnitude, background, limit = Z_DISPLAY_LIMIT }, describe) {
      readout = describe;
      layered = false;
      if (background) {
        // The background is the base layer: it sets the grid, and NiiVue
        // resamples the z-map onto it through the affines.
        const base = { url: background, name: background.name, colormap: "gray", opacity: 1, isColorbarVisible: false };
        if (!overlay || !magnitude) {
          await nv.loadVolumes([base]);
          return;
        }
        // Unscored voxels are 0 here, transparent in either style; the
        // sentinel colormap still draws z ≤ −limit in its first blue.
        await nv.loadVolumes([base, {
          url: overlay,
          name: overlay.name,
          colormap: ZMAP_COLORMAP,
          calMin: -limit - (2 * limit) / 255,
          calMax: limit,
          opacity: 1,
        }, { url: magnitude, name: magnitude.name, calMin: 0, calMax: limit, opacity: 0, isColorbarVisible: false }]);
        layered = true;
        await applyStyle();
        return;
      }
      if (!zmap) return nv.removeAllVolumes();
      // The colormap's transparency applies to overlays only, so the z-map is
      // loaded twice: an invisible base layer that sets the grid, then the
      // coloured overlay.
      await nv.loadVolumes([{ url: zmap, name: zmap.name, opacity: 0, isColorbarVisible: false }, {
        url: zmap,
        name: zmap.name,
        colormap: ZMAP_COLORMAP,
        calMin: -limit - (2 * limit) / 255,
        calMax: limit,
        isTransparentBelowCalMin: true,
        opacity: 1,
      }]);
    },
    setView(view) {
      nv.sliceType = { multiplanar: SLICE_TYPE.MULTIPLANAR, axial: SLICE_TYPE.AXIAL, coronal: SLICE_TYPE.CORONAL, sagittal: SLICE_TYPE.SAGITTAL }[view];
      nv.drawScene();
    },
    async setOverlayOpacity(next) {
      style = { ...next };
      await applyStyle();
    },
  };
}
