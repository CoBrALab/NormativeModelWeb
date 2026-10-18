// The z-map viewer: the z-map alone on white, in the CET-D1 diverging colormap
// at ±3, saturating, as in proc_nm's figures
// (voxel_nm/Analyses/norm_zscores_common/norm_zscores_common_6_viz.R, zscore_png).
import NiiVue, { MULTIPLANAR_TYPE, SHOW_RENDER, SLICE_TYPE } from "@niivue/niivue";
import { Z_DISPLAY_LIMIT } from "./config.ts";
// CET-D1 (Kovesi's diverging_bwr_40-95_c42), 256 entries, from holoviz/colorcet
// colorcet/__init__.py `diverging_bwr_40_95_c42`; cetcolor's cet_pal(256, "d1") in R.
import cetD1 from "./colormaps/cet-d1.json";

export const ZMAP_COLORMAP = "cet-d1";

export interface ZViewer {
  attach(): Promise<void>;
  /** Show one z-map (a displayNifti copy). */
  show(zmap: File): Promise<void>;
  setView(view: "multiplanar" | "axial" | "coronal" | "sagittal"): void;
}

export function createZViewer(canvasId: string, onLocation: (text: string) => void): ZViewer {
  // WebGL2: slices of one small volume need nothing more, and WebGPU stays free for the predictor.
  const nv = new NiiVue({ backend: "webgl2", isDragDropEnabled: false, backgroundColor: [1, 1, 1, 1] });
  let attached = false;

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
      nv.createExtensionContext().on("locationChange", (event: { detail: { string?: string } }) => onLocation(event.detail.string ?? ""));
      attached = true;
    },
    async show(zmap) {
      // The colormap's transparency applies to overlays only, so the z-map is
      // loaded twice: an invisible base layer that sets the grid, then the
      // coloured overlay.
      await nv.loadVolumes([{ url: zmap, name: zmap.name, opacity: 0, isColorbarVisible: false }, {
        url: zmap,
        name: zmap.name,
        colormap: ZMAP_COLORMAP,
        calMin: -Z_DISPLAY_LIMIT - (2 * Z_DISPLAY_LIMIT) / 255,
        calMax: Z_DISPLAY_LIMIT,
        isTransparentBelowCalMin: true,
        opacity: 1,
      }]);
    },
    setView(view) {
      nv.sliceType = { multiplanar: SLICE_TYPE.MULTIPLANAR, axial: SLICE_TYPE.AXIAL, coronal: SLICE_TYPE.CORONAL, sagittal: SLICE_TYPE.SAGITTAL }[view];
      nv.drawScene();
    },
  };
}
