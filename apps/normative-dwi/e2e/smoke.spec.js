// Real browser smoke test against the built app (vite preview): boot, shared
// shell, cross-origin isolation, and the controls' initial state.
import { test, expect } from "@playwright/test";

test("app boots with the workspace and a disabled run button", async ({ page }) => {
  await page.goto("/");
  await expect(page.locator("#viewer")).toBeVisible();
  await expect(page.locator("#runButton")).toBeDisabled();
  await expect(page.locator("#statusText")).toHaveText("Drop maps to begin");
});

test("shared app bar owns information actions and theme", async ({ page }) => {
  await page.goto("/");
  const bar = page.locator(".nd-app-bar:visible");
  await expect(bar).toHaveCount(1);
  await expect(page.locator("#controls > #aboutBtn")).toBeHidden();
  await bar.getByRole("button", { name: "About", exact: true }).click();
  await expect(page.locator("#infoDialog")).toBeVisible();
  await expect(page.locator("#infoDialog")).toContainText("harmonised to UK Biobank");
  await page.locator("#infoDialog").getByRole("button", { name: "Close" }).click();
});

test("page is cross-origin isolated (COOP/COEP active)", async ({ page }) => {
  await page.goto("/");
  expect(await page.evaluate(() => self.crossOriginIsolated === true)).toBe(true);
});

test("a map on the wrong grid is named in the status line", async ({ page }) => {
  await page.goto("/");
  await page.locator("#imageInput").setInputFiles({ name: "not-a-nifti_FA.nii", mimeType: "application/octet-stream", buffer: Buffer.from("nope") });
  await expect(page.locator("#statusText")).toContainText("not-a-nifti_FA.nii");
  await expect(page.locator("#runButton")).toBeDisabled();
});
