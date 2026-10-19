// A minimal FileSystemDirectoryHandle over a real folder, for the local-folder store adapter.
import { openAsBlob, statSync } from "node:fs";
import { join } from "node:path";

export function notFound() {
  return Object.assign(new Error("not found"), { name: "NotFoundError" });
}

export function fsDirectory(path) {
  return {
    name: path.split("/").pop(),
    async getDirectoryHandle(name) {
      const sub = join(path, name);
      if (!statSync(sub, { throwIfNoEntry: false })?.isDirectory()) throw notFound();
      return fsDirectory(sub);
    },
    async getFileHandle(name) {
      const file = join(path, name);
      if (!statSync(file, { throwIfNoEntry: false })?.isFile()) throw notFound();
      return { getFile: () => openAsBlob(file) };
    },
  };
}
