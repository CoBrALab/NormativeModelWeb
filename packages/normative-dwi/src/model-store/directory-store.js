// A zarrita store over a folder: one the user picked in the browser (File
// System Access API, or <input webkitdirectory>), or a local path under Node
// (node.js). Ranged reads slice the File, so a shard index or inner chunk is
// read without loading the whole shard.

function isNotFound(error) {
  const name = error?.name;
  return name === "NotFoundError" || name === "TypeMismatchError";
}

export function directoryStore(root) {
  async function file(key) {
    const parts = key.split("/").filter(Boolean);
    const name = parts.pop();
    if (!name) return undefined;
    try {
      let dir = root;
      for (const part of parts) dir = await dir.getDirectoryHandle(part);
      return await (await dir.getFileHandle(name)).getFile();
    } catch (error) {
      if (isNotFound(error)) return undefined;
      throw error;
    }
  }
  return {
    async get(key) {
      const blob = await file(key);
      return blob && new Uint8Array(await blob.arrayBuffer());
    },
    async getRange(key, range) {
      const blob = await file(key);
      if (!blob) return undefined;
      const part = "suffixLength" in range
        ? blob.slice(Math.max(0, blob.size - range.suffixLength))
        : blob.slice(range.offset, range.offset + range.length);
      return new Uint8Array(await part.arrayBuffer());
    },
  };
}

/** A DirectoryLike over files from <input webkitdirectory> (paths in webkitRelativePath). */
export function filesDirectory(files) {
  const byPath = new Map();
  for (const file of files) byPath.set(file.webkitRelativePath || file.name, file);
  const rootName = (files[0]?.webkitRelativePath || "").split("/")[0];
  const prefix = rootName ? `${rootName}/` : "";
  const missing = () => Object.assign(new Error("not found"), { name: "NotFoundError" });
  const at = (path) => ({
    name: path.split("/").filter(Boolean).pop() ?? rootName,
    async getDirectoryHandle(name) {
      const sub = `${path}${name}/`;
      for (const key of byPath.keys()) if (key.startsWith(sub)) return at(sub);
      throw missing();
    },
    async getFileHandle(name) {
      const file = byPath.get(`${path}${name}`);
      if (!file) throw missing();
      return { getFile: async () => file };
    },
  });
  return at(prefix);
}
