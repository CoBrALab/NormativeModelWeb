import type { AsyncReadable } from "zarrita";

/** The parts of FileSystemDirectoryHandle this store uses (lets Node and tests pass their own). */
export interface DirectoryLike {
  name: string;
  getDirectoryHandle(name: string): Promise<DirectoryLike>;
  getFileHandle(name: string): Promise<{ getFile(): Promise<Blob> }>;
}

export function directoryStore(root: DirectoryLike): AsyncReadable;
/** A DirectoryLike over files from <input webkitdirectory> (paths in webkitRelativePath). */
export function filesDirectory(files: readonly File[]): DirectoryLike;
