/** 取路径最后一段（兼容 Windows 反斜杠与 POSIX 正斜杠）。 */
export function basename(path: string): string {
  const normalized = path.replace(/\\/g, "/");
  return normalized.slice(normalized.lastIndexOf("/") + 1);
}
