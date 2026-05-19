export function audible(
  name: string,
  soloed: ReadonlySet<string>,
  muted: ReadonlySet<string>
): boolean {
  if (muted.has(name)) return false;
  if (soloed.size > 0 && !soloed.has(name)) return false;
  return true;
}
