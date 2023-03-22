export function stringify(x: any, format = false): string {
  return JSON.stringify(x, null, format ? 2 : 0);
}

export function log(x: any, format = false): void {
  console.log(typeof x === `string` ? x : stringify(x, format));
}
