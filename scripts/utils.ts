import fs from 'fs';

export function stringify(x: any, format = false): string {
  return JSON.stringify(x, null, format ? 2 : 0);
}

export function log(x: any, format = false): void {
  console.log(typeof x === `string` ? x : stringify(x, format));
}

export function parseJSONFile(fileName: string): any {
  return JSON.parse(fs.readFileSync(fileName).toString());
}

export function writeJSONFile(fileName: string, x: any): void {
  fs.writeFileSync(fileName, stringify(x, true));
}

export function isNotNullOrUndefined<T>(x: T | null | undefined): x is T {
  return x != null;
}
