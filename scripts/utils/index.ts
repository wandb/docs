import fs from 'fs';
import path from 'path';
import type {Interface as RLInterface} from 'readline';

export * from './data';

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

export function prompt(rl: RLInterface, query: string): Promise<string> {
  return new Promise(resolve => rl.question(`\n${query}`, resolve));
}

export async function promptChoice<T extends readonly string[]>(
  rl: RLInterface,
  query: string,
  choices: T
): Promise<T[number]> {
  const choiceSet = new Set(choices);
  const choicesStr = choices.join(`/`);
  while (true) {
    const answer = await prompt(rl, `${query} (${choicesStr}): `);
    if (choiceSet.has(answer)) {
      return answer;
    }
    log(`${answer} is not one of the available choices: ${choicesStr}`);
  }
}

type DirectoryAndExtension = {
  directoryPath: string;
  fileExtension: string;
};

export function findFilesWithExtension(
  directoryPath: string,
  fileExtension: string
): string[] {
  const filesWithExtension: string[] = [];

  function traverseDirectory({
    directoryPath,
    fileExtension,
  }: DirectoryAndExtension): void {
    const files = fs.readdirSync(directoryPath);

    files.forEach(file => {
      const fullPath = path.join(directoryPath, file);
      const fileStat = fs.lstatSync(fullPath);

      if (fileStat.isDirectory()) {
        traverseDirectory({directoryPath: fullPath, fileExtension});
      } else if (path.extname(fullPath) === fileExtension) {
        filesWithExtension.push(fullPath);
      }
    });
  }

  traverseDirectory({directoryPath, fileExtension});
  return filesWithExtension;
}
