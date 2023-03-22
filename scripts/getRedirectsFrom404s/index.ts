import {createInterface} from 'readline';
import yargs from 'yargs';

import {addIgnoredPath, addRedirect, loadData} from './data';
import {fetch404Paths, getSuggestionPrefixes} from './lib';
import {log, parseJSONFile, prompt, writeJSONFile} from '../utils';

const {dataFile} = yargs(process.argv.slice(2))
  .options({
    dataFile: {
      alias: 'd',
      type: 'string',
      description: 'Path to the data JSON file',
    },
  })
  .parseSync();

const rl = createInterface({
  input: process.stdin,
  output: process.stdout,
});

let data = loadData(dataFile);

(async () => {
  log(`Fetching 404 paths...`);
  const brokenPaths = await fetch404Paths();

  for (const path of brokenPaths) {
    if (data.encounteredPaths.has(path)) {
      continue;
    }

    const redirectTo = await prompt(
      rl,
      `Enter redirect for ${path} and press Enter (just press Enter to ignore): `
    );
    if (redirectTo) {
      data = addRedirect(
        data,
        {
          from: path,
          to: redirectTo,
        },
        dataFile
      );
    } else {
      data = addIgnoredPath(data, path, dataFile);
    }
  }

  rl.close();
})();
