import {createInterface} from 'readline';
import yargs from 'yargs';

import {addIgnoredPath, addRedirect, loadData} from './data';
import {fetch404Paths, getSuggestionPrefixes, Redirect} from './lib';
import {
  log,
  parseJSONFile,
  prompt,
  promptChoice,
  writeJSONFile,
} from '../utils';
import {getRedirectSuffix} from './utils';

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

  PathLoop: for (const path of brokenPaths) {
    if (data.encounteredPaths.has(path)) {
      continue;
    }

    for (const suggestionPrefix of data.suggestionPrefixes) {
      const suggestionTaken = await handleSuggestion(path, suggestionPrefix);
      if (suggestionTaken) {
        continue PathLoop;
      }
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

async function handleSuggestion(
  path: string,
  {from: prefixFrom, to: prefixTo}: Redirect
): Promise<boolean> {
  const suffix = getRedirectSuffix(path, prefixFrom);
  if (suffix == null) {
    return false;
  }

  const redirectTo = `${prefixTo}${suffix}`;
  const promptQuery = `SUGGESTION: Redirect ${path} to ${redirectTo}?`;
  const promptChoices = [`y`, `n`] as const;

  const suggestionAccepted =
    (await promptChoice(rl, promptQuery, promptChoices)) === `y`;
  if (!suggestionAccepted) {
    return false;
  }

  data = addRedirect(
    data,
    {
      from: path,
      to: redirectTo,
    },
    dataFile
  );
  return true;
}
