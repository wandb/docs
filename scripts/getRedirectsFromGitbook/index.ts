import fs from 'fs';
import {load} from 'js-yaml';
import yargs from 'yargs';

import {addRedirects, loadData} from './data';
import {Redirect, isNotNullOrUndefined} from '../utils';

const {dataFile, redirectsFile} = yargs(process.argv.slice(2))
  .options({
    dataFile: {
      alias: 'd',
      type: 'string',
      description: 'Path to the data JSON file',
    },
  })
  .options({
    redirectsFile: {
      alias: 'r',
      type: 'string',
      description: 'Path to the redirects YAML file',
      demandOption: true,
    },
  })
  .parseSync();

let data = loadData(dataFile);

type RedirectsYAML = {
  redirects: Record<string, string>;
};

(async () => {
  const yaml = load(fs.readFileSync(redirectsFile).toString()) as RedirectsYAML;

  const redirects = getRedirectsFromObject(yaml.redirects);

  console.log(`Will apply ${redirects.length} redirects`);

  for (const r of redirects) {
    console.log(r);
  }

  data = addRedirects(data, redirects, dataFile);
})();

function getRedirectsFromObject(obj: Record<string, string>): Redirect[] {
  return Object.entries(obj)
    .map(([fromUnformatted, toUnformatted]) => {
      const from = formatFrom(fromUnformatted);
      const to = formatTo(toUnformatted);

      if (redirectExists({from, to})) {
        // console.log(`Redirect from ${from} to ${to} already exists`);
        return null;
      }

      const chainWith = redirectWillChainWith({from, to});
      if (chainWith != null) {
        // console.log(`Redirect from ${from} to ${to} would result in a chain`);
        return {
          from,
          to: chainWith.to,
        };
      }

      return {
        from,
        to,
      };
    })
    .filter(isNotNullOrUndefined);
}

function redirectExists({from}: Redirect): boolean {
  return data.redirects.some(r => r.from === from);
}

function redirectWillChainWith({to}: Redirect): Redirect | null {
  return data.redirects.find(r => r.from === to) ?? null;
}

function formatFrom(from: string): string {
  return `/${from}`;
}

function formatTo(to: string): string {
  return to
    .replace(/^\./, ``)
    .replace(/\.md$/, '')
    .replace(/\/README$/, ``);
}
