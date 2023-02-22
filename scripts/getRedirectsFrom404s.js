const fs = require('fs');
const _ = require('lodash');
const readline = require('readline');
const url = require('url');
const bqResult = require('./404s.json');
const ignoreList = require('./ignore404s.json');
const existingRedirects = require('../redirects.json');

const rl = readline.createInterface({
  input: process.stdin,
  output: process.stdout,
});

const IGNORE_SET = new Set(ignoreList);

const existingRedirectMap = new Map(
  existingRedirects.map(({from, to}) => [from, to])
);

const brokenPaths = _.sortBy(
  _.uniq(
    bqResult
      .map(({context_page_url}) => {
        const {pathname} = url.parse(context_page_url);
        const pathnameWithoutTrailingSlash =
          pathname !== `/` && pathname.endsWith(`/`)
            ? pathname.slice(0, -1)
            : pathname;
        return pathnameWithoutTrailingSlash;
      })
      .filter(path => !IGNORE_SET.has(path) && !existingRedirectMap.has(path))
  )
);

(async () => {
  const newlyAddedRedirects = [];

  for (const path of brokenPaths) {
    const redirectTo = await prompt(
      `Enter redirect for ${path} and press Enter (just press Enter to ignore): `
    );
    if (!redirectTo) {
      ignoreList.push(path);
      continue;
    }
    newlyAddedRedirects.push({
      from: path,
      to: redirectTo,
    });
  }
  rl.close();

  fs.writeFileSync(
    `${__dirname}/ignore404s.json`,
    JSON.stringify(_.sortBy(ignoreList), null, 2)
  );

  if (newlyAddedRedirects.length === 0) {
    console.log(`No new redirects added`);
    return;
  }

  const newRedirects = _.sortBy(
    [...existingRedirects, ...newlyAddedRedirects],
    r => r.from
  );

  fs.writeFileSync(`redirects.json`, JSON.stringify(newRedirects, null, 2));
})();

function prompt(query) {
  return new Promise(resolve => rl.question(query, resolve));
}
