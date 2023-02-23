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
  for (const path of brokenPaths) {
    const redirectTo = await prompt(
      `Enter redirect for ${path} and press Enter (just press Enter to ignore): `
    );
    if (redirectTo) {
      addToRedirects({
        from: path,
        to: redirectTo,
      });
    } else {
      addToIgnoreList(path);
    }
  }
  rl.close();
})();

function addToRedirects(redirect) {
  const newRedirects = _.sortBy([...existingRedirects, redirect], r => r.from);
  fs.writeFileSync(`redirects.json`, JSON.stringify(newRedirects, null, 2));
}

function addToIgnoreList(path) {
  const newIgnoreList = _.sortBy([...ignoreList, path]);
  fs.writeFileSync(
    `${__dirname}/ignore404s.json`,
    JSON.stringify(newIgnoreList, null, 2)
  );
}

function prompt(query) {
  return new Promise(resolve => rl.question(query, resolve));
}
