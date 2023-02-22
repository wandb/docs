const fs = require('fs');
const _ = require('lodash');
const url = require('url');
const bqResult = require('./404s.json');
const existingRedirects = require('../redirects.json');

const existingRedirectMap = new Map(
  existingRedirects.map(({from, to}) => [from, to])
);

const brokenPaths = _.sortBy(
  _.uniq(
    bqResult
      .map(({context_page_url}) => {
        const {pathname} = url.parse(context_page_url);
        const pathnameWithoutTrailingSlash = pathname.endsWith(`/`)
          ? pathname.slice(0, -1)
          : pathname;
        return pathnameWithoutTrailingSlash;
      })
      .filter(path => !existingRedirectMap.has(path))
  )
);

const newlyAddedRedirects = [];

const newRedirects = _.sortBy(
  [...existingRedirects, ...newlyAddedRedirects],
  r => r.from
);

fs.writeFileSync(`redirects.json`, JSON.stringify(newRedirects, null, 2));
