const express = require('express');
const helmet = require('helmet');
const static = require('@laxels/serve-static');
const url = require('url');
const data = require('./data.json');

const app = express();

const isProduction = process.env.NODE_ENV === `production`;
const isDev = !isProduction;

// Force HTTPS through Strict-Transport-Security header
// See https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Strict-Transport-Security
app.use(helmet.hsts());

// Serve 301 redirects for old links in the wild
const redirectMap = new Map(data.redirects.map(({from, to}) => [from, to]));
app.use((req, res, next) => {
  const {pathname, search, hash} = url.parse(req.originalUrl);

  const pathnameWithoutTrailingSlash = pathname.endsWith(`/`)
    ? pathname.slice(0, -1)
    : pathname;

  const redirectTo = redirectMap.get(pathnameWithoutTrailingSlash);
  if (redirectTo == null) {
    next();
    return;
  }

  res.redirect(301, `${redirectTo}${search ?? ``}${hash ?? ``}`);
});

// Serve static files according to path
app.use(
  static('build', {
    transformPath,
    etag: false,
    lastModified: false,
    maxAge: 600000,
  })
);

// If no static file is found in that path, serve 404 HTML page with 404 status code
app.use((req, res, next) => {
  res.status(404);
  next();
});
app.use(
  static('build', {
    transformPath: transformPathInto404,
    etag: false,
    lastModified: false,
    maxAge: 600000,
  })
);

// Listen to the App Engine-specified port, or 8080 otherwise
const PORT = process.env.PORT || 8080;
app.listen(PORT, () => {
  console.log(`Server listening on port ${PORT}...`);
});

const STATIC_FILE_REGEX = /\/(.*\..+)$/;

// / -> /index.html
// /asset.png -> /asset.png
// /quickstart -> /quickstart/index.html
// /quickstart/ -> /quickstart/index.html
function transformPath(path) {
  if (path === `/`) {
    return `/index.html`;
  }
  if (STATIC_FILE_REGEX.test(path)) {
    return path;
  }
  if (path.endsWith(`/`)) {
    return `${path}index.html`;
  }
  return `${path}/index.html`;
}

function transformPathInto404(path) {
  return `/404.html`;
}
