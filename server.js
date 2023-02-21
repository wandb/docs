const express = require('express');
const helmet = require('helmet');
const static = require('@laxels/serve-static');
const app = express();

const isProduction = process.env.NODE_ENV === `production`;
const isDev = !isProduction;

app.use(helmet.hsts());

app.use(
  static('build', {
    transformPath,
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
