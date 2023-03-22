import yargs from 'yargs';

import {
  convert,
  ensureProperRedirectConversion,
  logConversionErrors,
} from './lib';
import {log, parseJSONFile, writeJSONFile} from '../utils';

const {file, out} = yargs(process.argv.slice(2))
  .options({
    file: {
      alias: 'f',
      type: 'string',
      demandOption: true,
      description: 'Path to the redirects file',
    },
    out: {
      alias: 'o',
      type: 'string',
      demandOption: true,
      description: 'Path to the converted redirects file',
    },
  })
  .parseSync();

const redirects = parseJSONFile(file);

const convertedRedirects = convert(redirects);
const conversionErrors = ensureProperRedirectConversion(
  redirects,
  convertedRedirects
);
log(`${redirects.length} -> ${convertedRedirects.length}`);
logConversionErrors(conversionErrors);

writeJSONFile(out, convertedRedirects);
