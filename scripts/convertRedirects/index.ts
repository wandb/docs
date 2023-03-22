import _ from 'lodash';

import {
  convert,
  ensureProperRedirectConversion,
  logConversionErrors,
} from './lib';
import {log} from '../utils';

import REDIRECTS from '../../redirects.json';

const convertedRedirects = convert(REDIRECTS);
const conversionErrors = ensureProperRedirectConversion(
  REDIRECTS,
  convertedRedirects
);
log(`${REDIRECTS.length} -> ${convertedRedirects.length}`);
logConversionErrors(conversionErrors);
