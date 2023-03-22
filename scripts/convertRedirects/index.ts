import _ from 'lodash';

import {
  convert,
  convertNew,
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
log(`Old conversion: ${REDIRECTS.length} -> ${convertedRedirects.length}`);
logConversionErrors(conversionErrors);

const convertedRedirectsNew = convertNew(REDIRECTS);
const conversionErrorsNew = ensureProperRedirectConversion(
  REDIRECTS,
  convertedRedirectsNew
);
log(`New conversion: ${REDIRECTS.length} -> ${convertedRedirectsNew.length}`);
logConversionErrors(conversionErrorsNew);
