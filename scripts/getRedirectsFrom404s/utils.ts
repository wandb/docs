import _ from 'lodash';

import type {Redirect} from './lib';

export function sortRedirects(redirects: Redirect[]): Redirect[] {
  return _.sortBy(redirects, r => r.from);
}

export function sortSuggestionPrefixes(prefixes: Redirect[]): Redirect[] {
  return _.sortBy(
    _.sortBy(prefixes, r => r.from),

    // Longer paths should take precedence over shorter paths
    r => -r.from.split('/').length
  );
}

export function getSegmentsFromPath(path: string): string[] {
  return killLeadingSlash(path).split('/');
}

export function getMaxFromSegmentCount(redirects: Redirect[]): number {
  const fromSegmentCounts = redirects.map(
    r => getSegmentsFromPath(r.from).length
  );
  return Math.max(...fromSegmentCounts);
}

export function truncateToNSegments(path: string, n: number): string {
  const segments = getSegmentsFromPath(path);
  return `/${segments.slice(0, n).join('/')}`;
}

export function killLeadingSlash(path: string): string {
  if (path.startsWith(`/`)) {
    return path.slice(1);
  }
  return path;
}

export function isRelativeRedirect(r: Redirect): boolean {
  return !r.from.startsWith(`http`) && !r.to.startsWith(`http`);
}

export function getRedirectSuffix(prefix: string, path: string): string | null {
  if (!isPrefix(prefix, path)) {
    return null;
  }
  return path.slice(prefix.length);
}

export function isPrefix(prefix: string, path: string): boolean {
  const prefixSegments = getSegmentsFromPath(prefix);
  const pathSegments = getSegmentsFromPath(path);
  for (let i = 0; i < prefixSegments.length; i++) {
    if (prefixSegments[i] !== pathSegments[i]) {
      return false;
    }
  }
  return true;
}
