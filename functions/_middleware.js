/**
 * Cloudflare Pages middleware to handle trailing slash redirects
 * 
 * HIGHLY OPTIMIZED: Minimal processing to avoid rate limits
 * Only checks specific patterns known to have redirect issues
 */

export async function onRequest(context) {
  const { request, env, next } = context;
  
  // OPTIMIZATION: Only check GET/HEAD requests
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    return next();
  }
  
  const url = new URL(request.url);
  const pathname = url.pathname;
  
  // OPTIMIZATION: Skip if already has trailing slash
  if (pathname.endsWith('/')) {
    return next();
  }
  
  // OPTIMIZATION: Skip if has file extension (quick check)
  if (pathname.includes('.')) {
    return next();
  }
  
  // OPTIMIZATION: Only process paths with 2+ segments
  // Single segment paths like /about rarely need this fix
  const segments = pathname.split('/').filter(s => s);
  if (segments.length < 2) {
    return next();
  }
  
  // OPTIMIZATION: Only check paths that match common redirect patterns
  // Add more patterns here as needed based on your _redirects file
  const needsCheck = 
    pathname.startsWith('/wandb/') ||
    pathname.startsWith('/library/') ||
    pathname.startsWith('/guides/') ||
    pathname.startsWith('/ref/') ||
    pathname.startsWith('/sweeps/') ||
    pathname.startsWith('/artifacts/') ||
    pathname.startsWith('/frameworks/') ||
    pathname.startsWith('/company/') ||
    pathname.startsWith('/tutorials/');
  
  if (!needsCheck) {
    return next();
  }
  
  // At this point, we have a high-probability redirect candidate
  // Redirect to path with trailing slash
  const redirectUrl = pathname + '/' + url.search + url.hash;
  
  return new Response(null, {
    status: 301,
    headers: {
      'Location': redirectUrl,
      'Cache-Control': 'public, max-age=3600'
    }
  });
}