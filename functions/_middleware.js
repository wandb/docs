/**
 * Cloudflare Pages middleware to handle trailing slash redirects
 * 
 * This middleware addresses the known Cloudflare Pages issue where URLs without
 * trailing slashes don't match redirect rules that include trailing slashes.
 */

export async function onRequest(context) {
  const { request, env, next } = context;
  const url = new URL(request.url);
  
  // Only process GET and HEAD requests
  if (request.method !== 'GET' && request.method !== 'HEAD') {
    return next();
  }
  
  // Skip if the path already has a trailing slash
  if (url.pathname.endsWith('/')) {
    return next();
  }
  
  // Skip if the path has a file extension
  const hasFileExtension = /\.[a-zA-Z0-9]+$/.test(url.pathname);
  if (hasFileExtension) {
    return next();
  }
  
  // Skip certain system paths
  const skipPaths = ['/robots.txt', '/sitemap.xml', '/_redirects', '/favicon.ico'];
  if (skipPaths.includes(url.pathname)) {
    return next();
  }
  
  // Skip paths with dots that might not be file extensions
  // but could be version numbers or other valid path segments
  // Only skip if the dot is followed by common file extensions
  const commonExtensions = /\.(html|htm|css|js|json|xml|txt|pdf|jpg|jpeg|png|gif|svg|ico|woff|woff2|ttf|eot)$/i;
  if (commonExtensions.test(url.pathname)) {
    return next();
  }
  
  // Create URL with trailing slash, preserving query and hash
  const urlWithSlash = new URL(request.url);
  urlWithSlash.pathname = url.pathname + '/';
  
  // Create a new request with the trailing slash
  const requestWithSlash = new Request(urlWithSlash.toString(), {
    method: request.method,
    headers: request.headers,
    redirect: 'manual' // Important: don't follow redirects
  });
  
  try {
    // Fetch with the trailing slash URL
    const response = await env.ASSETS.fetch(requestWithSlash);
    
    // If we get a redirect response (301, 302, etc.), redirect to URL with slash
    if (response.status >= 301 && response.status <= 308) {
      // The path with trailing slash matches a redirect rule
      // Redirect the browser to the URL with slash, which will then trigger the actual redirect
      // Preserve query parameters and hash
      const redirectUrl = new URL(request.url);
      redirectUrl.pathname = url.pathname + '/';
      
      return new Response(null, {
        status: 301,
        headers: {
          'Location': redirectUrl.pathname + redirectUrl.search + redirectUrl.hash,
          'Cache-Control': 'public, max-age=3600'
        }
      });
    }
    
    // If the response is successful (200), it means the trailing slash version exists
    // In this case, we should also redirect to maintain consistency
    if (response.status === 200) {
      const redirectUrl = new URL(request.url);
      redirectUrl.pathname = url.pathname + '/';
      
      return new Response(null, {
        status: 301,
        headers: {
          'Location': redirectUrl.pathname + redirectUrl.search + redirectUrl.hash,
          'Cache-Control': 'public, max-age=3600'
        }
      });
    }
  } catch (error) {
    // Log error but continue with original request
    console.error('Trailing slash middleware error:', error);
  }
  
  // No redirect found, continue with the original request
  return next();
}