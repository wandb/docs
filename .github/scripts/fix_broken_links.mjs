#!/usr/bin/env node
/*
Parses lychee JSON output at ./lychee/out.json, attempts auto-fixes for redirectable links,
updates files in-place, and writes a report to .github/lychee-report.md for manual follow-up.
*/
import fs from 'fs';
import path from 'path';
import { setTimeout as delay } from 'timers/promises';

const LYCHEE_JSON = './lychee/out.json';
const REPORT_MD = '.github/lychee-report.md';

function log(msg) {
	process.stdout.write(`${msg}\n`);
}

function readJsonSafe(filePath) {
	if (!fs.existsSync(filePath)) return null;
	const text = fs.readFileSync(filePath, 'utf8');
	try {
		return JSON.parse(text);
	} catch (e) {
		log(`Failed to parse JSON at ${filePath}: ${e.message}`);
		return null;
	}
}

function normalizeUrl(u) {
	try {
		const url = new URL(u);
		return url.toString();
	} catch {
		return u;
	}
}

function unique(array) {
	return Array.from(new Set(array));
}

function groupByUrl(entries) {
	const map = new Map();
	for (const e of entries) {
		const url = normalizeUrl(e.url);
		if (!map.has(url)) map.set(url, []);
		map.get(url).push(...e.sources);
	}
	return map;
}

function parseLychee(json) {
	// Support multiple lychee JSON shapes
	const broken = [];
	if (!json) return broken;

	if (Array.isArray(json)) {
		// Possibly an array of response objects
		for (const item of json) {
			if (!item) continue;
			const status = item.status || item.result || item.error || '';
			const ok = String(status).toLowerCase().includes('ok') || String(status).toLowerCase() === '200';
			if (!ok) {
				const sources = (item.sources || item.locations || []).map(s => ({ file: s.file || s.path || s.input || s.source || 'unknown', line: s.line || 0 }));
				broken.push({ url: item.uri || item.url || item.link || item.target || 'unknown', sources });
			}
		}
	} else if (json.fail_map && typeof json.fail_map === 'object') {
		for (const [url, arr] of Object.entries(json.fail_map)) {
			const sources = (arr || []).map(s => ({ file: s.input || s.file || s.source || 'unknown', line: s.line || 0 }));
			broken.push({ url, sources });
		}
	} else if (json.error_map && typeof json.error_map === 'object') {
		// Handle newer lychee output format
		for (const [file, errors] of Object.entries(json.error_map)) {
			for (const error of errors) {
				if (error.url && error.status && error.status.code !== 200) {
					broken.push({ 
						url: error.url, 
						sources: [{ file, line: 0 }]
					});
				}
			}
		}
	} else if (json.responses && Array.isArray(json.responses)) {
		for (const r of json.responses) {
			const status = r.status?.toString() || '';
			const ok = status === '200' || status.toLowerCase() === 'ok' || r.ok === true;
			if (!ok) {
				const sources = (r.sources || []).map(s => ({ file: s.file || s.input || 'unknown', line: s.line || 0 }));
				broken.push({ url: r.uri || r.url || 'unknown', sources });
			}
		}
	}
	return broken.filter(b => b.url && !b.url.startsWith('mailto:'));
}

async function probeRedirect(url) {
	const controllers = [];
	function withTimeout(ms) {
		const c = new AbortController();
		controllers.push(c);
		setTimeout(() => c.abort(), ms).unref?.();
		return c.signal;
	}
	try {
		let res = await fetch(url, { method: 'HEAD', redirect: 'follow', signal: withTimeout(10000) });
		if (!res.ok || res.status >= 400) {
			// Some servers don't support HEAD
			res = await fetch(url, { method: 'GET', redirect: 'follow', signal: withTimeout(10000) });
		}
		return { ok: res.ok, finalUrl: res.url || url, status: res.status };
	} catch (e) {
		return { ok: false, finalUrl: url, error: String(e) };
	} finally {
		for (const c of controllers) try { c.abort(); } catch {}
	}
}

function candidateVariants(url) {
	const variants = [];
	try {
		const u = new URL(url);
		// Try https
		if (u.protocol === 'http:') {
			u.protocol = 'https:';
			variants.push(u.toString());
		}
		// Try removing trailing slash
		if (u.pathname.endsWith('/') && (u.pathname !== '/')) {
			u.pathname = u.pathname.replace(/\/+$/, '');
			variants.push(u.toString());
		}
		// Try adding trailing slash (common for dirs)
		if (!u.pathname.endsWith('/')) {
			u.pathname = u.pathname + '/';
			variants.push(u.toString());
		}
		// Try adding/removing www
		if (u.hostname.startsWith('www.')) {
			u.hostname = u.hostname.replace(/^www\./, '');
			variants.push(u.toString());
		} else {
			u.hostname = 'www.' + u.hostname;
			variants.push(u.toString());
		}
	} catch {
		// ignore parse errors
	}
	return unique(variants).filter(v => v !== url);
}

async function tryAutoFix(url) {
	// First, follow redirects for the original URL
	let p = await probeRedirect(url);
	if (p.ok && normalizeUrl(p.finalUrl) !== normalizeUrl(url)) {
		return { fixed: true, replacement: normalizeUrl(p.finalUrl), method: 'redirect-follow' };
	}
	// Try common variants
	const variants = candidateVariants(url);
	for (const v of variants) {
		const pr = await probeRedirect(v);
		if (pr.ok) {
			return { fixed: true, replacement: normalizeUrl(pr.finalUrl), method: 'variant' };
		}
		// tiny backoff to be gentle
		await delay(100);
	}
	return { fixed: false };
}

function isGeneratedRefDoc(filePath) {
	// Check if file is in generated reference documentation directories
	// These are auto-generated and should not be auto-fixed
	const generatedPaths = [
		'/ref/js/',
		'/ref/python/'
	];
	
	const normalizedPath = filePath.replace(/\\/g, '/');
	return generatedPaths.some(genPath => normalizedPath.includes(genPath));
}

function replaceInFile(filePath, replacements) {
	// Skip generated reference documentation
	if (isGeneratedRefDoc(filePath)) {
		log(`Skipping generated reference doc: ${filePath}`);
		return false;
	}
	
	let content = fs.readFileSync(filePath, 'utf8');
	let changed = false;
	for (const [from, to] of replacements) {
		if (from === to) continue;
		const froms = generateFromCandidates(from);
		for (const f of froms) {
			const esc = f.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
			const re = new RegExp(esc, 'g');
			if (re.test(content)) {
				content = content.replace(re, to);
				changed = true;
			}
		}
	}
	if (changed) {
		fs.writeFileSync(filePath, content, 'utf8');
	}
	return changed;
}

function generateFromCandidates(fromUrl) {
	const set = new Set();
	set.add(fromUrl);
	// Pure string variants for trailing slash
	if (fromUrl.endsWith('/')) {
		set.add(fromUrl.replace(/\/+$/, ''));
	} else {
		set.add(fromUrl + '/');
	}
	try {
		const u = new URL(fromUrl);
		// add/remove trailing slash variants via URL API as well
		if (u.pathname.endsWith('/') && u.pathname !== '/') {
			const noSlash = new URL(fromUrl);
			noSlash.pathname = noSlash.pathname.replace(/\/+$/, '');
			set.add(noSlash.toString());
		} else {
			const withSlash = new URL(fromUrl);
			withSlash.pathname = withSlash.pathname + '/';
			set.add(withSlash.toString());
		}
	} catch {}
	return Array.from(set);
}

function ensureDir(dir) {
	if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
}

async function main() {
	const json = readJsonSafe(LYCHEE_JSON);
	if (!json) {
		log('No lychee JSON found; skipping fixes.');
		process.exit(0);
	}
	const brokenEntries = parseLychee(json);
	if (brokenEntries.length === 0) {
		log('No broken links detected.');
		// Remove old report if present
		if (fs.existsSync(REPORT_MD)) fs.unlinkSync(REPORT_MD);
		return;
	}

	const byUrl = groupByUrl(brokenEntries);
	const fixes = new Map(); // url -> replacement
	const manual = new Map(); // url -> sources

	let processed = 0;
	for (const [url, sources] of byUrl.entries()) {
		processed++;
		log(`Processing (${processed}/${byUrl.size}): ${url}`);
		const res = await tryAutoFix(url);
		if (res.fixed && res.replacement) {
			fixes.set(url, res.replacement);
		} else {
			manual.set(url, sources);
		}
	}

	// Apply fixes grouped by file
	const fileToRepls = new Map();
	for (const [url, repl] of fixes.entries()) {
		const sources = byUrl.get(url) || [];
		for (const s of sources) {
			const f = s.file;
			if (!fileToRepls.has(f)) fileToRepls.set(f, []);
			fileToRepls.get(f).push([url, repl]);
		}
	}

	let filesChanged = 0;
	for (const [file, repls] of fileToRepls.entries()) {
		try {
			if (fs.existsSync(file)) {
				const changed = replaceInFile(file, repls);
				if (changed) filesChanged++;
			} else {
				log(`File not found: ${file}`);
			}
		} catch (e) {
			log(`Failed to update ${file}: ${e.message}`);
		}
	}

	// Generate report
	ensureDir(path.dirname(REPORT_MD));
	const lines = [];
	lines.push('# Link Check Report');
	lines.push('');
	lines.push(`- Auto-fixed URLs: ${fixes.size}`);
	lines.push(`- Files changed: ${filesChanged}`);
	lines.push(`- Remaining manual follow-ups: ${manual.size}`);
	lines.push('');
	if (fixes.size > 0) {
		lines.push('## Auto-fixed');
		for (const [from, to] of fixes.entries()) {
			lines.push(`- ${from} → ${to}`);
		}
		lines.push('');
	}
	if (manual.size > 0) {
		lines.push('## Manual follow-up needed');
		for (const [url, sources] of manual.entries()) {
			const where = unique(sources.map(s => `${s.file}${s.line ? `:${s.line}` : ''}`)).slice(0, 10);
			lines.push(`- [ ] ${url}`);
			for (const loc of where) {
				lines.push(`  - ${loc}`);
			}
		}
		lines.push('');
	}
	fs.writeFileSync(REPORT_MD, lines.join('\n'), 'utf8');
	log(`Wrote report to ${REPORT_MD}`);
}

main().catch(e => {
	console.error(e);
	process.exit(0);
});