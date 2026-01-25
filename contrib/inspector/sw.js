// Service Worker for Datasculpt Inspector
// Caches PyScript, Pyodide, and package assets for faster repeat loads

const CACHE_NAME = 'datasculpt-inspector-v4';

// Add COOP/COEP headers to enable cross-origin isolation (faster WASM)
function addIsolationHeaders(response) {
    // Only modify same-origin HTML responses
    if (!response.url.startsWith(self.location.origin)) {
        return response;
    }
    const newHeaders = new Headers(response.headers);
    newHeaders.set('Cross-Origin-Opener-Policy', 'same-origin');
    newHeaders.set('Cross-Origin-Embedder-Policy', 'credentialless');
    return new Response(response.body, {
        status: response.status,
        statusText: response.statusText,
        headers: newHeaders,
    });
}

// URLs to cache on install (app shell)
const PRECACHE_URLS = [
    '/',
    '/index.html',
    '/css/inspector.css',
    '/pyscript.toml',
];

// Patterns for resources to cache on fetch
const CACHE_PATTERNS = [
    // PyScript core
    /pyscript\.net/,
    // Pyodide runtime and packages (any CDN)
    /cdn\.jsdelivr\.net/,
    /pyodide/,
    /files\.pythonhosted\.org/,
    // Python wheel files
    /\.whl/,
    // WebAssembly files
    /\.wasm/,
    // Bulma CSS
    /bulma/,
    // Font Awesome
    /font-?awesome/,
];

// Install: precache app shell
self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => cache.addAll(PRECACHE_URLS))
            .then(() => self.skipWaiting())
    );
});

// Activate: clean up old caches
self.addEventListener('activate', (event) => {
    event.waitUntil(
        caches.keys()
            .then((cacheNames) => {
                return Promise.all(
                    cacheNames
                        .filter((name) => name !== CACHE_NAME)
                        .map((name) => caches.delete(name))
                );
            })
            .then(() => self.clients.claim())
    );
});

// Fetch: cache-first for matching patterns, network-first for others
self.addEventListener('fetch', (event) => {
    const url = event.request.url;
    const shouldCache = CACHE_PATTERNS.some((pattern) => pattern.test(url));

    if (shouldCache) {
        // Cache-first strategy for heavy assets
        event.respondWith(
            caches.match(event.request)
                .then((cachedResponse) => {
                    if (cachedResponse) {
                        return cachedResponse;
                    }
                    return fetch(event.request)
                        .then((response) => {
                            if (!response || response.status !== 200) {
                                return response;
                            }
                            const responseToCache = response.clone();
                            caches.open(CACHE_NAME)
                                .then((cache) => cache.put(event.request, responseToCache));
                            return response;
                        });
                })
        );
    } else {
        // Network-first for app files (allows updates)
        event.respondWith(
            fetch(event.request)
                .then((response) => {
                    // Cache successful responses for app files
                    if (response && response.status === 200 && event.request.url.startsWith(self.location.origin)) {
                        const responseToCache = response.clone();
                        caches.open(CACHE_NAME)
                            .then((cache) => cache.put(event.request, responseToCache));
                    }
                    // Add isolation headers to enable faster WASM
                    return addIsolationHeaders(response);
                })
                .catch(() => caches.match(event.request).then(r => r ? addIsolationHeaders(r) : r))
        );
    }
});
