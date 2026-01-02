const CACHE_NAME = 'autoocr-v1';
const ASSETS = [
    '/',
    '/static/css/modern.css',
    'https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css',
    'https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css'
];

self.addEventListener('install', (event) => {
    event.waitUntil(
        caches.open(CACHE_NAME)
            .then((cache) => cache.addAll(ASSETS))
    );
});

self.addEventListener('fetch', (event) => {
    // Simple cache-first strategy for static assets
    if (event.request.url.includes('/static/') || event.request.url.includes('cdn')) {
        event.respondWith(
            caches.match(event.request)
                .then((response) => response || fetch(event.request))
        );
    } else {
        // Network first for everything else (pages, dynamic content)
        event.respondWith(fetch(event.request));
    }
});
