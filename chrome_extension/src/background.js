// Minimal MV3 service worker for Wikontext
self.addEventListener('install', () => {
  self.skipWaiting();
});

chrome.runtime.onInstalled.addListener(() => {
  // placeholder for future initialization
});

// Keep a minimal message handler so the service worker can be extended later
chrome.runtime.onMessage.addListener((message, sender, sendResponse) => {
  // Example: support simple pings from content scripts
  if (message && message.type === 'PING') {
    sendResponse({ pong: true });
  }
  // Return true if we will respond asynchronously
  return false;
});
