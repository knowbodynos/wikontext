// Modernized, standalone vanilla JS content script.
(function () {
	const APP_URL = 'https://wikontext.us';
	const state = new WeakMap();

	function clearPopup() {
		const popup = document.getElementById('popup-notification');
		if (popup) popup.remove();
	}

	function waitUntil(isReady, { interval = 50, timeout = 3000 } = {}) {
		const start = Date.now();
		return new Promise((resolve, reject) => {
			(function check() {
				if (isReady()) return resolve();
				if (Date.now() - start >= timeout) return reject(new Error('timeout'));
				setTimeout(check, interval);
			})();
		});
	}

	async function performWork(linkEl) {
		clearPopup();
		if (linkEl.title) {
			linkEl.dataset._savedTitle = linkEl.title;
			linkEl.removeAttribute('title');
		}

		const href = linkEl.getAttribute('href') || '';
		let fullURL;
		try {
			fullURL = new URL(href, window.location.href).href;
		} catch (e) {
			return;
		}

		if (!fullURL.includes('/wiki/')) return;

		const originTitle = document.title.replace(' - Wikipedia', '');
		let originContent = '';
		let originContextA = '';
		let originContextP = '';
		document.querySelectorAll('p').forEach((p) => {
			if (p.contains(linkEl) || p === linkEl.parentElement) {
				originContextA = linkEl.outerHTML.replace(/\n/g, ' ');
				originContextP = p.innerHTML.replace(/\n/g, ' ');
			}
			originContent += p.innerHTML.replace(/\n/g, ' ');
		});

		const parser = new DOMParser();
		try {
			const resp = await fetch(fullURL);
			const html = await resp.text();
			const doc = parser.parseFromString(html, 'text/html');
			const targetTitle = (doc.querySelector('title') || { textContent: '' }).textContent.replace(' - Wikipedia', '');
			let targetContent = '';
			doc.querySelectorAll('p').forEach((p) => {
				targetContent += p.innerHTML.replace(/\n/g, ' ');
			});

			const sendDict = {
				origin_title: originTitle,
				target_title: targetTitle,
				origin_content: originContent,
				target_content: targetContent,
				origin_context_a: originContextA,
				origin_context_p: originContextP,
			};

			const postController = new AbortController();
			state.set(linkEl, { controller: postController });

			const postResp = await fetch(APP_URL + '/api/ext', {
				method: 'POST',
				headers: { 'Content-Type': 'application/json' },
				body: JSON.stringify(sendDict),
				signal: postController.signal,
			});
			const hoverText = await postResp.text();

			try {
				await waitUntil(() => document.querySelectorAll('a.mwe-popups-extract p').length > 0, { interval: 50, timeout: 3000 });
				const paras = document.querySelectorAll('a.mwe-popups-extract p');
				paras.forEach((p) => {
					p.style.overflowY = 'auto';
					p.innerHTML = hoverText;
				});
			} catch (e) {
				console.log('Page preview taking too long to load.');
			}
		} catch (err) {
			if (err.name === 'AbortError') return;
			console.error('Error fetching content', err);
		}
	}

	function attachListeners() {
		const links = document.querySelectorAll('#bodyContent a');
		links.forEach((link) => {
			const onEnter = () => {
				if (link.querySelector('img')) return;
				const timer = setTimeout(() => performWork(link), 200);
				state.set(link, { timeout: timer });
			};
			const onLeave = () => {
				const st = state.get(link) || {};
				if (st.timeout) clearTimeout(st.timeout);
				if (st.controller) st.controller.abort();
				clearPopup();
				if (link.dataset._savedTitle) {
					link.title = link.dataset._savedTitle;
					delete link.dataset._savedTitle;
				}
			};
			link.addEventListener('mouseenter', onEnter);
			link.addEventListener('mouseleave', onLeave);
		});

		document.body.addEventListener('click', () => {
			clearPopup();
		});
	}

	if (document.readyState === 'complete' || document.readyState === 'interactive') {
		attachListeners();
	} else {
		window.addEventListener('DOMContentLoaded', attachListeners);
	}
})();