// ChatUI-inspired Beautiful Chat Widget (MIT License)
(function () {
  // Load marked.js from CDN if not present
  if (!window.marked) {
    var script = document.createElement('script');
    script.src = 'https://cdn.jsdelivr.net/npm/marked/marked.min.js';
    script.onload = main;
    document.head.appendChild(script);
  } else {
    main();
  }

  // CC0 Bee SVG from SVG Repo: https://www.svgrepo.com/svg/228586/bees-bee
  function beeSVG(size=28) {
    return `<svg width="${size}" height="${size}" viewBox="0 0 512 512" xmlns="http://www.w3.org/2000/svg"><ellipse cx="256" cy="320" rx="140" ry="112" fill="#FAC13C" stroke="#1A1D24" stroke-width="16"/><ellipse cx="176" cy="176" rx="56" ry="98" fill="#FAFAFA" stroke="#10BFCC" stroke-width="10"/><ellipse cx="336" cy="176" rx="56" ry="98" fill="#FAFAFA" stroke="#10BFCC" stroke-width="10"/><ellipse cx="256" cy="320" rx="140" ry="112" fill="none" stroke="#1A1D24" stroke-width="16"/><ellipse cx="256" cy="320" rx="49" ry="112" fill="none" stroke="#1A1D24" stroke-width="10"/><ellipse cx="256" cy="320" rx="98" ry="112" fill="none" stroke="#1A1D24" stroke-width="10"/><circle cx="200" cy="304" r="18" fill="#1A1D24"/><circle cx="312" cy="304" r="18" fill="#1A1D24"/><path d="M200 220 Q184 180 216 164" stroke="#1A1D24" stroke-width="6" fill="none"/><path d="M312 220 Q328 180 296 164" stroke="#1A1D24" stroke-width="6" fill="none"/><path d="M232 368 Q256 400 280 368" stroke="#1A1D24" stroke-width="8" fill="none"/></svg>`;
  }

  // Gradient Decahedron SVG for anonymous user
  function userDecahedronSVG(size=28) {
    // A decahedron (10-sided polygon) with a gradient fill
    return `<svg width="${size}" height="${size}" viewBox="0 0 28 28" xmlns="http://www.w3.org/2000/svg">
      <defs>
        <linearGradient id="userDecaGrad" x1="0" y1="0" x2="1" y2="1">
          <stop offset="0%" stop-color="#10BFCC"/>
          <stop offset="60%" stop-color="#FAC13C"/>
          <stop offset="100%" stop-color="#8E949E"/>
        </linearGradient>
      </defs>
      <polygon points="14,3 22,7 25,15 21,23 14,26 7,23 3,15 6,7" fill="url(#userDecaGrad)" stroke="#FAFAFA" stroke-width="1.5"/>
    </svg>`;
  }

  function main() {
    const BACKEND_URL = 'http://localhost:8000/docs-agent'; // Change to your backend endpoint

    // --- Theme Detection & Sync with Site ---
    function getSiteTheme() {
      const html = document.documentElement;
      const body = document.body;
      // Bootstrap 5.3+ uses data-bs-theme, older may use class 'dark'/'light'
      const attrTheme = html.getAttribute('data-bs-theme') || body.getAttribute('data-bs-theme');
      if (attrTheme === 'dark' || attrTheme === 'light') return attrTheme;
      if (html.classList.contains('dark') || body.classList.contains('dark')) return 'dark';
      if (html.classList.contains('light') || body.classList.contains('light')) return 'light';
      return null; // fallback to system
    }
    function applyChatTheme(win) {
      win.classList.remove('chat-widget-dark', 'chat-widget-light');
      const theme = getSiteTheme();
      if (theme === 'dark') win.classList.add('chat-widget-dark');
      else if (theme === 'light') win.classList.add('chat-widget-light');
    }

    // --- DOM Creation ---
    // Floating button
    const chatBtn = document.createElement('button');
    chatBtn.id = 'chat-widget-btn';
    chatBtn.setAttribute('aria-label', 'Open chat');
    chatBtn.innerHTML = beeSVG(30);
    document.body.appendChild(chatBtn);

    // Chat window
    const chatWin = document.createElement('div');
    chatWin.id = 'chat-widget-window';
    chatWin.innerHTML = `
      <div id="chat-widget-header">
        <div class="chat-widget-header-title">${beeSVG(24)} W&B Docs Agent</div>
        <button id="chat-widget-close" aria-label="Close chat">×</button>
      </div>
      <div class="chat-widget-gradient-bar"></div>
      <div id="chat-widget-messages" aria-live="polite"></div>
      <form id="chat-widget-form" autocomplete="off">
        <textarea id="chat-widget-input" rows="1" placeholder="Type a message..." aria-label="Type a message" maxlength="1000"></textarea>
        <button type="submit" aria-label="Send message">
          <svg class="chat-widget-send-icon" width="20" height="20" viewBox="0 0 24 24">
            <path d="M2 21l21-9-21-9v7l15 2-15 2z" fill="#1A1D24"/>
          </svg>
        </button>
      </form>
    `;
    document.body.appendChild(chatWin);
    chatWin.style.display = 'none';

    // Apply theme on load
    applyChatTheme(chatWin);
    // Listen for theme changes (MutationObserver)
    const observer = new MutationObserver(() => applyChatTheme(chatWin));
    observer.observe(document.documentElement, { attributes: true, attributeFilter: ['class', 'data-bs-theme'] });
    observer.observe(document.body, { attributes: true, attributeFilter: ['class', 'data-bs-theme'] });

    // --- UI Interactions ---
    chatBtn.onclick = () => { chatWin.style.display = 'flex'; chatBtn.style.display = 'none'; setTimeout(() => focusInput(), 150); };
    chatWin.querySelector('#chat-widget-close').onclick = () => { chatWin.style.display = 'none'; chatBtn.style.display = 'flex'; };
    function focusInput() {
      const input = chatWin.querySelector('#chat-widget-input');
      if (input) input.focus();
    }

    // --- Gradient bar animation control ---
    const gradientBar = chatWin.querySelector('.chat-widget-gradient-bar');
    function setGradientBarFast(isFast) {
      if (!gradientBar) return;
      if (isFast) gradientBar.classList.add('chat-widget-gradient-fast');
      else gradientBar.classList.remove('chat-widget-gradient-fast');
    }

    // --- Messaging Logic ---
    const msgArea = chatWin.querySelector('#chat-widget-messages');
    const chatForm = chatWin.querySelector('#chat-widget-form');
    const chatInput = chatWin.querySelector('#chat-widget-input');
    const chatSendBtn = chatForm.querySelector('button[type="submit"]');
    const MAXLEN = 1000;
    let isOverflow = false;

    // Auto-grow textarea
    function autoGrowTextarea(e) {
      chatInput.style.height = 'auto';
      chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + 'px';
    }
    chatInput.addEventListener('input', autoGrowTextarea);
    chatInput.addEventListener('paste', function(e) {
      setTimeout(autoGrowTextarea, 0);
    });
    // Initial grow
    autoGrowTextarea();

    // Disable send button if input is empty or whitespace
    function updateSendBtn() {
      chatSendBtn.disabled = !chatInput.value.trim() || isOverflow;
    }
    chatInput.addEventListener('input', updateSendBtn);
    updateSendBtn();

    // Visual feedback for overflow
    chatInput.addEventListener('input', function() {
      isOverflow = chatInput.value.length > MAXLEN;
      if (isOverflow) {
        chatInput.classList.add('chat-widget-input-overflow');
        chatInput.value = chatInput.value.slice(0, MAXLEN);
        // Shake animation
        chatInput.classList.add('shake');
        setTimeout(() => chatInput.classList.remove('shake'), 400);
      } else {
        chatInput.classList.remove('chat-widget-input-overflow');
      }
      updateSendBtn();
    });

    // Keyboard navigation: Tab/Shift+Tab, Ctrl+A, Home/End work by default in textarea
    chatInput.addEventListener('keydown', function(e) {
      if (e.key === 'Enter') {
        if (e.shiftKey) {
          // Insert newline (default behavior)
          return;
        } else {
          // Submit form
          e.preventDefault();
          if (!chatSendBtn.disabled) chatForm.requestSubmit();
        }
      }
      // Tab/Shift+Tab navigation
      if (e.key === 'Tab') {
        if (e.shiftKey) {
          // Shift+Tab: move focus to send button
          chatSendBtn.focus();
          e.preventDefault();
        }
      }
    });
    chatSendBtn.addEventListener('keydown', function(e) {
      if (e.key === 'Tab' && !e.shiftKey) {
        // Tab from button goes back to textarea
        chatInput.focus();
        e.preventDefault();
      }
    });

    // --- Animated Waiting Placeholder ---
    function animatedWaiting() {
      return `
        <span class="chat-widget-loading-animated">
          <span class="cw-blob cw-blob1"></span>
          <span class="cw-blob cw-blob2"></span>
          <span class="cw-blob cw-blob3"></span>
        </span>
      `;
    }

    chatForm.onsubmit = async function (e) {
      e.preventDefault();
      const input = chatWin.querySelector('#chat-widget-input');
      const msg = input.value.trim();
      if (!msg) return;
      appendMsg('user', msg);
      input.value = '';
      autoGrowTextarea();
      updateSendBtn();
      appendMsg('bot', animatedWaiting(), true);
      setGradientBarFast(true); // Speed up while waiting
      scrollToBottom();
      try {
        const res = await fetch(BACKEND_URL, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ message: msg })
        });
        const data = await res.json();
        replaceLoading(data.answer || '[No reply]');
      } catch (err) {
        replaceLoading('<span class="chat-widget-error">[Error: Could not reach backend]</span>');
      }
      setGradientBarFast(false); // Slow down when done
      scrollToBottom();
    };

    // --- Markdown Rendering ---
    function renderMarkdown(md) {
      if (!window.marked) {
        console.warn('marked.js not loaded, rendering as plain text.');
        return escapeHtml(md);
      }
      try {
        return window.marked.parse(md, { breaks: true });
      } catch (e) {
        console.error('Markdown parse error:', e);
        return escapeHtml(md);
      }
    }
    function escapeHtml(str) {
      return str.replace(/[&<>"']/g, function (c) {
        return {'&':'&amp;','<':'&lt;','>':'&gt;','\"':'&quot;','\'':'&#39;'}[c]||c;
      });
    }
    function enhanceCodeBlocks(container) {
      const codeBlocks = container.querySelectorAll('pre code');
      codeBlocks.forEach(code => {
        // Wrap pre in a div for styling and button
        const pre = code.parentElement;
        if (!pre.classList.contains('chat-widget-pre-block')) {
          pre.classList.add('chat-widget-pre-block');
          // Add copy button
          const copyBtn = document.createElement('button');
          copyBtn.className = 'chat-widget-copy-btn';
          copyBtn.type = 'button';
          copyBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24"><rect x="9" y="9" width="13" height="13" rx="2" fill="#10BFCC"/><rect x="2" y="2" width="13" height="13" rx="2" fill="none" stroke="#8E949E" stroke-width="2"/></svg>';
          copyBtn.title = 'Copy code';
          copyBtn.onclick = function(e) {
            e.preventDefault();
            navigator.clipboard.writeText(code.innerText);
            copyBtn.classList.add('copied');
            copyBtn.innerText = 'Copied!';
            setTimeout(() => {
              copyBtn.classList.remove('copied');
              copyBtn.innerHTML = '<svg width="16" height="16" viewBox="0 0 24 24"><rect x="9" y="9" width="13" height="13" rx="2" fill="#10BFCC"/><rect x="2" y="2" width="13" height="13" rx="2" fill="none" stroke="#8E949E" stroke-width="2"/></svg>';
            }, 1200);
          };
          pre.style.position = 'relative';
          pre.appendChild(copyBtn);
        }
      });
    }

    function appendMsg(role, text, isHtml=false) {
      const msgDiv = document.createElement('div');
      msgDiv.className = 'chat-widget-msg ' + role;
      if (role === 'user') {
        msgDiv.innerHTML = `
          <div class="chat-widget-bubble chat-widget-user-bubble">
            <span class="chat-widget-avatar-inside">${userDecahedronSVG(28)}</span>
            <span class="chat-widget-user-text">${escapeHtml(text).replace(/\n/g, '<br>')}</span>
          </div>
        `;
      } else {
        if (isHtml) {
          msgDiv.innerHTML = `
            <div class="chat-widget-ai-content-with-avatar">
              <div class="chat-widget-ai-content">${text}</div>
              <div class="chat-widget-ai-avatar-row">${beeSVG(24)}</div>
            </div>
          `;
        } else {
          msgDiv.innerHTML = `
            <div class="chat-widget-ai-content-with-avatar">
              <div class="chat-widget-ai-content">${renderMarkdown(text)}</div>
              <div class="chat-widget-ai-avatar-row">${beeSVG(24)}</div>
            </div>
          `;
        }
      }
      msgArea.appendChild(msgDiv);
      enhanceCodeBlocks(msgDiv);
      scrollToBottom();
    }
    function replaceLoading(text) {
      const loading = msgArea.querySelector('.chat-widget-loading-animated');
      if (loading) {
        const parent = loading.parentElement;
        parent.innerHTML = `<div class="chat-widget-ai-content">${renderMarkdown(text)}</div>`;
        enhanceCodeBlocks(parent);
      }
      setGradientBarFast(false); // Slow down when done
    }
    function scrollToBottom() {
      msgArea.scrollTop = msgArea.scrollHeight;
    }

    // --- Accessibility: ESC closes chat, Enter submits ---
    chatWin.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') {
        chatWin.style.display = 'none';
        chatBtn.style.display = 'flex';
      }
    });

    // --- Responsive: close on outside click (mobile) ---
    window.addEventListener('click', function(e) {
      if (chatWin.style.display === 'flex' && !chatWin.contains(e.target) && e.target !== chatBtn) {
        chatWin.style.display = 'none';
        chatBtn.style.display = 'flex';
      }
    });
  }
})();
