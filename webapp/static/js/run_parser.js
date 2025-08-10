document.addEventListener('DOMContentLoaded', function () {
  // Collapsible sections
  function toggleSection(id) {
    const el = document.getElementById(id);
    if (!el) return;
    el.style.display = (el.style.display === 'none' || el.style.display === '') ? 'block' : 'none';
  }
  document.querySelectorAll('.collapsible-btn[data-target]').forEach(btn => {
    btn.addEventListener('click', () => {
      const id = btn.getAttribute('data-target');
      if (id) toggleSection(id);
    });
  });

  if (window.__parserSocketInitialized) return;
  window.__parserSocketInitialized = true;

  // Enhanced Footer Expand/Collapse Logic
  const footer = document.getElementById('sessionFooter');
  const preview = document.getElementById('footerPreview');
  const full = document.getElementById('footerFull');

  function createExplosion(x, y, color) {
    for (let i = 0; i < 24; i++) {
      const particle = document.createElement('div');
      particle.className = 'particle';
      const angle = Math.random() * 2 * Math.PI;
      const radius = 60 + Math.random() * 40;
      const dx = Math.cos(angle) * radius;
      const dy = Math.sin(angle) * radius;
      particle.style.left = x + 'px';
      particle.style.top = y + 'px';
      particle.style.setProperty('--dx', dx + 'px');
      particle.style.setProperty('--dy', dy + 'px');
      if (color) particle.style.background = color;
      document.body.appendChild(particle);
      setTimeout(() => particle.remove(), 800);
    }
  }

  preview?.addEventListener('click', () => {
    if (!footer) return;
    footer.classList.remove('minimized');
    footer.classList.add('expanded');
    const rect = preview.getBoundingClientRect();
    const x = rect.left + rect.width / 2;
    const y = rect.top + rect.height / 2 + window.scrollY;
    createExplosion(x, y);
  });

  full?.addEventListener('click', (e) => {
    if (!footer) return;
    if (e.target.closest('.session-btn')) return;
    footer.classList.remove('expanded');
    footer.classList.add('minimized');
    const rect = full.getBoundingClientRect();
    const x = rect.left + rect.width / 2;
    const y = rect.top + 10 + window.scrollY;
    createExplosion(x, y, '#eb4f43');
  });

  // Session Count Update
  const sessionCount = document.getElementById('sessionCount');
  function updateSessionCount() {
    let sessions = JSON.parse(localStorage.getItem('active_sessions') || '[]');
    if (sessionCount) sessionCount.textContent = `(${sessions.length})`;
  }

  // Robust Socket.IO Reconnect Logic
  let reconnectAttempts = 0;
  const maxReconnectDelay = 30000;
  let socket = null;

  function showDisconnectedMessage() {
    let outputDiv = document.getElementById('terminal');
    let msgId = 'socket-disconnect-msg';
    if (!outputDiv || document.getElementById(msgId)) return;
    let div = document.createElement('div');
    div.id = msgId;
    div.style = 'background:#eb4f43;color:#fff;padding:1em;border-radius:8px;margin:1em 0;text-align:center;font-weight:bold;';
    div.textContent = 'Lost connection to server. Attempting to reconnect...';
    outputDiv.appendChild(div);
  }
  function hideDisconnectedMessage() {
    let msg = document.getElementById('socket-disconnect-msg');
    if (msg) msg.remove();
  }
  function scheduleReconnect() {
    reconnectAttempts += 1;
    let delay = Math.min(1000 * Math.pow(2, reconnectAttempts), maxReconnectDelay);
    showDisconnectedMessage();
    setTimeout(() => {
      hideDisconnectedMessage();
      connectSocket();
    }, delay);
  }

  // URL index mapping for prompt
  let urlMap = {};
  function renderUrlList(urls) {
    urlMap = {};
    let html = `<ul style="list-style:none;padding:0;">`;
    if (!urls || urls.length === 0) {
      html += `<li style="color:#bfc9d1;">No URLs in urls.txt.</li>`;
    } else {
      urls.forEach((url, idx) => {
        let num = idx + 1;
        urlMap[num] = url;
        html += `<li style="padding:0.3em 0.5em;word-break:break-all;">[${num}] ${url}</li>`;
      });
    }
    html += `</ul>`;
    const holder = document.getElementById('urlList');
    if (holder) holder.innerHTML = html;
  }

  function connectSocket() {
    if (socket && typeof socket.disconnect === 'function') {
      socket.disconnect();
    }
    let prevSessionId = localStorage.getItem('session_id');
    try {
      if (prevSessionId && prevSessionId.startsWith('{')) {
        let parsed = JSON.parse(prevSessionId);
        if (parsed.session_id) prevSessionId = parsed.session_id;
      }
    } catch {}

    socket = io({
      query: { prev_session_id: prevSessionId || '' },
      reconnection: false
    });

    // Elements
    const outputDiv = document.getElementById('terminal');
    const promptInput = document.getElementById('promptInput');
    const cancelBtn = document.getElementById('cancelParserBtn');

    let cancelRequested = false;

    const activeSessionIdSpan = document.getElementById('activeSessionId');
    const sessionListDiv = document.getElementById('sessionList');
    const addSessionBtn = document.getElementById('addSessionBtn');

    function getSessions() {
      let sessions = JSON.parse(localStorage.getItem('active_sessions') || '[]');
      sessions = sessions.map(s => (typeof s === 'object' && s.session_id) ? s.session_id : s);
      sessions = Array.from(new Set(sessions));
      localStorage.setItem('active_sessions', JSON.stringify(sessions));
      return sessions;
    }

    function setActiveSession(sid) {
      if (typeof sid === 'object' && sid.session_id) sid = sid.session_id;
      localStorage.setItem('session_id', sid);
      if (activeSessionIdSpan) activeSessionIdSpan.textContent = sid;
      highlightActiveSession();
    }

    function highlightActiveSession() {
      let sid = localStorage.getItem('session_id');
      if (!sessionListDiv) return;
      Array.from(sessionListDiv.children).forEach(btn => {
        btn.classList.toggle('active', btn.dataset.sid === sid);
      });
    }

    function renderSessionList() {
      const sessions = getSessions();
      if (!sessionListDiv) return;
      sessionListDiv.innerHTML = '';
      sessions.forEach(sid => {
        if (typeof sid === 'object' && sid.session_id) sid = sid.session_id;
        let btn = document.createElement('button');
        btn.textContent = sid;
        btn.className = 'session-btn';
        btn.dataset.sid = sid;
        btn.style.width = '100%';
        btn.style.marginBottom = '0.5em';
        btn.onclick = function () {
          setActiveSession(sid);
          highlightActiveSession();
          if (outputDiv) outputDiv.innerHTML = '';
          socket.emit('join', { session_id: sid });
          loadSessionLogs(sid);
        };
        let removeBtn = document.createElement('span');
        removeBtn.textContent = '✖';
        removeBtn.style.cssText = 'float:right;color:#eb4f43;cursor:pointer;font-weight:bold;margin-left:1em;';
        removeBtn.onclick = function (e) {
          e.stopPropagation();
          socket.emit('delete_session', { session_id: sid });
          let sessions = getSessions().filter(s => s !== sid);
          localStorage.setItem('active_sessions', JSON.stringify(sessions));
          if (localStorage.getItem('session_id') === sid) {
            setActiveSession(sessions[0] || '');
          }
          renderSessionList();
        };
        btn.appendChild(removeBtn);
        sessionListDiv.appendChild(btn);
      });
      highlightActiveSession();
      updateSessionCount();
    }

    addSessionBtn && (addSessionBtn.onclick = function () {
      let newSid = 'sess_' + Math.random().toString(36).substr(2, 9);
      let sessions = getSessions();
      sessions.push(newSid);
      localStorage.setItem('active_sessions', JSON.stringify(sessions));
      setActiveSession(newSid);
      renderSessionList();
      socket.emit('join', { session_id: newSid });
      if (outputDiv) outputDiv.innerHTML = '';
    });

    const runBtn = document.getElementById('runParserBtn');
    runBtn?.addEventListener('click', function (e) {
      e.preventDefault();
      let sid = localStorage.getItem('session_id');
      if (!sid) {
        sid = 'sess_' + Math.random().toString(36).substr(2, 9);
        let sessions = getSessions();
        sessions.push(sid);
        localStorage.setItem('active_sessions', JSON.stringify(sessions));
        setActiveSession(sid);
        renderSessionList();
        socket.emit('join', { session_id: sid });
      }
      if (outputDiv) outputDiv.innerHTML = '';
      const rect = this.getBoundingClientRect();
      const x = rect.left + rect.width / 2;
      const y = rect.top + rect.height / 2 + window.scrollY;
      // Local explosion for run button
      function createExplosionLocal(x, y, color) {
        for (let i = 0; i < 32; i++) {
          const p = document.createElement('div');
          p.className = 'particle';
          const ang = Math.random() * 2 * Math.PI;
          const rad = 80 + Math.random() * 60;
          const dx = Math.cos(ang) * rad;
          const dy = Math.sin(ang) * rad;
          p.style.left = x + 'px';
          p.style.top = y + 'px';
          p.style.setProperty('--dx', dx + 'px');
          p.style.setProperty('--dy', dy + 'px');
          document.body.appendChild(p);
          setTimeout(() => p.remove(), 800);
        }
      }
      createExplosionLocal(x, y);
      socket.emit('run_parser');
    });

    function loadSessionLogs(sid) {
      socket.emit('get_session_history', { session_id: sid });
    }

    // Log filtering and rendering
    const seenLogTypes = new Set(['all', 'status', 'router', 'ai_analysis', 'stream', 'manual_override', 'input', 'output', 'validation', 'network', 'cancel', 'heartbeat', 'summary', 'cache', 'other']);
    const logTypeSelect = document.getElementById('logTypeFilterSelect');
    function addLogTypeOption(type) {
      type = (type || 'other').toLowerCase();
      if (!seenLogTypes.has(type)) {
        seenLogTypes.add(type);
        if (!logTypeSelect) return;
        const opt = document.createElement('option');
        opt.value = type;
        opt.textContent = type.charAt(0).toUpperCase() + type.slice(1).replace(/_/g, ' ');
        logTypeSelect.appendChild(opt);
      }
    }

    cancelBtn?.addEventListener('click', function (e) {
      e.preventDefault();
      if (cancelRequested) return;
      cancelRequested = true;
      socket.emit('cancel_parser');
      if (outputDiv) {
        outputDiv.innerHTML += '<br><span style="color:#eb4f43;font-weight:bold;">[CANCEL REQUESTED]</span><br>';
        outputDiv.scrollTop = outputDiv.scrollHeight;
      }
      window.scrollTo(0, document.body.scrollHeight);
      cancelBtn.disabled = true;
      cancelBtn.textContent = 'Canceling...';
      setTimeout(() => {
        cancelBtn.disabled = false;
        cancelBtn.textContent = 'Cancel';
        cancelRequested = false;
        let sid = localStorage.getItem('session_id');
        if (sid) {
          let sessions = JSON.parse(localStorage.getItem('active_sessions') || '[]').filter(s => s !== sid);
          localStorage.setItem('active_sessions', JSON.stringify(sessions));
          if (sessions.length > 0) {
            localStorage.setItem('session_id', sessions[0]);
          } else {
            localStorage.removeItem('session_id');
          }
          renderSessionList();
        }
      }, 5000);
    });

    function sendPrompt(event) {
      event.preventDefault();
      const promptInput = document.getElementById('promptInput');
      if (!promptInput) return;
      let value = promptInput.value.trim();
      let sessionId = localStorage.getItem('session_id');
      let match = value.match(/^\[?(\d+)\]?$/);
      if (match && urlMap[match[1]]) {
        value = urlMap[match[1]];
      }
      if (value && sessionId) {
        socket.emit('parser_prompt', { session_id: sessionId, value: value });
        promptInput.value = '';
      }
    }
    const promptForm = document.getElementById('promptForm');
    promptForm?.addEventListener('submit', sendPrompt);

    promptInput?.addEventListener('keydown', function (event) {
      if (event.key === 'Escape') this.value = '';
    });

    const logFilterSelect = document.getElementById('logFilterSelect');
    logFilterSelect?.addEventListener('change', filterLogs);
    logTypeSelect?.addEventListener('change', filterLogs);

    function filterLogs() {
      const levelFilter = logFilterSelect ? logFilterSelect.value : 'all';
      const typeFilter = logTypeSelect ? logTypeSelect.value : 'all';
      const logLines = document.querySelectorAll('.log-line, .log-panel');
      logLines.forEach(line => {
        const level = (line.getAttribute('data-level') || 'other').toLowerCase();
        const type = (line.getAttribute('data-type') || 'other').toLowerCase();
        const showLevel = (levelFilter === 'all' || level === levelFilter);
        const showType = (typeFilter === 'all' || type === typeFilter);
        line.style.display = (showLevel && showType) ? '' : 'none';
      });
    }

    const outputModeSelect = document.getElementById('outputModeSelect');
    outputModeSelect?.addEventListener('change', function () {
      const mode = this.value;
      socket.emit('set_output_mode', { mode });
    });

    function renderParserOutput(data) {
      const outputDiv = document.getElementById('terminal');
      if (!outputDiv) return;
      if (outputDiv.innerHTML.includes('Parser output will appear')) {
        outputDiv.innerHTML = '';
      }
      let obj = data;
      let isJson = false;
      if (typeof data === 'string' && data.trim().startsWith('{')) {
        try {
          obj = JSON.parse(data);
          if (typeof obj === 'string' && obj.trim().startsWith('{')) {
            obj = JSON.parse(obj);
          }
          isJson = true;
        } catch {
          obj = data;
        }
      }

      if (isJson && obj && obj.type === 'prompt' && typeof promptInput !== 'undefined' && promptInput) {
        promptInput.placeholder = obj.message || 'Type a command...';
        promptInput.focus();
      }

      let logLevel = (isJson && obj && obj.level) ? String(obj.level).toUpperCase() : 'OTHER';
      let logType = (isJson && obj && obj.type) ? String(obj.type).toLowerCase() : 'other';
      addLogTypeOption(logType);

      let sessionId = (isJson && obj.session_id) ? obj.session_id : '';
      let timeStr = '';
      if (obj && obj.timestamp) {
        const d = new Date(obj.timestamp * 1000);
        timeStr = `<span class="log-time">[${d.toLocaleTimeString()}]</span>`;
      }
      const levelIconMap = { INFO: '🛈', DEBUG: '⚙️', WARNING: '⚠️', ERROR: '⛔', CRITICAL: '🚨', CANCELLED: '🛑', CANCEL: '🛑' };
      const levelColorMap = { INFO: '#00ffe7', DEBUG: '#8ecae6', WARNING: '#ffd166', ERROR: '#eb4f43', CRITICAL: '#ff006e', CANCELLED: '#eb4f43', CANCEL: '#eb4f43' };
      const levelIcon = levelIconMap[logLevel] || '🛈';
      const levelColor = levelColorMap[logLevel] || '#fff';

      let statusColor = (isJson && obj.status_color) ? obj.status_color : null;
      let messageColor = (isJson && obj.message_color) ? obj.message_color : null;

      function formatValue(val, key) {
        if (Array.isArray(val)) {
          return `<ul style="margin:0 0 0 1.5em;padding:0;">` +
            val.map(item => `<li style="margin-bottom:0.2em;">${formatValue(item)}</li>`).join('') + `</ul>`;
        } else if (typeof val === 'object' && val !== null) {
          const keys = Object.keys(val).filter(k => !['level', 'type', 'session_id', 'timestamp'].includes(k));
          if (keys.length === 0) return '';
          return `<div style="margin:0.5em 0 0 1.5em;padding:0.5em 1em;background:rgba(40,60,80,0.7);border-radius:8px;font-size:0.97em;color:#fff;">
              ${keys.map(k => `<div><span style="color:#ffd166;">${k}:</span> ${formatValue(val[k], k)}</div>`).join('')}
          </div>`;
        } else if (typeof val === 'string') {
          if (key === 'status' && statusColor) return `<span style="color:${statusColor};font-weight:bold;">${val}</span>`;
          if (key === 'message' && messageColor) return `<span style="color:${messageColor};text-shadow:0 0 6px ${messageColor};">${val}</span>`;
          return `<span style="color:#fff;text-shadow:0 0 6px #00ffe7;">${val}</span>`;
        } else {
          return `<span style="color:#fff;">${JSON.stringify(val)}</span>`;
        }
      }

      let mainMsg = '';
      let extraPayload = '';
      if (isJson && obj) {
        if (obj.message !== undefined) {
          mainMsg = formatValue(obj.message, 'message');
        } else {
          mainMsg = `<span style="color:#fff;">${JSON.stringify(obj)}</span>`;
        }
        const extraFields = Object.keys(obj).filter(k => !['level', 'type', 'session_id', 'timestamp', 'message'].includes(k));
        if (extraFields.length > 0) {
          extraPayload = `<div style="margin:0.5em 0 0 1.5em;padding:0.5em 1em;background:rgba(40,60,80,0.5);border-radius:8px;font-size:0.97em;color:#bfc9d1;">
              ${extraFields.map(k => `<div><span style="color:#ffd166;">${k}:</span> ${formatValue(obj[k], k)}</div>`).join('')}
          </div>`;
        }
      } else {
        mainMsg = `<span style="color:#fff;">${typeof obj === 'string' ? obj.replace(/\n/g, '<br>') : JSON.stringify(obj)}</span>`;
      }

      let infoPopup = `
        <div class="log-popup" style="display:none;position:absolute;top:100%;left:0;z-index:9999;background:rgba(30,40,60,0.98);color:#fff;border-radius:12px;box-shadow:0 4px 32px #00ffe799;padding:1em 1.5em;min-width:260px;max-width:400px;pointer-events:none;transition:opacity 0.2s;">
          <div style="margin-bottom:0.5em;">
            <span style="color:#ffd166;font-weight:700;">Level:</span> <span style="color:${levelColor};font-weight:900;">${levelIcon} ${logLevel}</span>
          </div>
          <div style="margin-bottom:0.5em;">
            <span style="color:#ffd166;font-weight:700;">Type:</span> <span style="color:#00ffe7;font-weight:700;">${logType}</span>
          </div>
          <div style="margin-bottom:0.5em;">
            <span style="color:#ffd166;font-weight:700;">Session ID:</span> <span style="color:#bfc9d1;font-family:monospace;">${sessionId}</span>
          </div>
        </div>
      `;

      let msg = `
        <div class="log-line futuristic-log" data-level="${logLevel}" data-type="${logType}" style="margin-bottom:0.3em;padding:1.1em 1.5em 1.1em 1.1em;border-radius:16px;background:linear-gradient(90deg,rgba(30,40,60,0.85),rgba(0,255,231,0.07));box-shadow:0 0 16px 2px ${levelColor}33,0 2px 12px #bfc9d1;overflow-x:auto;position:relative;cursor:pointer;transition:box-shadow 0.2s;">
          <span class="log-main" style="font-size:1.13em;display:block;">
            ${timeStr}
            <span class="log-level" style="color:${levelColor};font-weight:900;text-shadow:0 0 8px ${levelColor};margin-right:0.7em;">${levelIcon} ${logLevel}</span>
            ${mainMsg}
            ${extraPayload}
          </span>
          ${infoPopup}
        </div>
      `;

      const temp = document.createElement('div');
      temp.innerHTML = msg;
      const logElem = temp.firstElementChild;
      outputDiv.appendChild(logElem);

      const popup = logElem.querySelector('.log-popup');
      let popupVisible = false;
      logElem.addEventListener('click', function (e) {
        if (popup.contains(e.target)) return;
        popupVisible = !popupVisible;
        if (popupVisible) {
          popup.style.display = 'block';
          popup.style.opacity = '1';
          popup.style.pointerEvents = 'auto';
          setTimeout(() => { document.addEventListener('mousedown', outsideClickListener); }, 0);
        } else {
          popup.style.opacity = '0';
          popup.style.pointerEvents = 'none';
          setTimeout(() => { popup.style.display = 'none'; }, 200);
          document.removeEventListener('mousedown', outsideClickListener);
        }
      });
      function outsideClickListener(event) {
        if (!logElem.contains(event.target)) {
          popupVisible = false;
          popup.style.opacity = '0';
          popup.style.pointerEvents = 'none';
          setTimeout(() => { popup.style.display = 'none'; }, 200);
          document.removeEventListener('mousedown', outsideClickListener);
        }
      }

      logElem.classList.add('log-glow');
      setTimeout(() => logElem.classList.remove('log-glow'), 1200);
      outputDiv.scrollTop = outputDiv.scrollHeight;
      window.scrollTo(0, document.body.scrollHeight);
      if (typeof filterLogs === 'function') filterLogs();
    }

    socket.on('session_history', function (data) {
      if (!data || !data.logs || !Array.isArray(data.logs)) return;
      if (outputDiv) outputDiv.innerHTML = '';
      data.logs.forEach(renderParserOutput);
      if (outputDiv) outputDiv.scrollTop = outputDiv.scrollHeight;
    });

    socket.on('parser_output', renderParserOutput);

    socket.on('connect', function () {
      reconnectAttempts = 0;
      hideDisconnectedMessage();
      let sid = localStorage.getItem('session_id');
      renderSessionList();
      if (sid) {
        setActiveSession(sid);
        socket.emit('join', { session_id: sid });
        loadSessionLogs(sid);
      }
    });

    socket.on('disconnect', function () {
      showDisconnectedMessage();
      scheduleReconnect();
    });

    socket.on('connect_error', function () {
      showDisconnectedMessage();
      scheduleReconnect();
    });

    socket.on('session_id', function (data) {
      let sid = typeof data === 'string' ? data : data.session_id;
      let sessions = getSessions();
      if (!sessions.includes(sid)) {
        sessions.push(sid);
        localStorage.setItem('active_sessions', JSON.stringify(sessions));
      }
      setActiveSession(sid);
      renderSessionList();
    });

    renderSessionList();
    setActiveSession(localStorage.getItem('session_id') || '');
    socket.emit('get_sessions');

    socket.on('session_list', function (data) {
      if (Array.isArray(data.sessions)) {
        let sessionIds = data.sessions.map(s => (typeof s === 'object' && s.session_id) ? s.session_id : s);
        localStorage.setItem('active_sessions', JSON.stringify(sessionIds));
        renderSessionList();
        if (!sessionIds.includes(localStorage.getItem('session_id'))) {
          setActiveSession(sessionIds[0] || '');
        }
      }
    });

    socket.on('session_deleted', function (data) {
      let sid = data.session_id;
      let sessions = getSessions().filter(s => s !== sid);
      localStorage.setItem('active_sessions', JSON.stringify(sessions));
      if (localStorage.getItem('session_id') === sid) {
        setActiveSession(sessions[0] || '');
      }
      renderSessionList();
    });

    socket.on('session_heartbeat', function (data) {
      if (!data || !data.session_id) return;
      let sid = data.session_id;
      let btn = document.querySelector('.session-btn[data-sid="' + sid + '"]');
      if (!btn) return;
      let hb = btn.querySelector('.heartbeat-indicator');
      if (!hb) {
        hb = document.createElement('span');
        hb.className = 'heartbeat-indicator';
        hb.style.cssText = 'margin-left:0.7em;vertical-align:middle;';
        hb.innerHTML = `
          <svg width="36" height="18" viewBox="0 0 36 18" style="vertical-align:middle;">
            <polyline class="ekg-wave" points="0,9 6,9 9,3 12,15 15,9 18,9 21,6 24,12 27,9 36,9" 
              fill="none" stroke="#00ffe7" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>
          </svg>
        `;
        btn.appendChild(hb);
      }
      let svg = hb.querySelector('svg');
      let wave = hb.querySelector('.ekg-wave');
      wave.style.stroke = '#00ffe7';
      wave.style.filter = 'drop-shadow(0 0 6px #00ffe7)';
      svg.style.opacity = '1';
      svg.style.transition = 'opacity 0.3s';
      wave.style.animation = 'ekg-pulse 0.7s cubic-bezier(.61,-0.01,.7,1.01)';
      clearTimeout(hb._ekgTimeout);
      hb._ekgTimeout = setTimeout(() => { wave.style.animation = ''; }, 700);

      clearTimeout(hb._flatlineTimeout);
      hb._flatlineTimeout = setTimeout(() => {
        wave.setAttribute('points', '0,9 36,9');
        wave.style.stroke = '#eb4f43';
        wave.style.filter = 'drop-shadow(0 0 6px #eb4f43)';
        svg.style.opacity = '0.7';
      }, 3000);
    });
  }

  // URL Hint Overrides and URL List logic
  function fetchOverrides() {
    fetch('/api/url_hint_overrides')
      .then(r => r.json())
      .then(data => renderOverridesTable(data.overrides || {}))
      .catch(() => renderOverridesTable({}));
  }
  function renderOverridesTable(overrides) {
    let html = `<table style="width:100%;border-collapse:collapse;">
      <tr style="background:#23272b;color:#00ffe7;">
        <th style="padding:0.5em 1em;">URL Fragment</th>
        <th style="padding:0.5em 1em;">Module Path</th>
        <th style="padding:0.5em 1em;">Actions</th>
      </tr>`;
    const keys = Object.keys(overrides || {});
    if (keys.length === 0) {
      html += `<tr><td colspan="3" style="text-align:center;color:#bfc9d1;">No overrides set.</td></tr>`;
    } else {
      for (let frag of keys) {
        html += `<tr>
          <td style="padding:0.5em 1em;">${frag}</td>
          <td style="padding:0.5em 1em;">${overrides[frag]}</td>
          <td style="padding:0.5em 1em;">
            <button class="btn btn-danger" data-delete-override="${frag.replace(/"/g, '&quot;')}">Delete</button>
          </td>
        </tr>`;
      }
    }
    html += `</table>`;
    const holder = document.getElementById('hintOverridesTable');
    if (holder) {
      holder.innerHTML = html;
      holder.querySelectorAll('[data-delete-override]').forEach(btn => {
        btn.addEventListener('click', () => {
          const frag = btn.getAttribute('data-delete-override');
          if (!confirm('Delete override for ' + frag + '?')) return;
          fetch('/api/url_hint_overrides', {
            method: 'DELETE',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ fragment: frag })
          }).then(r => r.json()).then(fetchOverrides);
        });
      });
    }
  }

  document.getElementById('addUrlHintForm')?.addEventListener('submit', function (e) {
    e.preventDefault();
    let frag = document.getElementById('urlFragment').value.trim();
    let path = document.getElementById('modulePath').value.trim();
    if (!frag || !path) return;
    fetch('/api/url_hint_overrides', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ fragment: frag, module_path: path })
    }).then(r => r.json()).then(res => {
      if (res.success) {
        fetchOverrides();
        this.reset();
      } else {
        alert(res.error || 'Failed to add override.');
      }
    });
  });

  function fetchUrls() {
    fetch('/api/urls')
      .then(r => r.json())
      .then(data => renderUrlList(data.urls || []))
      .catch(() => renderUrlList([]));
  }
  document.getElementById('addUrlForm')?.addEventListener('submit', function (e) {
    e.preventDefault();
    let url = document.getElementById('newUrl').value.trim();
    if (!url) return;
    fetch('/api/urls', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ url })
    }).then(r => r.json()).then(res => {
      if (res.success) {
        fetchUrls();
        this.reset();
      } else {
        alert(res.error || 'Failed to add URL.');
      }
    });
  });

  // Initial data for sidebar lists and socket setup
  fetchOverrides();
  fetchUrls();
  connectSocket();
});