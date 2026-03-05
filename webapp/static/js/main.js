/* Smart Elections Parser - Home Page Script
   - Modular init functions
   - Removed inline style mutations (CSS classes instead)
   - Asset hydration isolated
   - Popup API without inline onclick
   - Reduced motion respect
   - Defensive guards & idempotent patterns
*/

(() => {
  'use strict';

  /* ---------- Asset Hydration (CSP-safe) ---------- */
  const W = /** @type {any} */ (window);
  (function hydrateStaticAssets() {
    const el = document.getElementById('assetPaths');
    if (!el) return;
    W.STATIC_ASSETS = {
      sunSvg:  el.dataset.sunSvg,
      sunPng:  el.dataset.sunPng,
      moonSvg: el.dataset.moonSvg,
      moonPng: el.dataset.moonPng,
      earth:   el.dataset.earth
    };
  })();

  /* ---------- Bootstrap Enhancements ---------- */
  function initBootstrap() {
    if (!W.bootstrap) return;
    document.querySelectorAll('[data-bs-toggle="tooltip"]')
      .forEach(el => W.bootstrap.Tooltip.getOrCreateInstance(el));
    document.querySelectorAll('[data-bs-toggle="popover"]')
      .forEach(el => W.bootstrap.Popover.getOrCreateInstance(el));
  }

  /* ---------- Mission Rim Tracer Animation ---------- */
  function initRimTracer() {
    /** @type {HTMLCanvasElement|null} */
    const canvas = /** @type {HTMLCanvasElement|null} */ (document.getElementById('rimTracer'));
    /** @type {HTMLElement|null} */
    const panel = /** @type {HTMLElement|null} */ (document.getElementById('missionPanel'));
    if (!canvas || !panel) return;

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    const SPEED_DIVISOR = 3;
    const R = 3;
    const TRAIL_STEPS = 8;
    const HEAD_R = 5;
    const TAIL_R = 1;
    const sparks = [];
    const prefersReduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    function resize() {
      canvas.width = panel.offsetWidth;
      canvas.height = panel.offsetHeight;
    }

    function perimeterPos(d, w, h, r) {
      if (d < w - 2 * r) return { x: r + d, y: r, a: 0 };
      d -= (w - 2 * r);
      if (d < r * Math.PI / 2)
        return {
          x: w - r + r * Math.cos(Math.PI / 2 - d / r),
          y: r + r * Math.sin(Math.PI / 2 - d / r),
          a: Math.PI / 2 - d / r
        };
      d -= r * Math.PI / 2;
      if (d < h - 2 * r) return { x: w - r, y: r + d, a: Math.PI / 2 };
      d -= (h - 2 * r);
      if (d < r * Math.PI / 2)
        return {
          x: w - r + r * Math.sin(d / r),
          y: h - r + r * Math.cos(d / r),
          a: Math.PI + d / r
        };
      d -= r * Math.PI / 2;
      if (d < w - 2 * r) return { x: w - r - d, y: h - r, a: Math.PI };
      d -= (w - 2 * r);
      if (d < r * Math.PI / 2)
        return {
          x: r + r * Math.cos(3 * Math.PI / 2 + d / r),
          y: h - r + r * Math.sin(3 * Math.PI / 2 + d / r),
          a: 3 * Math.PI / 2 + d / r
        };
      d -= r * Math.PI / 2;
      return { x: r, y: h - r - d, a: 3 * Math.PI / 2 };
    }

    function frame() {
      const w = canvas.width;
      const h = canvas.height;
      ctx.clearRect(0, 0, w, h);

      const t = Date.now() / (2200 * SPEED_DIVISOR);
      const perim = 2 * (w + h - 4 * R) + 2 * Math.PI * R;
      const speed = (t % 1) * perim;

      for (let i = 1; i <= TRAIL_STEPS; i++) {
        const d = (speed - i * 7 + perim) % perim;
        const { x, y } = perimeterPos(d, w, h, R);
        const radius = TAIL_R + (HEAD_R - TAIL_R) * ((TRAIL_STEPS - i) / (TRAIL_STEPS - 1));
        ctx.save();
        ctx.globalAlpha = 0.03 + 0.04 * (TRAIL_STEPS - i + 1);
        ctx.shadowColor = "#00ffe7";
        ctx.shadowBlur = 16 + ((TRAIL_STEPS - i + 1) * 2);
        ctx.beginPath();
        ctx.arc(x, y, radius, 0, Math.PI * 2);
        ctx.fillStyle = "#00ffe7";
        ctx.fill();
        ctx.restore();
      }

      const head = perimeterPos(speed, w, h, R);
      ctx.save();
      ctx.globalAlpha = 0.18;
      ctx.shadowColor = "#00ffe7";
      ctx.shadowBlur = 18;
      ctx.beginPath();
      ctx.arc(head.x, head.y, HEAD_R, 0, Math.PI * 2);
      ctx.fillStyle = "#00ffe7";
      ctx.fill();
      ctx.restore();

      // Border path
      ctx.save();
      ctx.strokeStyle = "rgba(0,255,231,0.08)";
      ctx.shadowColor = "#00ffe7";
      ctx.shadowBlur = 6;
      ctx.lineWidth = 3;
      ctx.beginPath();
      ctx.moveTo(R, 0);
      ctx.lineTo(w - R, 0);
      ctx.arcTo(w, 0, w, R, R);
      ctx.lineTo(w, h - R);
      ctx.arcTo(w, h, w - R, h, R);
      ctx.lineTo(R, h);
      ctx.arcTo(0, h, 0, h - R, R);
      ctx.lineTo(0, R);
      ctx.arcTo(0, 0, R, 0, R);
      ctx.closePath();
      ctx.stroke();
      ctx.restore();

      // Sparks
      if (Math.random() < 0.08) {
        const a = head.a + (Math.random() - 0.5) * 0.7;
        const len = 12 + Math.random() * 12;
        sparks.push({
          x: head.x,
          y: head.y,
          dx: Math.cos(a) * len,
            dy: Math.sin(a) * len,
          alpha: 0.5 + Math.random() * 0.2,
          life: 0
        });
      }
      for (let i = sparks.length - 1; i >= 0; i--) {
        const s = sparks[i];
        s.life++;
        const px = s.x + s.dx * (s.life / 10);
        const py = s.y + s.dy * (s.life / 10);
        ctx.save();
        ctx.globalAlpha = s.alpha * (1 - s.life / 10);
        ctx.shadowColor = "#00ffe7";
        ctx.shadowBlur = 8;
        ctx.beginPath();
        ctx.arc(px, py, 1.5, 0, Math.PI * 2);
        ctx.fillStyle = "#00ffe7";
        ctx.fill();
        ctx.restore();
        if (s.life > 10) sparks.splice(i, 1);
      }

      if (!prefersReduce) requestAnimationFrame(frame);
    }

    resize();
    window.addEventListener('resize', resize);
    panel.addEventListener('mouseenter', () => panel.classList.add('glossy'));
    panel.addEventListener('mouseleave', () => panel.classList.remove('glossy'));

    if (!prefersReduce) frame();
  }

  /* ---------- Feature Hover (class-based) ---------- */
  function initFeatureHover() {
    const features = document.querySelectorAll('.feature');
    if (!features.length) return;

    function clear(el) {
      el.classList.remove('is-hovered-run','is-hovered-history','is-hovered-data','is-hovered-none');
    }
    function apply(el) {
      clear(el);
      const href = el.getAttribute('href') || '';
      if (href.includes('ballot_lens')) el.classList.add('is-hovered-run');
      else if (href.includes('history')) el.classList.add('is-hovered-history');
      else if (href.includes('data_framework')) el.classList.add('is-hovered-data');
      else el.classList.add('is-hovered-none');
    }

    features.forEach(el => {
      el.addEventListener('mouseenter', () => apply(el));
      el.addEventListener('mousemove', () => apply(el));
      el.addEventListener('mouseleave', () => clear(el));
      el.addEventListener('focus', () => apply(el));
      el.addEventListener('blur', () => clear(el));
    });
  }

  /* ---------- Popup API (no inline handlers) ---------- */
  function buildPopupMessage(primaryText, highlightText) {
    const message = document.createElement('div');
    if (primaryText) message.append(String(primaryText));
    if (highlightText) {
      message.appendChild(document.createElement('br'));
      const highlight = document.createElement('span');
      highlight.className = 'popup-highlight';
      highlight.textContent = String(highlightText);
      message.appendChild(highlight);
    }
    return message;
  }

  function initPopup() {
    const popup = document.getElementById('popup');
    if (!popup) return;

    const FOCUSABLE = 'a[href],button:not([disabled]),[tabindex]:not([tabindex="-1"])';
    let lastActive = null;

    function trapFocus(e) {
      if (e.key !== 'Tab') return;
      const nodes = popup.querySelectorAll(FOCUSABLE);
      if (!nodes.length) return;
      const first = nodes[0];
      const last = nodes[nodes.length - 1];
      if (e.shiftKey && document.activeElement === first) {
        e.preventDefault(); if (last instanceof HTMLElement) last.focus();
      } else if (!e.shiftKey && document.activeElement === last) {
        e.preventDefault(); if (first instanceof HTMLElement) first.focus();
      }
    }

    function open(msg) {
      lastActive = document.activeElement;
      // Build popup content without innerHTML
      const wrapper = document.createElement('div');
      wrapper.className = 'custom-popup';
      wrapper.setAttribute('role', 'dialog');
      wrapper.setAttribute('aria-modal', 'true');
      if (msg instanceof Node) {
        wrapper.appendChild(msg);
      } else {
        const msgWrap = document.createElement('div');
        msgWrap.textContent = String(msg || '');
        wrapper.appendChild(msgWrap);
      }
      wrapper.appendChild(document.createElement('br'));
      const closeBtn = document.createElement('button');
      closeBtn.type = 'button';
      closeBtn.setAttribute('data-popup-close', '');
      closeBtn.textContent = 'Close';
      wrapper.appendChild(closeBtn);
      // replace popup children
      while (popup.firstChild) popup.removeChild(popup.firstChild);
      popup.appendChild(wrapper);
      popup.classList.add('is-open');
      popup.addEventListener('keydown', trapFocus);
      const btn = popup.querySelector('[data-popup-close]');
      if (btn instanceof HTMLElement) btn.focus();
      if (btn) btn.addEventListener('click', close, { once: true });
      popup.addEventListener('click', backdropClose);
      document.addEventListener('keydown', escClose);
    }

    function close() {
      popup.classList.remove('is-open');
      // remove children safely
      while (popup.firstChild) popup.removeChild(popup.firstChild);
      popup.removeEventListener('click', backdropClose);
      document.removeEventListener('keydown', escClose);
      popup.removeEventListener('keydown', trapFocus);
      if (lastActive && typeof lastActive.focus === 'function') lastActive.focus();
    }

    function backdropClose(e) {
      if (e.target === popup) close();
    }
    function escClose(e) {
      if (e.key === 'Escape') close();
    }

    W.showPopup = (msg) => open(msg);
    W.closePopup = () => close();
  }

  /* ---------- Solar System Canvas ---------- */
  function initSolarSystem() {
    /** @type {HTMLCanvasElement|null} */
    const canvas = /** @type {HTMLCanvasElement|null} */ (document.getElementById('container'));
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    if (!ctx) return;

    const prefersReduce = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
    const DPR = Math.max(1, Math.min(2, window.devicePixelRatio || 1));

    function size() {
      const W = 300, H = 300;
      canvas.width = Math.round(W * DPR);
      canvas.height = Math.round(H * DPR);
    }
    size();
    window.addEventListener('resize', size);

    /* Assets */
    const { sunSvg, sunPng, moonSvg, moonPng, earth } = W.STATIC_ASSETS || {};
    const imgSun = new Image();
    const imgMoon = new Image();
    const imgEarth = new Image();
    let loaded = 0;
    function ready() { if (++loaded === 3 && !prefersReduce) requestAnimationFrame(draw); }
    function loadWithFallback(img, svg, png) {
      img.onload = ready;
      img.onerror = () => {
        if (!img.__fallback && png) { img.__fallback = true; img.src = png; }
        else ready();
      };
      img.src = svg;
    }
    loadWithFallback(imgSun, sunSvg, sunPng);
    loadWithFallback(imgMoon, moonSvg, moonPng);
    imgEarth.onload = ready;
    imgEarth.onerror = ready;
    imgEarth.src = earth;

    /* Stars */
    const stars = Array.from({ length: 40 }, () => ({
      x: Math.random() * 300,
      y: Math.random() * 300,
      r: 0.3 + Math.random() * 1.2,
      tw: 0.5 + Math.random() * 0.6,
      ph: Math.random() * Math.PI * 2
    }));

    /* Fractal background */
    const BG_W = 150, BG_H = 150, LOOP_MS = 24000;
    const bgCanvas = document.createElement('canvas');
    bgCanvas.width = BG_W; bgCanvas.height = BG_H;
    const bgCtx = bgCanvas.getContext('2d');

    function baseWave(x, y, u, o) {
      const A = 2 * Math.PI * u;
      const p1 = 0.9 + 0.1 * o;
      const p2 = 1.2 + 0.07 * o;
      return 0.33 * (
        Math.sin(x * 4 + 0.8 * Math.cos(A * p1)) +
        Math.sin(y * 4 + 0.7 * Math.sin(A * p2)) +
        Math.sin((x + y) * 2.2 + 0.5 * Math.cos(A * 0.6))
      );
    }
    function fbm2(x, y, u) {
      let v = 0, amp = 0.6, f = 1;
      for (let o = 0; o < 3; o++) {
        v += amp * baseWave(x * 3 * f, y * 3 * f, u, o);
        f *= 2.1;
        amp *= 0.55;
      }
      v = 0.5 + 0.5 * v;
      return Math.min(1, Math.max(0, v));
    }
    function shadeBlue(v) {
      const k = 0.35 + 0.65 * v;
      return [
        Math.floor(15 + 25 * k),
        Math.floor(40 + 80 * k),
        Math.floor(80 + 140 * k),
        Math.floor(120 + 90 * v)
      ];
    }
    function renderFractal(now) {
      const u = (now % LOOP_MS) / LOOP_MS;
      const img = bgCtx.createImageData(BG_W, BG_H);
      let p = 0;
      for (let y = 0; y < BG_H; y++) {
        for (let x = 0; x < BG_W; x++) {
          const nx = x / BG_W, ny = y / BG_H;
          const v = fbm2(nx, ny, u);
            const [r,g,b,a] = shadeBlue(v);
          img.data[p++] = r;
          img.data[p++] = g;
          img.data[p++] = b;
          img.data[p++] = a;
        }
      }
      bgCtx.putImageData(img, 0, 0);
    }

    /* Geometry & Timing */
    const CX = 150, CY = 150;
    const SUN_R = 52;
    const EARTH_ORBIT_R = 105;
    const MOON_ORBIT_R = 28.5;
    const EARTH_R = 8;
    const MOON_R = 2.5;
    const EARTH_ORBIT_MS = 60000;
    const EARTH_ROT_MS = Math.max(2000, EARTH_ORBIT_MS / 365.25);
    const MOON_ORBIT_MS = EARTH_ORBIT_MS * (27.3 / 365.25);
    const MOON_ROT_MS = MOON_ORBIT_MS;
    const SUNSPOT_ROT_MS = 90000;

    /* Sun texture */
    const sunTex = (() => {
      const os = document.createElement('canvas');
      os.width = os.height = 300;
      const c = os.getContext('2d');
      c.translate(CX, CY);
      for (let i = 0; i < 260; i++) {
        const a = Math.random() * Math.PI * 2;
        const r = Math.pow(Math.random(), 0.35) * (SUN_R - 6);
        const x = Math.cos(a) * r;
        const y = Math.sin(a) * r;
        const g = c.createRadialGradient(x, y, 0, x, y, 3);
        g.addColorStop(0, 'rgba(255,220,120,0.15)');
        g.addColorStop(1, 'rgba(255,170,60,0)');
        c.fillStyle = g;
        c.beginPath();
        c.arc(x, y, 3, 0, Math.PI * 2);
        c.fill();
      }
      return os;
    })();

    const sunspots = Array.from({ length: 6 }, () => ({
      a: Math.random() * Math.PI * 2,
      r: SUN_R * Math.sqrt(Math.random() * 0.9),
      size: rand(1.2, 3.2),
      dark: rand(0.35, 0.55)
    }));

    /* Flares */
    const flares = [];
    function rand(min, max) { return min + Math.random() * (max - min); }
    function spawnFlare(now) {
      const aMid = Math.random() * Math.PI * 2;
      const spread = rand(0.22, 0.55);
      const a0 = aMid - spread / 2;
      const a1 = aMid + spread / 2;
      flares.push({
        phase: 'loop',
        start: now,
        dur: rand(900, 1600),
        a0, a1, aMid,
        height: rand(SUN_R + 12, SUN_R + 42),
        hue: rand(10, 45),
        power: rand(0.45, 0.9),
        particles: []
      });
      if (flares.length > 5) flares.shift();
    }
    function drawFlares(now) {
      if (Math.random() < 0.015) spawnFlare(now);
      for (let i = flares.length - 1; i >= 0; i--) {
        const f = flares[i];
        if (f.phase === 'loop') {
          const t = Math.min(1, (now - f.start) / f.dur);
          const up = t < 0.5 ? (2 * t) : (2 - 2 * t);
          const ease = up * up * (3 - 2 * up);
          const x0 = CX + Math.cos(f.a0) * SUN_R;
          const y0 = CY + Math.sin(f.a0) * SUN_R;
          const x1 = CX + Math.cos(f.a1) * SUN_R;
          const y1 = CY + Math.sin(f.a1) * SUN_R;
          const aCtrl = f.aMid + (Math.random() - 0.5) * 0.2;
          const xc = CX + Math.cos(aCtrl) * (f.height * (0.7 + 0.3 * ease));
          const yc = CY + Math.sin(aCtrl) * (f.height * (0.7 + 0.3 * ease));
          ctx.save();
          ctx.globalCompositeOperation = 'lighter';
          ctx.shadowColor = `hsla(${f.hue},100%,60%,${0.35 * f.power})`;
          ctx.shadowBlur = 12 + 14 * f.power;
          ctx.strokeStyle = `hsla(${f.hue},100%,60%,${0.30 * f.power})`;
          ctx.lineWidth = 1.4 + 1.2 * f.power;
          ctx.beginPath();
          ctx.moveTo(x0, y0);
          ctx.quadraticCurveTo(xc, yc, x1, y1);
          ctx.stroke();
          for (let b = 0; b < 4; b++) {
            const u = ((t + b / 4) % 1);
            const bx = (1 - u) ** 2 * x0 + 2 * (1 - u) * u * xc + u * u * x1;
            const by = (1 - u) ** 2 * y0 + 2 * (1 - u) * u * yc + u * u * y1;
            ctx.beginPath();
            ctx.arc(bx, by, 0.8 + 0.9 * f.power, 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${f.hue},100%,65%,${0.35 * f.power})`;
            ctx.fill();
          }
          ctx.restore();
          if (t >= 1) {
            const n = Math.round(24 * f.power);
            for (let p = 0; p < n; p++) {
              const ang = f.aMid + rand(-0.15, 0.15);
              f.particles.push({
                x: CX + Math.cos(ang) * (SUN_R + rand(8, 20)),
                y: CY + Math.sin(ang) * (SUN_R + rand(8, 20)),
                vx: Math.cos(ang) * rand(0.10, 0.35),
                vy: Math.sin(ang) * rand(0.10, 0.35),
                life: 0,
                max: rand(900, 1700),
                size: rand(0.6, 1.6),
                hue: f.hue,
                alpha: rand(0.25, 0.45)
              });
            }
            f.phase = 'eject';
            f.start = now;
            f.dur = rand(1200, 2200);
          }
        } else {
          const parts = f.particles;
          ctx.save();
          ctx.globalCompositeOperation = 'lighter';
          for (let p = parts.length - 1; p >= 0; p--) {
            const pt = parts[p];
            pt.life += 16.6;
            pt.x += pt.vx * (1 + 0.6 * Math.random());
            pt.y += pt.vy * (1 + 0.6 * Math.random());
            const fade = Math.max(0, 1 - pt.life / pt.max);
            if (fade <= 0) { parts.splice(p, 1); continue; }
            ctx.beginPath();
            ctx.arc(pt.x, pt.y, pt.size * (0.6 + 0.8 * fade), 0, Math.PI * 2);
            ctx.fillStyle = `hsla(${pt.hue},100%,${60 - 10 * (1 - fade)}%,${pt.alpha * fade})`;
            ctx.shadowColor = `hsla(${pt.hue},100%,60%,${0.25 * fade})`;
            ctx.shadowBlur = 8;
            ctx.fill();
          }
          ctx.restore();
          if (!parts.length) flares.splice(i, 1);
        }
      }
    }

    /* Sun rendering */
    function drawSun(now) {
      const pulse = 1 + 0.025 * Math.sin(now / 700) + 0.018 * Math.sin(now / 1230);
      const innerR = SUN_R * pulse;
      const outerR = 140;
      const g1 = ctx.createRadialGradient(CX, CY, 0, CX, CY, innerR + 10);
      g1.addColorStop(0, 'rgba(255,235,140,0.95)');
      g1.addColorStop(0.65, 'rgba(255,200,80,0.55)');
      g1.addColorStop(1, 'rgba(255,160,50,0.06)');
      ctx.beginPath();
      ctx.arc(CX, CY, innerR + 10, 0, Math.PI * 2);
      ctx.fillStyle = g1;
      ctx.fill();

      ctx.save();
      ctx.globalCompositeOperation = 'lighter';
      ctx.globalAlpha = 0.28;
      ctx.drawImage(sunTex, 0, 0);
      ctx.restore();

      // Sunspots
      ctx.save();
      ctx.beginPath();
      ctx.arc(CX, CY, innerR, 0, Math.PI * 2);
      ctx.clip();
      const sunRot = (now % SUNSPOT_ROT_MS) / SUNSPOT_ROT_MS * Math.PI * 2;
      for (const s of sunspots) {
        const a = s.a + sunRot;
        const x = CX + Math.cos(a) * s.r;
        const y = CY + Math.sin(a) * s.r;
        const spot = ctx.createRadialGradient(x, y, 0, x, y, s.size * 2.4);
        spot.addColorStop(0, `rgba(0,0,0,${s.dark})`);
        spot.addColorStop(1, 'rgba(0,0,0,0)');
        ctx.fillStyle = spot;
        ctx.beginPath();
        ctx.arc(x, y, s.size * 2.4, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();

      const g2 = ctx.createRadialGradient(CX, CY, innerR, CX, CY, outerR);
      g2.addColorStop(0, 'rgba(255,180,0,0.22)');
      g2.addColorStop(0.65, 'rgba(255,180,0,0.10)');
      g2.addColorStop(1, 'rgba(255,180,0,0.00)');
      ctx.beginPath();
      ctx.arc(CX, CY, outerR, 0, Math.PI * 2);
      ctx.fillStyle = g2;
      ctx.fill();

      if (imgSun.complete) {
        ctx.save();
        ctx.globalCompositeOperation = 'lighter';
        ctx.globalAlpha = 0.5;
        ctx.drawImage(imgSun, 0, 0, 300, 300);
        ctx.restore();
      }
    }

    function drawBodyWithShadow(x, y, r, img, rot, lightAngle, fallback) {
      ctx.save();
      ctx.translate(x, y);
      if (rot) ctx.rotate(rot);
      if (img && img.complete) ctx.drawImage(img, -r, -r, r * 2, r * 2);
      else {
        const g = ctx.createRadialGradient(0, 0, 0, 0, 0, r);
        g.addColorStop(0, fallback || '#bbb');
        g.addColorStop(1, '#aaa');
        ctx.fillStyle = g;
        ctx.beginPath();
        ctx.arc(0, 0, r, 0, Math.PI * 2);
        ctx.fill();
      }
      ctx.restore();

      // Limb
      ctx.save();
      ctx.globalCompositeOperation = 'multiply';
      ctx.translate(x, y);
      const limb = ctx.createRadialGradient(0, 0, 0, 0, 0, r);
      limb.addColorStop(0, 'rgba(0,0,0,0)');
      limb.addColorStop(0.7, 'rgba(0,0,0,0.05)');
      limb.addColorStop(1, 'rgba(0,0,0,0.18)');
      ctx.fillStyle = limb;
      ctx.beginPath();
      ctx.arc(0, 0, r, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      // Terminator
      ctx.save();
      ctx.translate(x, y);
      ctx.beginPath();
      ctx.arc(0, 0, r, 0, Math.PI * 2);
      ctx.clip();
      ctx.globalCompositeOperation = 'multiply';
      const gx1 = Math.cos(lightAngle + Math.PI) * r;
      const gy1 = Math.sin(lightAngle + Math.PI) * r;
      const gx2 = Math.cos(lightAngle) * r;
      const gy2 = Math.sin(lightAngle) * r;
      const grad = ctx.createLinearGradient(gx2, gy2, gx1, gy1);
      grad.addColorStop(0, 'rgba(0,0,0,0)');
      grad.addColorStop(.38, 'rgba(0,0,0,0.18)');
      grad.addColorStop(.62, 'rgba(0,0,0,0.60)');
      grad.addColorStop(1, 'rgba(0,0,0,0.90)');
      ctx.fillStyle = grad;
      ctx.fillRect(-r - 2, -r - 2, (r + 2) * 2, (r + 2) * 2);
      ctx.restore();

      // Specular
      const hx = x + Math.cos(lightAngle) * (r * 0.55);
      const hy = y + Math.sin(lightAngle) * (r * 0.55);
      ctx.save();
      const spec = ctx.createRadialGradient(hx, hy, 0, hx, hy, r * 0.6);
      spec.addColorStop(0, 'rgba(255,255,255,0.18)');
      spec.addColorStop(1, 'rgba(255,255,255,0)');
      ctx.globalCompositeOperation = 'screen';
      ctx.fillStyle = spec;
      ctx.beginPath();
      ctx.arc(hx, hy, r * 0.7, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();

      // Outline
      ctx.save();
      ctx.beginPath();
      ctx.arc(x, y, r + 0.3, 0, Math.PI * 2);
      ctx.strokeStyle = 'rgba(255,255,255,0.5)';
      ctx.lineWidth = 0.7;
      ctx.stroke();
      ctx.restore();
    }

    function castShadowOnTarget(sunX, sunY, casterX, casterY, casterR, targetX, targetY, targetR, strength = 1) {
      const dx = casterX - sunX, dy = casterY - sunY;
      const dist = Math.hypot(dx, dy);
      if (dist < 1e-6) return;
      const ux = dx / dist, uy = dy / dist;
      const tx = targetX - sunX, ty = targetY - sunY;
      const tTarget = tx * ux + ty * uy;
      const tCaster = dist;
      if (tTarget <= tCaster) return;
      const px = sunX + ux * tTarget;
      const py = sunY + uy * tTarget;
      if (Math.hypot(px - targetX, py - targetY) > targetR * 1.3) return;
      const dtc = Math.hypot(targetX - casterX, targetY - casterY);
      let rUmbra = casterR * (dtc / dist);
      rUmbra = Math.min(targetR * 0.6, Math.max(0.2, rUmbra));
      let rPen = Math.min(targetR * 0.8, rUmbra * 1.8);
      const vLx = sunX - targetX, vLy = sunY - targetY;
      const vLL = Math.hypot(vLx, vLy) || 1;
      const lUx = vLx / vLL, lUy = vLy / vLL;
      const vSx = px - targetX, vSy = py - targetY;
      const sL = Math.hypot(vSx, vSy) || 1;
      const sUx = vSx / sL, sUy = vSy / sL;
      const dayDot = Math.max(0, lUx * sUx + lUy * sUy);
      if (dayDot <= 0.02) return;
      ctx.save();
      ctx.beginPath();
      ctx.arc(targetX, targetY, targetR, 0, Math.PI * 2);
      ctx.clip();
      ctx.globalCompositeOperation = 'multiply';
      const g = ctx.createRadialGradient(px, py, rUmbra * 0.6, px, py, rPen);
      g.addColorStop(0, `rgba(0,0,0,${(0.45 * strength * dayDot).toFixed(3)})`);
      g.addColorStop(.55, `rgba(0,0,0,${(0.14 * strength * dayDot).toFixed(3)})`);
      g.addColorStop(1, 'rgba(0,0,0,0)');
      ctx.fillStyle = g;
      ctx.beginPath();
      ctx.arc(px, py, rPen, 0, Math.PI * 2);
      ctx.fill();
      ctx.restore();
    }

    function draw(now) {
      ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
      ctx.clearRect(0, 0, 300, 300);

      renderFractal(now);
      ctx.save();
      ctx.globalAlpha = 0.55;
      ctx.globalCompositeOperation = 'lighter';
      ctx.drawImage(bgCanvas, 0, 0, 300, 300);
      ctx.restore();

      // Stars
      stars.forEach(s => {
        const a = 0.25 + 0.55 * (0.5 + 0.5 * Math.sin(now / 1000 * s.tw + s.ph));
        ctx.save();
        ctx.globalAlpha = a;
        ctx.beginPath();
        ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
        ctx.fillStyle = '#9cc5ff';
        ctx.shadowColor = '#9cc5ff';
        ctx.shadowBlur = 2;
        ctx.fill();
        ctx.restore();
      });

      drawSun(now);
      drawFlares(now);

      // Orbits
      ctx.beginPath();
      ctx.arc(CX, CY, EARTH_ORBIT_R, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(0,255,231,0.18)";
      ctx.lineWidth = 2;
      ctx.stroke();

      const earthOrbitAngle = (now % EARTH_ORBIT_MS) / EARTH_ORBIT_MS * Math.PI * 2;
      const earthSpinAngle = (now % EARTH_ROT_MS) / EARTH_ROT_MS * Math.PI * 2;
      const ex = CX + EARTH_ORBIT_R * Math.cos(earthOrbitAngle);
      const ey = CY + EARTH_ORBIT_R * Math.sin(earthOrbitAngle);

      ctx.beginPath();
      ctx.arc(ex, ey, MOON_ORBIT_R, 0, Math.PI * 2);
      ctx.strokeStyle = "rgba(0,255,231,0.10)";
      ctx.lineWidth = 1.2;
      ctx.stroke();

      const moonOrbitAngle = (now % MOON_ORBIT_MS) / MOON_ORBIT_MS * Math.PI * 2;
      const moonSpinAngle = (now % MOON_ROT_MS) / MOON_ROT_MS * Math.PI * 2;
      const mx = ex + MOON_ORBIT_R * Math.cos(moonOrbitAngle);
      const my = ey + MOON_ORBIT_R * Math.sin(moonOrbitAngle);

      drawBodyWithShadow(ex, ey, EARTH_R, imgEarth, earthSpinAngle, Math.atan2(CY - ey, CX - ex), '#4aa3ff');
      drawBodyWithShadow(mx, my, MOON_R, imgMoon, moonSpinAngle, Math.atan2(CY - my, CX - mx), '#d0d0d0');

      castShadowOnTarget(CX, CY, mx, my, MOON_R, ex, ey, EARTH_R, 1.0);
      castShadowOnTarget(CX, CY, ex, ey, EARTH_R, mx, my, MOON_R, 0.85);

      if (!prefersReduce) requestAnimationFrame(draw);
    }

    if (!prefersReduce) requestAnimationFrame(draw);

    canvas.addEventListener('click', () => {
      const message = buildPopupMessage(
        '🌞 You clicked the solar system!',
        'Keep exploring the universe of transparent data!'
      );
      W.showPopup?.(message);
    });
  }

  /* ---------- Init All ---------- */
  document.addEventListener('DOMContentLoaded', () => {
    initBootstrap();
    initRimTracer();
    initFeatureHover();
    initPopup();
    initSolarSystem();
  });

})();