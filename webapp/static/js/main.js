document.addEventListener('DOMContentLoaded', function () {
  // Enable Bootstrap tooltips and popovers if present
  if (window.bootstrap) {
    const tEls = [].slice.call(document.querySelectorAll('[data-bs-toggle="tooltip"]'));
    tEls.forEach(el => bootstrap.Tooltip.getOrCreateInstance(el));
    const pEls = [].slice.call(document.querySelectorAll('[data-bs-toggle="popover"]'));
    pEls.forEach(el => bootstrap.Popover.getOrCreateInstance(el));
  }

  // Rim tracer for mission panel (subtle, smooth, with sparks)
  const rimCanvas = document.getElementById('rimTracer');
  const missionPanel = document.getElementById('missionPanel');
  let sparks = [];

  function resizeRimCanvas() {
    if (!rimCanvas || !missionPanel) return;
    rimCanvas.width = missionPanel.offsetWidth;
    rimCanvas.height = missionPanel.offsetHeight;
  }

  function getPos(d, w, h, r) {
    if (d < w - 2 * r) return { x: r + d, y: r, a: 0 };
    d -= (w - 2 * r);
    if (d < r * Math.PI / 2) return { x: w - r + r * Math.cos(Math.PI / 2 - d / r), y: r + r * Math.sin(Math.PI / 2 - d / r), a: Math.PI / 2 - d / r };
    d -= r * Math.PI / 2;
    if (d < h - 2 * r) return { x: w - r, y: r + d, a: Math.PI / 2 };
    d -= (h - 2 * r);
    if (d < r * Math.PI / 2) return { x: w - r + r * Math.sin(d / r), y: h - r + r * Math.cos(d / r), a: Math.PI + d / r };
    d -= r * Math.PI / 2;
    if (d < w - 2 * r) return { x: w - r - d, y: h - r, a: Math.PI };
    d -= (w - 2 * r);
    if (d < r * Math.PI / 2) return { x: r + r * Math.cos(3 * Math.PI / 2 + d / r), y: h - r + r * Math.sin(3 * Math.PI / 2 + d / r), a: 3 * Math.PI / 2 + d / r };
    d -= r * Math.PI / 2;
    return { x: r, y: h - r - d, a: 3 * Math.PI / 2 };
  }

  function animateRimTracer() {
    if (!rimCanvas) return;
    const ctx = rimCanvas.getContext('2d');
    ctx.clearRect(0, 0, rimCanvas.width, rimCanvas.height);
    const w = rimCanvas.width, h = rimCanvas.height;
    const r = 3;

    const SPEED_DIVISOR = 3;
    const t = Date.now() / (2200 * SPEED_DIVISOR);
    const perim = 2 * (w + h - 4 * r) + 2 * Math.PI * r;
    const speed = (t % 1) * perim;

    const TRAIL_HEAD_RADIUS = 5;
    const TRAIL_TAIL_RADIUS = 1;
    const TRAIL_STEPS = 8;

    for (let i = 1; i <= TRAIL_STEPS; i++) {
      let d = (speed - i * 7 + perim) % perim;
      let { x, y } = getPos(d, w, h, r);
      let radius = TRAIL_TAIL_RADIUS + (TRAIL_HEAD_RADIUS - TRAIL_TAIL_RADIUS) * ((TRAIL_STEPS - i) / (TRAIL_STEPS - 1));
      ctx.save();
      ctx.globalAlpha = 0.03 + 0.04 * (TRAIL_STEPS - i + 1);
      ctx.shadowColor = "#00ffe7";
      ctx.shadowBlur = 16 + ((TRAIL_STEPS - i + 1) * 2);
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, 2 * Math.PI);
      ctx.fillStyle = "#00ffe7";
      ctx.fill();
      ctx.restore();
    }

    let { x, y, a } = getPos(speed, w, h, r);
    ctx.save();
    ctx.globalAlpha = 0.18;
    ctx.shadowColor = "#00ffe7";
    ctx.shadowBlur = 18;
    ctx.beginPath();
    ctx.arc(x, y, TRAIL_HEAD_RADIUS, 0, 2 * Math.PI);
    ctx.fillStyle = "#00ffe7";
    ctx.fill();
    ctx.restore();

    ctx.save();
    ctx.strokeStyle = "rgba(0,255,231,0.08)";
    ctx.shadowColor = "#00ffe7";
    ctx.shadowBlur = 6;
    ctx.lineWidth = 3;
    ctx.beginPath();
    ctx.moveTo(r, 0);
    ctx.lineTo(w - r, 0);
    ctx.arcTo(w, 0, w, r, r);
    ctx.lineTo(w, h - r);
    ctx.arcTo(w, h, w - r, h, r);
    ctx.lineTo(r, h);
    ctx.arcTo(0, h, 0, h - r, r);
    ctx.lineTo(0, r);
    ctx.arcTo(0, 0, r, 0, r);
    ctx.closePath();
    ctx.stroke();
    ctx.restore();

    if (Math.random() < 0.08) {
      let angle = a + (Math.random() - 0.5) * 0.7;
      let len = 12 + Math.random() * 12;
      sparks.push({
        x, y,
        dx: Math.cos(angle) * len,
        dy: Math.sin(angle) * len,
        alpha: 0.5 + Math.random() * 0.2,
        life: 0
      });
    }
    for (let i = sparks.length - 1; i >= 0; i--) {
      let s = sparks[i];
      s.life += 1;
      let px = s.x + s.dx * (s.life / 10);
      let py = s.y + s.dy * (s.life / 10);
      ctx.save();
      ctx.globalAlpha = s.alpha * (1 - s.life / 10);
      ctx.shadowColor = "#00ffe7";
      ctx.shadowBlur = 8;
      ctx.beginPath();
      ctx.arc(px, py, 1.5, 0, 2 * Math.PI);
      ctx.fillStyle = "#00ffe7";
      ctx.fill();
      ctx.restore();
      if (s.life > 10) sparks.splice(i, 1);
    }
    requestAnimationFrame(animateRimTracer);
  }

  resizeRimCanvas();
  animateRimTracer();
  window.addEventListener('resize', resizeRimCanvas);

  // Glossy hover effect (subtle)
  if (missionPanel) {
    missionPanel.addEventListener('mouseenter', () => missionPanel.classList.add('glossy'));
    missionPanel.addEventListener('mouseleave', () => missionPanel.classList.remove('glossy'));
  }

  // Feature hover effect
  const featureStyles = {
    data_framework: { color: "#45818e", transform: "translate(-8px, -8px) scale(1.06) rotate(-2deg)" },
    run_parser: { color: "#00ffe7", transform: "translateY(-8px) scale(1.06) rotate(0deg)" },
    history: { color: "#ffd700", transform: "translate(8px, -8px) scale(1.06) rotate(2deg)" }
  };

  document.querySelectorAll('.feature').forEach(btn => {
    btn.addEventListener('mousemove', function () {
      let style = { color: "#eb4f43", transform: "" };
      if (this.href.includes('data_framework')) style = featureStyles.data_framework;
      else if (this.href.includes('run_parser')) style = featureStyles.run_parser;
      else if (this.href.includes('history')) style = featureStyles.history;

      this.style.boxShadow = `0 0 24px 4px ${style.color}, 0 2px 12px #bfc9d1`;
      this.style.transform = style.transform;
    });
    btn.addEventListener('mouseleave', function () {
      this.style.boxShadow = "";
      this.style.transform = "";
    });
  });

  // Canvas animation (solar system)
  const canvasEl = document.getElementById('container');
  if (!canvasEl) return;

  // HiDPI setup
  const DPR = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
  function setupCanvasSize() {
    const cssW = 300, cssH = 300;
    canvasEl.width = Math.round(cssW * DPR);
    canvasEl.height = Math.round(cssH * DPR);
    canvasEl.style.width = cssW + 'px';
    canvasEl.style.height = cssH + 'px';
  }
  setupCanvasSize();
  window.addEventListener('resize', setupCanvasSize);

  // Assets
  const sun = new Image();
  const moon = new Image();
  const earth = new Image();
  let imagesLoaded = 0;

  function tryStart() {
    imagesLoaded++;
    if (imagesLoaded === 3) window.requestAnimationFrame(draw);
  }
  function loadWithFallback(img, svgPath, pngPath) {
    img.onload = tryStart;
    img.onerror = function () {
      if (!img.__triedPng && pngPath) {
        img.__triedPng = true;
        img.src = pngPath;
      } else {
        console.warn("Falling back to vector placeholder for:", svgPath || pngPath);
        tryStart();
      }
    };
    img.src = svgPath;
  }
  const ASSETS = window.STATIC_ASSETS || {};
  loadWithFallback(sun, ASSETS.sunSvg, ASSETS.sunPng);
  loadWithFallback(moon, ASSETS.moonSvg, ASSETS.moonPng);
  earth.onload = tryStart;
  earth.onerror = function () { console.warn("Falling back to vector placeholder for earth.png"); tryStart(); };
  earth.src = ASSETS.earth;

  // Subtle starfield
  const stars = Array.from({ length: 40 }, () => ({
    x: Math.random() * 300,
    y: Math.random() * 300,
    r: 0.3 + Math.random() * 1.2,
    tw: 0.5 + Math.random() * 0.6,
    phase: Math.random() * Math.PI * 2
  }));


  // Tileable fractal background (seamless loop)
  const BG_W = 150, BG_H = 150;
  const bgCanvas = document.createElement('canvas');
  bgCanvas.width = BG_W; bgCanvas.height = BG_H;
  const bgCtx = bgCanvas.getContext('2d', { willReadFrequently: false });
  const LOOP_MS = 24000;

  function baseWave(x, y, u, o) {
    const a = 2 * Math.PI * u;
    // Per-octave phase that follows a circle in time -> perfect loop
    const p1 = 0.9 + 0.1 * o;
    const p2 = 1.2 + 0.07 * o;
    return (
      0.33 *
      (Math.sin(x * 4.0 + 0.8 * Math.cos(a * p1)) +
       Math.sin(y * 4.0 + 0.7 * Math.sin(a * p2)) +
       Math.sin((x + y) * 2.2 + 0.5 * Math.cos(a * 0.6)))
    );
  }

  function fbm2(x, y, u) {
    // Fractal sum of base waves with looping time
    let v = 0, amp = 0.6, freq = 1.0;
    for (let o = 0; o < 3; o++) {
      v += amp * baseWave(x * 3 * freq, y * 3 * freq, u, o);
      freq *= 2.1;
      amp *= 0.55;
    }
    v = 0.5 + 0.5 * v; // map to 0..1
    return Math.max(0, Math.min(1, v));
  }

  function shadeBlue(v) {
    // Blue-nebula palette
    const k = 0.35 + 0.65 * v;
    const r = Math.floor(15 + 25 * k);
    const g = Math.floor(40 + 80 * k);
    const b = Math.floor(80 + 140 * k);
    const a = Math.floor(120 + 90 * v); // 120..210
    return [r, g, b, a];
  }

  function renderFractalBackground(now) {
    const u = (now % LOOP_MS) / LOOP_MS; // 0..1 loop
    const img = bgCtx.createImageData(BG_W, BG_H);
    let p = 0;
    for (let j = 0; j < BG_H; j++) {
      const y = j / BG_H;
      for (let i = 0; i < BG_W; i++) {
        const x = i / BG_W;
        const v = fbm2(x, y, u);
        const [r, g, b, a] = shadeBlue(v);
        img.data[p++] = r;
        img.data[p++] = g;
        img.data[p++] = b;
        img.data[p++] = a;
      }
    }
    bgCtx.putImageData(img, 0, 0);
  }

  // Solar flares (coronal loops + CME)
  const flares = [];
  function rand(min, max) { return min + Math.random() * (max - min); }

  // Periods (scaled realism)
  const EARTH_ORBIT_MS = 60000;                                 // ~60s per "year"
  const EARTH_ROT_MS = Math.max(2000, EARTH_ORBIT_MS / 365.25); // min 2s for visibility
  const MOON_ORBIT_MS = EARTH_ORBIT_MS * (27.3 / 365.25);       // ~4.49s per lunar month
  const MOON_ROT_MS = MOON_ORBIT_MS;                            // tidal locking

  // Geometry
  const CX = 150, CY = 150;   // CSS pixels (scaled by DPR via setTransform)
  const SUN_R = 52;           // base sun radius used for loops/spots
  const EARTH_ORBIT_R = 105;
  const MOON_ORBIT_R = 28.5;
  const EARTH_R = 8;
  const MOON_R = 2.5;

  // Precomputed sun granulation pattern (offscreen)
  const sunTex = (() => {
    const os = document.createElement('canvas');
    os.width = os.height = 300;
    const c = os.getContext('2d');
    c.translate(CX, CY);
    for (let i = 0; i < 260; i++) {
      const a = Math.random() * Math.PI * 2;
      const r = Math.pow(Math.random(), 0.35) * (SUN_R - 6);
      const x = Math.cos(a) * r, y = Math.sin(a) * r;
      const gr = c.createRadialGradient(x, y, 0, x, y, rand(1.8, 3.2));
      gr.addColorStop(0, 'rgba(255,220,120,0.15)');
      gr.addColorStop(1, 'rgba(255,170,60,0)');
      c.fillStyle = gr;
      c.beginPath();
      c.arc(x, y, rand(2.0, 3.6), 0, Math.PI * 2);
      c.fill();
    }
    return os;
  })();

  // Sunspots (slow rotation)
  const sunspots = Array.from({ length: 6 }, () => ({
    a: Math.random() * Math.PI * 2,
    r: SUN_R * Math.sqrt(Math.random() * 0.9),
    size: rand(1.2, 3.2),
    dark: rand(0.35, 0.55)
  }));
  const SUNSPOT_ROT_MS = 90000; // slow drift

  // Improved body rendering with limb-darkening and specular on lit side
  function drawBodyWithShadow(ctx, x, y, r, img, rotationAngle, lightAngle, fallbackColor) {
    // Body or texture
    ctx.save();
    ctx.translate(x, y);
    if (rotationAngle) ctx.rotate(rotationAngle);
    if (img && img.complete) {
      ctx.drawImage(img, -r, -r, 2 * r, 2 * r);
    } else {
      const rg = ctx.createRadialGradient(0, 0, 0, 0, 0, r);
      rg.addColorStop(0, fallbackColor || '#bbb');
      rg.addColorStop(1, '#aaa');
      ctx.fillStyle = rg;
      ctx.beginPath();
      ctx.arc(0, 0, r, 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.restore();

    // Limb darkening (multiply)
    ctx.save();
    ctx.globalCompositeOperation = 'multiply';
    ctx.translate(x, y);
    const limb = ctx.createRadialGradient(0, 0, 0, 0, 0, r);
    limb.addColorStop(0.0, 'rgba(0,0,0,0)');
    limb.addColorStop(0.7, 'rgba(0,0,0,0.05)');
    limb.addColorStop(1.0, 'rgba(0,0,0,0.18)');
    ctx.fillStyle = limb;
    ctx.beginPath();
    ctx.arc(0, 0, r, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();

    // Stronger non-rotating day/night terminator (from sun direction)
    ctx.save();
    ctx.translate(x, y);
    ctx.beginPath();
    ctx.arc(0, 0, r, 0, Math.PI * 2);
    ctx.clip();

    // Multiply to darken underlying texture/color realistically
    ctx.globalCompositeOperation = 'multiply';

    const gx1 = Math.cos(lightAngle + Math.PI) * r;
    const gy1 = Math.sin(lightAngle + Math.PI) * r;
    const gx2 = Math.cos(lightAngle) * r;
    const gy2 = Math.sin(lightAngle) * r;

    const grad = ctx.createLinearGradient(gx2, gy2, gx1, gy1);
    // At least 3x darker on the night side
    grad.addColorStop(0.00, 'rgba(0,0,0,0.00)'); // fully lit side
    grad.addColorStop(0.38, 'rgba(0,0,0,0.18)'); // softer penumbra start
    grad.addColorStop(0.62, 'rgba(0,0,0,0.60)'); // much darker mid
    grad.addColorStop(1.00, 'rgba(0,0,0,0.90)'); // deep night
    ctx.fillStyle = grad;
    ctx.fillRect(-r - 2, -r - 2, (r + 2) * 2, (r + 2) * 2);
    ctx.restore();

    // Small specular highlight on lit side (subtle)
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

    // Outline for visibility
    ctx.save();
    ctx.beginPath();
    ctx.arc(x, y, r + 0.3, 0, Math.PI * 2);
    ctx.strokeStyle = 'rgba(255,255,255,0.5)';
    ctx.lineWidth = 0.7;
    ctx.stroke();
    ctx.restore();
  }

  // Cast a soft eclipse shadow from a caster onto a target, using sun direction.
  // Computes the point where the ray (sun -> caster) crosses the target's plane and
  // draws an umbra/penumbra blob clipped to the target disc, using multiply blending.
  function castShadowOnTarget(ctx, sunX, sunY, casterX, casterY, casterR, targetX, targetY, targetR, strength = 1.0) {
    // Direction from sun to caster (unit)
    const dx = casterX - sunX, dy = casterY - sunY;
    const dsm = Math.hypot(dx, dy);
    if (dsm < 1e-6) return;
    const ux = dx / dsm, uy = dy / dsm;

    // Distance along the ray to the target center and to the caster
    const tx = targetX - sunX, ty = targetY - sunY;
    const t_target = tx * ux + ty * uy;
    const t_caster = dsm;

    if (t_target <= t_caster) return;

    const px = sunX + ux * t_target;
    const py = sunY + uy * t_target;

    const off = Math.hypot(px - targetX, py - targetY);
    if (off > targetR * 1.3) return;

    const dtc = Math.hypot(targetX - casterX, targetY - casterY);
    let rUmbra = casterR * (dtc / dsm);
    rUmbra = Math.max(0.2, Math.min(rUmbra, targetR * 0.6));

    // Less dramatic penumbra
    let rPen = Math.min(targetR * 0.8, rUmbra * 1.8);

    // Only on day side
    const vLightX = sunX - targetX, vLightY = sunY - targetY;
    const vLightLen = Math.hypot(vLightX, vLightY) || 1;
    const lUx = vLightX / vLightLen, lUy = vLightY / vLightLen;
    const vSpotX = px - targetX, vSpotY = py - targetY;
    const vSpotLen = Math.hypot(vSpotX, vSpotY) || 1;
    const sUx = vSpotX / vSpotLen, sUy = vSpotY / vSpotLen;
    const dayDot = Math.max(0, lUx * sUx + lUy * sUy);
    if (dayDot <= 0.02) return;

    ctx.save();
    ctx.beginPath();
    ctx.arc(targetX, targetY, targetR, 0, Math.PI * 2);
    ctx.clip();

    ctx.globalCompositeOperation = 'multiply';

    const g = ctx.createRadialGradient(px, py, rUmbra * 0.6, px, py, rPen);
    const centerAlpha = 0.45 * strength * dayDot; // strong umbra
    const midAlpha = 0.14 * strength * dayDot;    // slightly softer penumbra
    g.addColorStop(0.0, `rgba(0,0,0,${centerAlpha.toFixed(3)})`);
    g.addColorStop(0.55, `rgba(0,0,0,${midAlpha.toFixed(3)})`);
    g.addColorStop(1.0, 'rgba(0,0,0,0)');
    ctx.fillStyle = g;
    ctx.beginPath();
    ctx.arc(px, py, rPen, 0, Math.PI * 2);
    ctx.fill();

    ctx.restore();
  }

  function drawSunAndCorona(ctx, now) {
    // Pulsating core
    const pulse = 1 + 0.025 * Math.sin(now / 700) + 0.018 * Math.sin(now / 1230);
    const innerR = SUN_R * pulse;
    const outerR = 140;

    // Core glow
    const g1 = ctx.createRadialGradient(CX, CY, 0, CX, CY, innerR + 10);
    g1.addColorStop(0, 'rgba(255,235,140,0.95)');
    g1.addColorStop(0.65, 'rgba(255,200,80,0.55)');
    g1.addColorStop(1, 'rgba(255,160,50,0.06)');
    ctx.beginPath();
    ctx.arc(CX, CY, innerR + 10, 0, Math.PI * 2);
    ctx.fillStyle = g1;
    ctx.fill();

    // Granulation texture (additive, subtle)
    ctx.save();
    ctx.globalCompositeOperation = 'lighter';
    ctx.globalAlpha = 0.28;
    ctx.drawImage(sunTex, 0, 0);
    ctx.restore();

    // Sunspots (darken within disc)
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

    // Corona (outer glow)
    const g2 = ctx.createRadialGradient(CX, CY, innerR, CX, CY, outerR);
    g2.addColorStop(0, 'rgba(255,180,0,0.22)');
    g2.addColorStop(0.65, 'rgba(255,180,0,0.10)');
    g2.addColorStop(1, 'rgba(255,180,0,0.00)');
    ctx.beginPath();
    ctx.arc(CX, CY, outerR, 0, Math.PI * 2);
    ctx.fillStyle = g2;
    ctx.fill();

    // Optional sun image blended additively
    if (sun && sun.complete) {
      ctx.save();
      ctx.globalCompositeOperation = 'lighter';
      ctx.globalAlpha = 0.5;
      ctx.drawImage(sun, 0, 0, 300, 300);
      ctx.restore();
    }
  }

  function spawnFlare(now) {
    // Two nearby footpoints on the surface
    const aMid = Math.random() * Math.PI * 2;
    const spread = rand(0.22, 0.55);
    const a0 = aMid - spread / 2;
    const a1 = aMid + spread / 2;
    const height = rand(SUN_R + 12, SUN_R + 42); // loop apex
    const hue = rand(10, 45);                    // reddish/orange
    const power = rand(0.45, 0.9);

    flares.push({
      phase: 'loop',
      start: now,
      dur: rand(900, 1600),
      a0, a1, aMid, height, hue, power,
      particles: []
    });

    // Cap the number of active flares
    if (flares.length > 5) flares.shift();
  }

  function drawFlares(ctx, now) {
    // Spawn with small probability
    if (Math.random() < 0.015) spawnFlare(now);

    for (let i = flares.length - 1; i >= 0; i--) {
      const f = flares[i];

      if (f.phase === 'loop') {
        const t = Math.min(1, (now - f.start) / f.dur);
        // Ease out-and-back
        const up = t < 0.5 ? (2 * t) : (2 - 2 * t);
        const ease = up * up * (3 - 2 * up); // smoothstep

        // Footpoints on the surface
        const x0 = CX + Math.cos(f.a0) * SUN_R;
        const y0 = CY + Math.sin(f.a0) * SUN_R;
        const x1 = CX + Math.cos(f.a1) * SUN_R;
        const y1 = CY + Math.sin(f.a1) * SUN_R;

        // Apex control point (bends slightly around mid-angle)
        const aCtrl = f.aMid + (Math.random() - 0.5) * 0.2;
        const xc = CX + Math.cos(aCtrl) * (f.height * (0.7 + 0.3 * ease));
        const yc = CY + Math.sin(aCtrl) * (f.height * (0.7 + 0.3 * ease));

        // Draw bright loop
        ctx.save();
        ctx.globalCompositeOperation = 'lighter';
        ctx.shadowColor = `hsla(${f.hue},100%,60%,${0.35 * f.power})`;
        ctx.shadowBlur = 12 + 14 * f.power;
        ctx.strokeStyle = `hsla(${f.hue},100%,${60}%,${0.30 * f.power})`;
        ctx.lineWidth = 1.4 + 1.2 * f.power;

        ctx.beginPath();
        ctx.moveTo(x0, y0);
        ctx.quadraticCurveTo(xc, yc, x1, y1);
        ctx.stroke();

        // Moving beads along the loop
        const beads = 4;
        for (let b = 0; b < beads; b++) {
          const pct = (t + b / beads) % 1;
          const u = pct;
          // Quadratic Bezier point
          const bx = (1 - u) * (1 - u) * x0 + 2 * (1 - u) * u * xc + u * u * x1;
          const by = (1 - u) * (1 - u) * y0 + 2 * (1 - u) * u * yc + u * u * y1;
          ctx.beginPath();
          ctx.arc(bx, by, 0.8 + 0.9 * f.power, 0, Math.PI * 2);
          ctx.fillStyle = `hsla(${f.hue},100%,${65}%,${0.35 * f.power})`;
          ctx.fill();
        }
        ctx.restore();

        if (t >= 1) {
          // Start ejection outward along mid-angle
          const ejectionCount = Math.round(24 * f.power);
          for (let p = 0; p < ejectionCount; p++) {
            const ang = f.aMid + rand(-0.15, 0.15);
            const r0 = SUN_R + rand(8, 20);
            const vx = Math.cos(ang) * rand(0.10, 0.35);
            const vy = Math.sin(ang) * rand(0.10, 0.35);
            const x = CX + Math.cos(ang) * r0;
            const y = CY + Math.sin(ang) * r0;
            f.particles.push({
              x, y, vx, vy,
              life: 0, max: rand(900, 1700),
              size: rand(0.6, 1.6),
              hue: f.hue,
              alpha: rand(0.25, 0.45)
            });
          }
          f.phase = 'eject';
          f.start = now;
          f.dur = rand(1200, 2200);
        }
      } else if (f.phase === 'eject') {
        // Update and draw CME particles
        ctx.save();
        ctx.globalCompositeOperation = 'lighter';
        for (let p = f.particles.length - 1; p >= 0; p--) {
          const pt = f.particles[p];
          pt.life += 16.6; // approx per frame at 60fps
          pt.x += pt.vx * (1 + 0.6 * Math.random());
          pt.y += pt.vy * (1 + 0.6 * Math.random());

          const fade = Math.max(0, 1 - pt.life / pt.max);
          if (fade <= 0) { f.particles.splice(p, 1); continue; }

          ctx.beginPath();
          ctx.arc(pt.x, pt.y, pt.size * (0.6 + 0.8 * fade), 0, Math.PI * 2);
          ctx.fillStyle = `hsla(${pt.hue},100%,${60 - 10 * (1 - fade)}%,${pt.alpha * fade})`;
          ctx.shadowColor = `hsla(${pt.hue},100%,60%,${0.25 * fade})`;
          ctx.shadowBlur = 8;
          ctx.fill();
        }
        ctx.restore();

        // Remove when done
        if (f.particles.length === 0) flares.splice(i, 1);
      }
    }
  }

  function draw() {
    const ctx = canvasEl.getContext('2d');
    if (!ctx) return;

    // Reset transform for DPR
    ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
    ctx.clearRect(0, 0, 300, 300);

    const now = performance.now();

    // Fractal backdrop (loops seamlessly)
    renderFractalBackground(now);
    ctx.save();
    ctx.globalAlpha = 0.55;
    ctx.globalCompositeOperation = 'lighter';
    ctx.drawImage(bgCanvas, 0, 0, 300, 300);
    ctx.restore();

    // Background stars (twinkle)
    for (let i = 0; i < stars.length; i++) {
      const s = stars[i];
      const a = 0.25 + 0.55 * (0.5 + 0.5 * Math.sin(now / 1000 * s.tw + s.phase));
      ctx.save();
      ctx.globalAlpha = a;
      ctx.beginPath();
      ctx.arc(s.x, s.y, s.r, 0, Math.PI * 2);
      ctx.fillStyle = '#9cc5ff';
      ctx.shadowColor = '#9cc5ff';
      ctx.shadowBlur = 2;
      ctx.fill();
      ctx.restore();
    }

    // Sun core + corona + texture + spots
    drawSunAndCorona(ctx, now);

    // Coronal loops and CME
    drawFlares(ctx, now);

    // Orbits
    ctx.beginPath();
    ctx.arc(CX, CY, EARTH_ORBIT_R, 0, Math.PI * 2);
    ctx.strokeStyle = "rgba(0,255,231,0.18)";
    ctx.lineWidth = 2.0;
    ctx.stroke();

    // Earth/Moon positions, lighting, shadows (unchanged)
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

    const earthLightAngle = Math.atan2(CY - ey, CX - ex);
    const moonLightAngle = Math.atan2(CY - my, CX - mx);

    drawBodyWithShadow(ctx, ex, ey, EARTH_R, earth, earthSpinAngle, earthLightAngle, '#4aa3ff');
    drawBodyWithShadow(ctx, mx, my, MOON_R, moon, moonSpinAngle, moonLightAngle, '#d0d0d0');

    // Eclipse shadows
    castShadowOnTarget(ctx, CX, CY, mx, my, MOON_R, ex, ey, EARTH_R, 1.0);
    castShadowOnTarget(ctx, CX, CY, ex, ey, EARTH_R, mx, my, MOON_R, 0.85);

    window.requestAnimationFrame(draw);
  }

  // Click -> popup
  canvasEl.addEventListener('click', function () {
    window.showPopup('🌞 You clicked the solar system!<br><br><span style="color:var(--neon);font-size:1.2em;">Keep exploring the universe of transparent data!</span>');
  });

  // Expose popup helpers globally (used by inline onclick)
  window.showPopup = function (msg) {
    const popup = document.getElementById('popup');
    if (!popup) return;
    popup.innerHTML = '<div class="custom-popup">' + (msg || '') + '<br><button onclick="closePopup()">Close</button></div>';
    popup.style.display = 'flex';
  };
  window.closePopup = function () {
    const el = document.getElementById('popup');
    if (el) el.style.display = 'none';
  };
});