// Letter recognizer: works on a 400x400 grayscale board image (Uint8 array).
// Splits into 4x4 cells, finds the letter blob in each, normalises it to a
// 20x20 feature grid, and matches against stored samples (nearest neighbour).
(function (root) {
  const N = 20;

  function cellFeature(px, W, cx0, cy0, cw, ch) {
    // search region: central part of the tile, above the dots
    const x0 = Math.round(cx0 + cw * 0.12), x1 = Math.round(cx0 + cw * 0.88);
    const y0 = Math.round(cy0 + ch * 0.08), y1 = Math.round(cy0 + ch * 0.74);
    const w = x1 - x0, h = y1 - y0, g = new Uint8Array(w * h);
    const hist = new Array(256).fill(0);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const v = px[(y0 + y) * W + x0 + x]; g[y * w + x] = v; hist[v]++;
    }
    // Otsu threshold
    const tot = w * h; let sum = 0; for (let i = 0; i < 256; i++) sum += i * hist[i];
    let sB = 0, wB = 0, best = 0, thr = 128;
    for (let t = 0; t < 256; t++) {
      wB += hist[t]; if (!wB) continue; const wF = tot - wB; if (!wF) break;
      sB += t * hist[t]; const mB = sB / wB, mF = (sum - sB) / wF, v = wB * wF * (mB - mF) ** 2;
      if (v > best) { best = v; thr = t; }
    }
    // connected components of dark pixels
    const lab = new Int32Array(w * h), comps = [], st = [];
    let id = 0;
    for (let i = 0; i < w * h; i++) {
      if (g[i] > thr || lab[i]) continue;
      id++; let n = 0, mnx = 1e9, mny = 1e9, mxx = -1, mxy = -1, edge = false;
      st.push(i); lab[i] = id;
      while (st.length) {
        const p = st.pop(), x = p % w, y = (p / w) | 0; n++;
        if (x < mnx) mnx = x; if (x > mxx) mxx = x; if (y < mny) mny = y; if (y > mxy) mxy = y;
        if (x === 0 || y === 0 || x === w - 1 || y === h - 1) edge = true;
        const nb = [p - 1, p + 1, p - w, p + w];
        for (let k = 0; k < 4; k++) {
          const q = nb[k];
          if (k === 0 && x === 0 || k === 1 && x === w - 1 || k === 2 && y === 0 || k === 3 && y === h - 1) continue;
          if (!lab[q] && g[q] <= thr) { lab[q] = id; st.push(q); }
        }
      }
      comps.push({ id, n, mnx, mny, mxx, mxy, edge });
    }
    const inner = comps.filter(c => !c.edge && (c.mxy - c.mny) > h * 0.2);
    const pool = inner.length ? inner : comps;
    if (!pool.length) return null;
    const c = pool.reduce((a, b) => (b.n > a.n ? b : a));
    // dark contrast check: a real letter is much darker than the tile
    const bw = c.mxx - c.mnx + 1, bh = c.mxy - c.mny + 1, s = Math.max(bw, bh) / N;
    const ox = c.mnx + (bw - s * N) / 2, oy = c.mny + (bh - s * N) / 2;
    const f = new Float32Array(N * N), cnt = new Float32Array(N * N);
    for (let y = c.mny; y <= c.mxy; y++) for (let x = c.mnx; x <= c.mxx; x++) {
      const fx = Math.min(N - 1, Math.max(0, ((x - ox) / s) | 0)), fy = Math.min(N - 1, Math.max(0, ((y - oy) / s) | 0));
      const k = fy * N + fx; cnt[k]++; if (lab[y * w + x] === c.id) f[k]++;
    }
    for (let k = 0; k < N * N; k++) f[k] = cnt[k] ? f[k] / cnt[k] : 0;
    return { f: Array.from(f, v => Math.round(v * 100) / 100), ar: Math.round(bw / bh * 100) / 100 };
  }

  function boardFeatures(px, W) {
    const cw = W / 4, out = [];
    for (let r = 0; r < 4; r++) for (let c = 0; c < 4; c++) out.push(cellFeature(px, W, c * cw, r * cw, cw, cw));
    return out;
  }

  function findGrid(e, W, H, k) {
    const n = W * H, lab = new Int32Array(n), comps = [], st = [];
    let id = 0;
    for (let i = 0; i < n; i++) {
      if (!e[i] || lab[i]) continue;
      id++; let a = 0, sx = 0, sy = 0, mnx = W, mny = H, mxx = 0, mxy = 0;
      st.push(i); lab[i] = id;
      while (st.length) {
        const p = st.pop(), x = p % W, y = (p / W) | 0; a++; sx += x; sy += y;
        if (x < mnx) mnx = x; if (x > mxx) mxx = x; if (y < mny) mny = y; if (y > mxy) mxy = y;
        if (x > 0 && e[p - 1] && !lab[p - 1]) { lab[p - 1] = id; st.push(p - 1); }
        if (x < W - 1 && e[p + 1] && !lab[p + 1]) { lab[p + 1] = id; st.push(p + 1); }
        if (y > 0 && e[p - W] && !lab[p - W]) { lab[p - W] = id; st.push(p - W); }
        if (y < H - 1 && e[p + W] && !lab[p + W]) { lab[p + W] = id; st.push(p + W); }
      }
      if (a > n * 0.003) comps.push({ a, cx: sx / a, cy: sy / a, x: Math.max(0, mnx - k), y: Math.max(0, mny - k), w: Math.min(W, mxx + k + 1) - Math.max(0, mnx - k), h: Math.min(H, mxy + k + 1) - Math.max(0, mny - k) });
    }
    const cand = comps.filter(c => c.w / c.h > 0.6 && c.w / c.h < 1.6).sort((p, q) => q.a - p.a);
    let pick = null, bestR = 1e9;
    for (let i = 0; i + 16 <= cand.length; i++) {
      const r = cand[i].a / cand[i + 15].a;
      if (r < bestR) { bestR = r; pick = cand.slice(i, i + 16); }
    }
    return pick && bestR < 2.2 ? pick : null;
  }

  // Find the 16 tiles in an RGBA capture: tiles are light, neutral blobs on a blue board.
  function locateTiles(rgba, W, H) {
    const n = W * H, gray = new Uint8Array(n), m = new Uint8Array(n);
    // tiles are bright and neutral; the board and backgrounds are dark and/or blue.
    // score = brightness - 1.5 x blueness, split with Otsu
    const d = new Uint8Array(n), hist = new Array(256).fill(0);
    for (let i = 0, j = 0; i < n; i++, j += 4) {
      const r = rgba[j], g = rgba[j + 1], b = rgba[j + 2];
      gray[i] = (r * 299 + g * 587 + b * 114) / 1000;
      d[i] = Math.max(0, Math.min(255, gray[i] - 1.5 * (b - r)));
      hist[d[i]]++;
    }
    const cnt = n;
    let sum = 0; for (let i = 0; i < 256; i++) sum += i * hist[i];
    let sB = 0, wB = 0, best = 0, thr = 128;
    for (let t = 0; t < 256; t++) {
      wB += hist[t]; if (!wB) continue; const wF = cnt - wB; if (!wF) break;
      sB += t * hist[t]; const v = wB * wF * (sB / wB - (sum - sB) / wF) ** 2;
      if (v > best) { best = v; thr = t; }
    }
    for (let i = 0; i < n; i++) m[i] = d[i] > thr ? 1 : 0;
    // erode step by step until the tiles separate into 16 similar blobs
    let e = m, pick = null;
    for (let k = 1; k <= 6 && !pick; k++) {
      const e2 = new Uint8Array(n);
      for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
        const i = y * W + x;
        e2[i] = e[i] & e[i - 1] & e[i + 1] & e[i - W] & e[i + W];
      }
      e = e2;
      pick = findGrid(e, W, H, k);
    }
    if (!pick) return { ok: false, gray };
    pick.sort((p, q) => p.cy - q.cy);
    const tiles = [];
    for (let r = 0; r < 4; r++) tiles.push(...pick.slice(r * 4, r * 4 + 4).sort((p, q) => p.cx - q.cx));
    return { ok: true, gray, tiles };
  }

  function recognizeBoard(rgba, W, H) {
    const L = locateTiles(rgba, W, H);
    if (!L.ok) return { ok: false };
    return { ok: true, tiles: L.tiles, feats: L.tiles.map(t => cellFeature(L.gray, W, t.x, t.y, t.w, t.h)) };
  }

  function dist(a, b) {
    let d = 0; for (let i = 0; i < a.f.length; i++) { const t = a.f[i] - b.f[i]; d += t * t; }
    return d + 40 * (a.ar - b.ar) ** 2;
  }

  // samples: {A:[feat,...], ...}. Returns {ch, conf, d}
  function classify(feat, samples) {
    if (!feat) return { ch: '?', conf: 0, d: Infinity };
    const best = {};
    for (const ch in samples) for (const s of samples[ch]) {
      const d = dist(feat, s); if (best[ch] === undefined || d < best[ch]) best[ch] = d;
    }
    const ranked = Object.entries(best).sort((a, b) => a[1] - b[1]);
    if (!ranked.length) return { ch: '?', conf: 0, d: Infinity };
    const [c1, d1] = ranked[0], d2 = ranked[1] ? ranked[1][1] : d1 * 3 + 50;
    // confidence: how clearly the best beats the runner-up, and how close it is at all
    const conf = Math.max(0, Math.min(1, (d2 - d1) / (d2 + 1e-6))) * (d1 < 60 ? 1 : 0.4);
    return { ch: c1, conf, d: d1 };
  }

  const api = { cellFeature, boardFeatures, locateTiles, recognizeBoard, classify, dist, N };
  if (typeof module !== 'undefined') module.exports = api; else root.Recog = api;
})(typeof globalThis !== "undefined" ? globalThis : this);
