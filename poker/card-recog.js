// Card recognizer: finds 16 cards on green felt and reads each card's corner (rank + suit).
(function (root) {
  const N = 16;

  function otsu(hist, tot) {
    let sum = 0; for (let i = 0; i < 256; i++) sum += i * hist[i];
    let sB = 0, wB = 0, best = 0, thr = 128;
    for (let t = 0; t < 256; t++) {
      wB += hist[t]; if (!wB) continue; const wF = tot - wB; if (!wF) break;
      sB += t * hist[t]; const v = wB * wF * (sB / wB - (sum - sB) / wF) ** 2;
      if (v > best) { best = v; thr = t; }
    }
    return thr;
  }

  function blobs(e, W, H, minA, k) {
    const n = W * H, lab = new Int32Array(n), out = [], st = [];
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
      if (a >= minA) out.push({ id, a, cx: sx / a, cy: sy / a, x: Math.max(0, mnx - k), y: Math.max(0, mny - k),
        w: Math.min(W, mxx + k + 1) - Math.max(0, mnx - k), h: Math.min(H, mxy + k + 1) - Math.max(0, mny - k) });
    }
    return { out, lab };
  }

  // 1. find the 16 cards: anything that is not green felt and not near-black
  function locateCards(rgba, W, H) {
    const n = W * H, gray = new Uint8Array(n), m = new Uint8Array(n);
    for (let i = 0, j = 0; i < n; i++, j += 4) {
      const r = rgba[j], g = rgba[j + 1], b = rgba[j + 2];
      gray[i] = (r * 299 + g * 587 + b * 114) / 1000;
      m[i] = (g - Math.max(r, b) <= 10 && gray[i] > 55) ? 1 : 0;
    }
    // fill each card's inside (pictures, ink) by closing small holes row-wise is unnecessary:
    // the card border keeps each card one connected blob.
    let e = m;
    for (let k = 0; k <= 5; k++) {
      if (k) {
        const e2 = new Uint8Array(n);
        for (let y = 1; y < H - 1; y++) for (let x = 1; x < W - 1; x++) {
          const i = y * W + x; e2[i] = e[i] & e[i - 1] & e[i + 1] & e[i - W] & e[i + W];
        }
        e = e2;
      }
      const cand = blobs(e, W, H, n * 0.004, k).out
        .filter(c => c.w / c.h > 0.45 && c.w / c.h < 0.95).sort((p, q) => q.w * q.h - p.w * p.h);
      let pick = null, bestR = 1e9;
      for (let i = 0; i + 16 <= cand.length; i++) {
        const r = (cand[i].w * cand[i].h) / (cand[i + 15].w * cand[i + 15].h);
        if (r < bestR) { bestR = r; pick = cand.slice(i, i + 16); }
      }
      if (pick && bestR < 1.8) {
        pick.sort((p, q) => p.cy - q.cy);
        const cards = [];
        for (let r = 0; r < 4; r++) cards.push(...pick.slice(r * 4, r * 4 + 4).sort((p, q) => p.cx - q.cx));
        return { ok: true, gray, cards };
      }
    }
    return { ok: false, gray };
  }

  // 2. read one corner region: union of ink blobs -> normalised NxN grid
  function glyph(rgba, gray, W, x0, y0, x1, y1) {
    x0 = Math.round(x0); y0 = Math.round(y0); x1 = Math.round(x1); y1 = Math.round(y1);
    const w = x1 - x0, h = y1 - y0; if (w < 4 || h < 4) return null;
    const g = new Uint8Array(w * h), hist = new Array(256).fill(0);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) { const v = gray[(y0 + y) * W + x0 + x]; g[y * w + x] = v; hist[v]++; }
    const thr = otsu(hist, w * h), ink = new Uint8Array(w * h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const i = y * w + x, j = ((y0 + y) * W + x0 + x) * 4;
      const green = rgba[j + 1] - Math.max(rgba[j], rgba[j + 2]) > 8;   // felt / card rim is not ink
      ink[i] = g[i] <= thr && !green ? 1 : 0;
    }
    const { out, lab } = blobs(ink, w, h, w * h * 0.012, 0);
    // drop slivers hugging the region edge (card border, neighbouring glyph)
    const keep = out.filter(c => !((c.x <= 0 || c.y <= 0 || c.x + c.w >= w || c.y + c.h >= h) && (c.a < w * h * 0.04 || c.w < w * 0.12 || c.h < h * 0.12)));
    if (!keep.length) return null;
    const ids = new Set(keep.map(c => c.id));
    let mnx = w, mny = h, mxx = 0, mxy = 0, rs = 0, bs = 0, cnt = 0;
    for (const c of keep) { mnx = Math.min(mnx, c.x); mny = Math.min(mny, c.y); mxx = Math.max(mxx, c.x + c.w - 1); mxy = Math.max(mxy, c.y + c.h - 1); }
    const bw = mxx - mnx + 1, bh = mxy - mny + 1, s = Math.max(bw, bh) / N;
    const ox = mnx + (bw - s * N) / 2, oy = mny + (bh - s * N) / 2;
    const f = new Float32Array(N * N), tot = new Float32Array(N * N);
    for (let y = mny; y <= mxy; y++) for (let x = mnx; x <= mxx; x++) {
      const fx = Math.min(N - 1, Math.max(0, ((x - ox) / s) | 0)), fy = Math.min(N - 1, Math.max(0, ((y - oy) / s) | 0));
      const kk = fy * N + fx; tot[kk]++;
      if (ids.has(lab[y * w + x])) {
        f[kk]++;
        const j = ((y0 + y) * W + x0 + x) * 4; rs += rgba[j]; bs += rgba[j + 2]; cnt++;
      }
    }
    for (let i = 0; i < N * N; i++) f[i] = tot[i] ? f[i] / tot[i] : 0;
    return { f: Array.from(f, v => Math.round(v * 100) / 100), ar: Math.round(bw / bh * 100) / 100, red: cnt ? (rs - bs) / cnt : 0 };
  }

  function cardFeature(rgba, gray, W, c) {
    return {
      rank: glyph(rgba, gray, W, c.x + c.w * 0.03, c.y + c.h * 0.02, c.x + c.w * 0.50, c.y + c.h * 0.27),
      suit: glyph(rgba, gray, W, c.x + c.w * 0.54, c.y + c.h * 0.02, c.x + c.w * 0.99, c.y + c.h * 0.27),
      mid: glyph(rgba, gray, W, c.x + c.w * 0.06, c.y + c.h * 0.36, c.x + c.w * 0.94, c.y + c.h * 0.97)
    };
  }

  function recognizeBoard(rgba, W, H) {
    const L = locateCards(rgba, W, H);
    if (!L.ok) return { ok: false };
    return { ok: true, cards: L.cards, feats: L.cards.map(c => cardFeature(rgba, L.gray, W, c)) };
  }

  function dist(a, b) {
    let d = 0; for (let i = 0; i < a.f.length; i++) { const t = a.f[i] - b.f[i]; d += t * t; }
    return d + 30 * (a.ar - b.ar) ** 2;
  }
  function nearest(feat, samples, allowed) {
    if (!feat) return { v: '?', d: Infinity, conf: 0 };
    const best = {};
    for (const k in samples) {
      if (allowed && !allowed.includes(k)) continue;
      for (const s of samples[k]) { const d = dist(feat, s); if (best[k] === undefined || d < best[k]) best[k] = d; }
    }
    const r = Object.entries(best).sort((a, b) => a[1] - b[1]);
    if (!r.length) return { v: '?', d: Infinity, conf: 0 };
    const d1 = r[0][1], d2 = r[1] ? r[1][1] : d1 * 3 + 30;
    return { v: r[0][0], d: d1, conf: Math.max(0, Math.min(1, (d2 - d1) / (d2 + 1e-6))) };
  }
  function nearestRank(feat, samples) {
    if (!feat.rank) return { v: '?', d: Infinity, conf: 0 };
    const best = {};
    for (const k in samples) for (const s of samples[k]) {
      let d = dist(feat.rank, s.rank);
      if (feat.mid && s.mid) d += 0.5 * dist(feat.mid, s.mid);
      if (best[k] === undefined || d < best[k]) best[k] = d;
    }
    const r = Object.entries(best).sort((a, b) => a[1] - b[1]);
    if (!r.length) return { v: '?', d: Infinity, conf: 0 };
    const d1 = r[0][1], d2 = r[1] ? r[1][1] : d1 * 3 + 30;
    return { v: r[0][0], d: d1, conf: Math.max(0, Math.min(1, (d2 - d1) / (d2 + 1e-6))) };
  }
  // samples: {rank:{A:[{rank,mid},..],..}, suit:{H:[feat,..],..}}
  function classify(feat, samples) {
    const isRed = feat.suit ? feat.suit.red > 35 : (feat.rank ? feat.rank.red > 35 : false);
    const rk = nearestRank(feat, samples.rank);
    const st = nearest(feat.suit, samples.suit, isRed ? ['H', 'D'] : ['C', 'S']);
    return { rank: rk.v, suit: st.v, rk, st, red: isRed };
  }

  const api = { locateCards, recognizeBoard, classify, dist, N };
  if (typeof module !== 'undefined') module.exports = api; else root.CardRecog = api;
})(typeof globalThis !== 'undefined' ? globalThis : this);
