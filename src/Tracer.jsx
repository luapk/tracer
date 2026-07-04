import { useState, useRef, useEffect, useCallback } from "react";

/* ═══════════════════════════════════════════════════
   TRACER v1.3 — Motion Tracking Overlay Engine
   Skeletrix + N4ture + Surveil + Mocap render modes
   Patch-based optical-flow tracker, mobile-optimised
   ═══════════════════════════════════════════════════ */

const COLORS = {
  cyan:    { primary: "#00ffff", secondary: "#007777", glow: "rgba(0,255,255,0.15)", text: "#00cccc" },
  green:   { primary: "#39ff14", secondary: "#1a7a0a", glow: "rgba(57,255,20,0.15)", text: "#2bcc10" },
  amber:   { primary: "#ffbf00", secondary: "#7a5c00", glow: "rgba(255,191,0,0.15)", text: "#cc9900" },
  magenta: { primary: "#ff44ff", secondary: "#7a007a", glow: "rgba(255,68,255,0.15)", text: "#cc33cc" },
  white:   { primary: "#ffffff", secondary: "#666666", glow: "rgba(255,255,255,0.1)", text: "#999999" },
};

const DEFAULTS = {
  renderMode: "skeletrix",
  colorScheme: "cyan",
  lineWeight: 1.0,
  background: "dimmed",
  motionThreshold: 18,
  sampleRate: 4,
  dimAmount: 0.75,
  // Skeletrix
  skxPoints: 22,
  skxSwatchSize: 18,
  skxMarkerSize: 6,
  skxExtendLines: true,
  skxLabels: true,
  // N4ture
  n4tPoints: 30,
  n4tThreshold: 12,
  n4tFlowLines: true,
  n4tSwatches: true,
  n4tLabels: true,
  n4tOrganicRadius: 8,
  n4tLineWeight: 1.0,
  // Surveil
  survPoints: 18,
  survThreshold: 22,
  survLineWeight: 1.2,
  survBrackets: true,
  survCrosshair: true,
  survHud: true,
  survScanLine: true,
  survPixelate: true,
  survBracketSize: 44,
  // Mocap
  mocPoints: 42,
  mocThreshold: 16,
  mocLineWeight: 1.0,
  mocVectors: true,
  mocWireframe: true,
  mocGrid: true,
  mocData: true,
  mocRgb: true,
  mocVectorScale: 4,
};

/* ─── Delaunay Triangulation (Bowyer-Watson) ─── */

function delaunay(points) {
  if (points.length < 3) return [];
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of points) {
    if (p.x < minX) minX = p.x; if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x; if (p.y > maxY) maxY = p.y;
  }
  const dx = maxX - minX, dy = maxY - minY, dmax = Math.max(dx, dy);
  const midX = (minX + maxX) / 2, midY = (minY + maxY) / 2;
  const p0 = { x: midX - 20 * dmax, y: midY - dmax, idx: -1 };
  const p1 = { x: midX, y: midY + 20 * dmax, idx: -2 };
  const p2 = { x: midX + 20 * dmax, y: midY - dmax, idx: -3 };
  let tris = [{ a: p0, b: p1, c: p2 }];
  for (const p of points) {
    const good = [], bad = [];
    for (const t of tris) { if (inCircum(p, t)) bad.push(t); else good.push(t); }
    const edges = [];
    for (const t of bad) { edges.push([t.a, t.b]); edges.push([t.b, t.c]); edges.push([t.c, t.a]); }
    const boundary = [];
    for (let i = 0; i < edges.length; i++) {
      let shared = false;
      for (let j = 0; j < edges.length; j++) {
        if (i !== j && edges[i][0] === edges[j][1] && edges[i][1] === edges[j][0]) { shared = true; break; }
      }
      if (!shared) boundary.push(edges[i]);
    }
    for (const e of boundary) good.push({ a: e[0], b: e[1], c: p });
    tris = good;
  }
  return tris.filter(t => t.a.idx >= 0 && t.b.idx >= 0 && t.c.idx >= 0);
}

function inCircum(p, t) {
  const ax = t.a.x - p.x, ay = t.a.y - p.y;
  const bx = t.b.x - p.x, by = t.b.y - p.y;
  const cx = t.c.x - p.x, cy = t.c.y - p.y;
  return (ax*ax+ay*ay)*(bx*cy-cx*by) - (bx*bx+by*by)*(ax*cy-cx*ay) + (cx*cx+cy*cy)*(ax*by-bx*ay) > 0;
}

/* ─── Detection ─── */

function getGrayscale(data, w, h) {
  const g = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) { const j = i*4; g[i] = (data[j]*0.299+data[j+1]*0.587+data[j+2]*0.114)|0; }
  return g;
}

/* ─── Optical-flow point tracker ───
   Real frame-to-frame motion lock: each tracked point block-matches its
   7x7 grayscale patch from the previous frame against a search window
   around its velocity-predicted position (SAD minimisation). Points coast
   on their velocity through brief occlusion and are culled after repeated
   misses. New points seed on moving, textured spots with min spacing.
   Runs on a downscaled processing frame so it is cheap on mobile. */

const TRK = { PATCH: 3, SEARCH: 6, MAX_MISS: 6, SAD_LOST: 26 };

function trackerUpdate(state, prev, curr, w, h, opts) {
  const P = TRK.PATCH, S = TRK.SEARCH, margin = P + S + 1;
  const side = P * 2 + 1, area = side * side;

  if (prev) {
    for (const tp of state.points) {
      const px = Math.round(tp.x), py = Math.round(tp.y);
      if (px < P || py < P || px >= w - P || py >= h - P) { tp.missed = TRK.MAX_MISS; continue; }
      // Search around the velocity-predicted position
      let cx = Math.round(tp.x + tp.vx), cy = Math.round(tp.y + tp.vy);
      cx = Math.max(margin, Math.min(w - margin, cx));
      cy = Math.max(margin, Math.min(h - margin, cy));
      let best = Infinity, bx = cx, by = cy;
      for (let dy = -S; dy <= S; dy++) {
        for (let dx = -S; dx <= S; dx++) {
          const nx = cx + dx, ny = cy + dy;
          let sad = 0;
          for (let oy = -P; oy <= P; oy++) {
            const rp = (py + oy) * w + px, rc = (ny + oy) * w + nx;
            for (let ox = -P; ox <= P; ox++) sad += Math.abs(prev[rp + ox] - curr[rc + ox]);
          }
          if (sad < best) { best = sad; bx = nx; by = ny; }
        }
      }
      if (best / area < TRK.SAD_LOST) {
        tp.vx = tp.vx * 0.4 + (bx - tp.x) * 0.6;
        tp.vy = tp.vy * 0.4 + (by - tp.y) * 0.6;
        tp.x = bx; tp.y = by;
        tp.age++; tp.missed = 0;
      } else {
        tp.missed++;
        tp.x += tp.vx; tp.y += tp.vy;
      }
      if (tp.x < margin || tp.y < margin || tp.x > w - margin || tp.y > h - margin) tp.missed = TRK.MAX_MISS;
    }
    state.points = state.points.filter(p => p.missed < TRK.MAX_MISS);
  }

  // Seed new points: moving (or, on the first frame, textured) spots, spaced out
  if (state.points.length < opts.maxPoints) {
    const cand = [];
    const step = Math.max(2, opts.step);
    for (let y = margin; y < h - margin; y += step) {
      for (let x = margin; x < w - margin; x += step) {
        const i = y * w + x;
        const motion = prev ? Math.abs(curr[i] - prev[i]) : 0;
        if (prev && motion < opts.threshold) continue;
        const gx = curr[i + 1] - curr[i - 1], gy = curr[i + w] - curr[i - w];
        const tex = Math.abs(gx) + Math.abs(gy);
        if (tex < 10) continue; // untextured patches can't be tracked
        let score = (motion + 8) * (tex + 4);
        if (opts.lumBias) { const l = curr[i] / 255; score *= 0.25 + 0.75 * l * l; }
        cand.push({ x, y, score, intensity: Math.min(1, motion / 60 + 0.3) });
      }
    }
    cand.sort((a, b) => b.score - a.score);
    const minDist = Math.max(10, Math.min(w, h) * 0.055);
    const md2 = minDist * minDist;
    for (const c of cand) {
      if (state.points.length >= opts.maxPoints) break;
      let clash = false;
      for (const tp of state.points) {
        const ddx = tp.x - c.x, ddy = tp.y - c.y;
        if (ddx * ddx + ddy * ddy < md2) { clash = true; break; }
      }
      if (!clash) state.points.push({ x: c.x, y: c.y, vx: 0, vy: 0, age: 0, missed: 0, intensity: c.intensity, id: state.nextId++ });
    }
  }

  // Live intensity for renderers: local motion + speed
  if (prev) {
    for (const tp of state.points) {
      const i = Math.round(tp.y) * w + Math.round(tp.x);
      if (i >= 0 && i < curr.length)
        tp.intensity = Math.min(1, Math.abs(curr[i] - prev[i]) / 50 + Math.hypot(tp.vx, tp.vy) / 6 + 0.25);
    }
  }
}

function thinPoints(points, maxCount) {
  if (points.length <= maxCount) return points;
  const sorted = [...points].sort((a, b) => b.intensity - a.intensity);
  const step = Math.ceil(sorted.length / maxCount);
  const res = [];
  for (let i = 0; i < sorted.length && res.length < maxCount; i += step) res.push(sorted[i]);
  return res;
}

/* ─── Sample pixel color from the (downscaled) processing frame ───
   frame = { data, w, h, scale } where scale maps display coords → frame coords */

function samplePixel(frame, x, y) {
  if (!frame) return null;
  const px = Math.round(x * frame.scale), py = Math.round(y * frame.scale);
  if (px < 0 || py < 0 || px >= frame.w) return null;
  const i = (py * frame.w + px) * 4;
  if (i < 0 || i + 2 >= frame.data.length) return null;
  return [frame.data[i], frame.data[i + 1], frame.data[i + 2]];
}

function sampleColor(frame, x, y) {
  const p = samplePixel(frame, x, y);
  return p ? `rgb(${p[0]},${p[1]},${p[2]})` : "rgba(180,180,180,0.6)";
}

/* ─── Extend a line from p through q to the canvas edge ─── */

function extendToEdge(px, py, qx, qy, w, h) {
  const dx = qx - px, dy = qy - py;
  if (dx === 0 && dy === 0) return { x: qx, y: qy };
  let tMax = 10000;
  if (dx > 0) tMax = Math.min(tMax, (w - px) / dx);
  else if (dx < 0) tMax = Math.min(tMax, -px / dx);
  if (dy > 0) tMax = Math.min(tMax, (h - py) / dy);
  else if (dy < 0) tMax = Math.min(tMax, -py / dy);
  return { x: px + dx * tMax, y: py + dy * tMax };
}

/* ─── Skeletrix Renderer ─── */

function renderSkeletrix(ctx, points, triangles, settings, colors, width, height, frame) {
  ctx.clearRect(0, 0, width, height);
  if (points.length === 0) return;

  const lw = settings.lineWeight;

  // 1. Extended radial lines from select vertices — dramatic projections to edges
  if (settings.skxExtendLines) {
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    ctx.lineWidth = lw * 0.5;
    ctx.shadowBlur = 0;
    ctx.globalAlpha = 1;
    ctx.beginPath();
    // Pick ~30% of points for extension lines
    const extStep = Math.max(1, Math.floor(points.length / Math.max(6, points.length * 0.3)));
    for (let i = 0; i < points.length; i += extStep) {
      const p = points[i];
      // Find a nearby point to define direction
      let nearest = null, nd = Infinity;
      for (let j = 0; j < points.length; j++) {
        if (j === i) continue;
        const d = (points[j].x-p.x)**2 + (points[j].y-p.y)**2;
        if (d < nd) { nd = d; nearest = points[j]; }
      }
      if (nearest) {
        const e = extendToEdge(p.x, p.y, nearest.x, nearest.y, width, height);
        ctx.moveTo(p.x, p.y); ctx.lineTo(e.x, e.y);
        // Also extend in opposite direction
        const e2 = extendToEdge(p.x, p.y, p.x-(nearest.x-p.x), p.y-(nearest.y-p.y), width, height);
        ctx.moveTo(p.x, p.y); ctx.lineTo(e2.x, e2.y);
      }
    }
    ctx.stroke();
  }

  // 2. Triangle edges — thin, sharp, white, large confident shapes
  if (triangles.length > 0) {
    ctx.strokeStyle = "rgba(255,255,255,0.55)";
    ctx.lineWidth = lw * 0.8;
    ctx.shadowColor = "rgba(255,255,255,0.3)";
    ctx.shadowBlur = 2;
    ctx.globalAlpha = 1;
    ctx.beginPath();
    for (const tri of triangles) {
      ctx.moveTo(tri.a.x, tri.a.y);
      ctx.lineTo(tri.b.x, tri.b.y);
      ctx.lineTo(tri.c.x, tri.c.y);
      ctx.closePath();
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  // 3. Colour-sampled swatches at vertices
  if (frame) {
    const sw = settings.skxSwatchSize;
    for (const p of points) {
      const col = sampleColor(frame, p.x, p.y);
      ctx.globalAlpha = 0.65;
      ctx.fillStyle = col;
      // Offset the swatch slightly so it doesn't cover the vertex
      const ox = (p.x > width/2) ? -sw - 4 : 4;
      const oy = (p.y > height/2) ? -sw - 4 : 4;
      ctx.fillRect(p.x + ox, p.y + oy, sw, sw);
      // Thin border
      ctx.strokeStyle = "rgba(255,255,255,0.3)";
      ctx.lineWidth = 0.5;
      ctx.globalAlpha = 0.5;
      ctx.strokeRect(p.x + ox, p.y + oy, sw, sw);
    }
    ctx.globalAlpha = 1;
  }

  // 4. Rectangular tracking markers at each vertex
  const ms = settings.skxMarkerSize;
  ctx.strokeStyle = "rgba(255,255,255,0.7)";
  ctx.lineWidth = lw * 0.6;
  ctx.globalAlpha = 1;
  for (const p of points) {
    ctx.strokeRect(p.x - ms/2, p.y - ms/2, ms, ms);
    // Small crosshair inside
    const ch = ms * 0.3;
    ctx.beginPath();
    ctx.moveTo(p.x - ch, p.y); ctx.lineTo(p.x + ch, p.y);
    ctx.moveTo(p.x, p.y - ch); ctx.lineTo(p.x, p.y + ch);
    ctx.stroke();
  }

  // 5. Connecting lines between nearby rect markers — secondary detail
  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.lineWidth = lw * 0.4;
  ctx.beginPath();
  for (let i = 0; i < points.length; i++) {
    for (let j = i+1; j < points.length; j++) {
      const dx = points[i].x - points[j].x, dy = points[i].y - points[j].y;
      const d = Math.sqrt(dx*dx + dy*dy);
      if (d < width * 0.35 && d > width * 0.08) {
        ctx.moveTo(points[i].x, points[i].y);
        ctx.lineTo(points[j].x, points[j].y);
      }
    }
  }
  ctx.stroke();

  // 6. Text labels at anchor points
  if (settings.skxLabels) {
    ctx.shadowBlur = 0;
    const fs = Math.max(8, Math.round(width / 160));
    ctx.font = `${fs}px 'Courier New', monospace`;
    ctx.globalAlpha = 0.45;
    ctx.fillStyle = "#ffffff";
    const names = ["vertex", "anchor", "node", "track", "point", "marker", "joint", "edge", "ref", "scan"];
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      const label = `${names[i % names.length]}_${String(i).padStart(2,"0")}`;
      const lx = p.x + ms + 3;
      const ly = p.y + fs * 0.35;
      ctx.fillText(label, lx, ly);
    }
    ctx.globalAlpha = 1;
  }
}

/* ─── N4ture Renderer ─── */

function renderN4ture(ctx, points, triangles, settings, width, height, frame) {
  ctx.clearRect(0, 0, width, height);
  if (points.length === 0) return;

  const lw = settings.n4tLineWeight;
  const N4C = "#ffffff";
  const N4S = "rgba(255,255,255,";
  const drawDiamond = (cx, cy, r) => {
    ctx.beginPath();
    ctx.moveTo(cx, cy - r);
    ctx.lineTo(cx + r, cy);
    ctx.lineTo(cx, cy + r);
    ctx.lineTo(cx - r, cy);
    ctx.closePath();
  };

  // 1. Flowing arcs extending to canvas edges — organic version of skx extend lines
  if (settings.n4tFlowLines) {
    const extStep = Math.max(1, Math.floor(points.length / Math.max(6, points.length * 0.4)));
    for (let i = 0; i < points.length; i += extStep) {
      const p = points[i];
      let nearest = null, nd = Infinity;
      for (let j = 0; j < points.length; j++) {
        if (j === i) continue;
        const d = (points[j].x - p.x) ** 2 + (points[j].y - p.y) ** 2;
        if (d < nd) { nd = d; nearest = points[j]; }
      }
      if (nearest) {
        const e = extendToEdge(p.x, p.y, nearest.x, nearest.y, width, height);
        const ddx = e.x - p.x, ddy = e.y - p.y;
        const len = Math.sqrt(ddx * ddx + ddy * ddy) || 1;
        const nx = -ddy / len, ny = ddx / len;
        const sway = Math.sin(i * 2.1) * width * 0.04;
        ctx.strokeStyle = N4S + "0.1)";
        ctx.lineWidth = lw * 0.5;
        ctx.shadowBlur = 0;
        ctx.globalAlpha = 1;
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.quadraticCurveTo(p.x + ddx * 0.5 + nx * sway, p.y + ddy * 0.5 + ny * sway, e.x, e.y);
        ctx.stroke();
        // Reverse arc
        const e2 = extendToEdge(p.x, p.y, p.x - (nearest.x - p.x), p.y - (nearest.y - p.y), width, height);
        const d2x = e2.x - p.x, d2y = e2.y - p.y;
        const len2 = Math.sqrt(d2x * d2x + d2y * d2y) || 1;
        const n2x = -d2y / len2, n2y = d2x / len2;
        ctx.beginPath();
        ctx.moveTo(p.x, p.y);
        ctx.quadraticCurveTo(p.x + d2x * 0.5 + n2x * (-sway), p.y + d2y * 0.5 + n2y * (-sway), e2.x, e2.y);
        ctx.stroke();
      }
    }
  }

  // 2. Triangle edges as gentle organic arcs
  if (triangles.length > 0) {
    ctx.strokeStyle = N4S + "0.7)";
    ctx.lineWidth = lw * 1.0;
    ctx.shadowColor = N4C;
    ctx.shadowBlur = 4;
    ctx.globalAlpha = 1;
    const drawArc = (ax, ay, bx, by) => {
      const mx = (ax + bx) / 2, my = (ay + by) / 2;
      const ddx = bx - ax, ddy = by - ay;
      const len = Math.sqrt(ddx * ddx + ddy * ddy) || 1;
      const bulge = len * 0.08;
      ctx.beginPath();
      ctx.moveTo(ax, ay);
      ctx.quadraticCurveTo(mx + (-ddy / len) * bulge, my + (ddx / len) * bulge, bx, by);
      ctx.stroke();
    };
    for (const tri of triangles) {
      drawArc(tri.a.x, tri.a.y, tri.b.x, tri.b.y);
      drawArc(tri.b.x, tri.b.y, tri.c.x, tri.c.y);
      drawArc(tri.c.x, tri.c.y, tri.a.x, tri.a.y);
    }
    ctx.shadowBlur = 0;
  }

  // 3. Diamond colour swatches — sampled pixel colour at each tracked point, offset outward
  if (frame && settings.n4tSwatches) {
    const r = settings.n4tOrganicRadius;
    for (const p of points) {
      const col = sampleColor(frame, p.x, p.y);
      const angle = Math.atan2(p.y - height / 2, p.x - width / 2);
      const ox = Math.cos(angle + Math.PI) * (r * 2.0);
      const oy = Math.sin(angle + Math.PI) * (r * 2.0);
      const cx = p.x + ox, cy = p.y + oy;
      ctx.globalAlpha = 0.85;
      ctx.fillStyle = col;
      drawDiamond(cx, cy, r);
      ctx.fill();
      ctx.strokeStyle = N4S + "0.85)";
      ctx.lineWidth = lw * 0.6;
      ctx.globalAlpha = 1;
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  // 4. Diamond node markers with radial spokes
  const ms = settings.n4tOrganicRadius;
  ctx.lineWidth = lw * 0.55;
  ctx.shadowColor = N4C;
  ctx.shadowBlur = 4;
  const hs = ms * 0.65;
  for (const p of points) {
    ctx.globalAlpha = 0.45 + p.intensity * 0.45;
    ctx.strokeStyle = N4C;
    drawDiamond(p.x, p.y, hs);
    ctx.stroke();
    ctx.beginPath();
    for (let s = 0; s < 4; s++) {
      const a = (s / 4) * Math.PI * 2 + Math.PI * 0.25;
      ctx.moveTo(p.x + Math.cos(a) * ms * 0.25, p.y + Math.sin(a) * ms * 0.25);
      ctx.lineTo(p.x + Math.cos(a) * ms * 0.9, p.y + Math.sin(a) * ms * 0.9);
    }
    ctx.stroke();
  }
  ctx.shadowBlur = 0;

  // 5. Connecting arcs between nearby points — like vines or currents
  ctx.strokeStyle = N4S + "0.18)";
  ctx.lineWidth = lw * 0.55;
  ctx.shadowColor = N4C;
  ctx.shadowBlur = 5;
  for (let i = 0; i < points.length; i++) {
    for (let j = i + 1; j < points.length; j++) {
      const dx = points[i].x - points[j].x, dy = points[i].y - points[j].y;
      const d = Math.sqrt(dx * dx + dy * dy);
      if (d < width * 0.32 && d > width * 0.07) {
        const mx = (points[i].x + points[j].x) / 2, my = (points[i].y + points[j].y) / 2;
        const bulge = d * 0.12 * (Math.sin((i + j) * 1.3) > 0 ? 1 : -1);
        ctx.globalAlpha = 0.3 + 0.4 * (1 - d / (width * 0.32));
        ctx.beginPath();
        ctx.moveTo(points[i].x, points[i].y);
        ctx.quadraticCurveTo(mx + (-dy / d) * bulge, my + (dx / d) * bulge, points[j].x, points[j].y);
        ctx.stroke();
      }
    }
  }
  ctx.shadowBlur = 0;
  ctx.globalAlpha = 1;

  // 6. Nature labels — diamond colour swatch + text id
  if (settings.n4tLabels) {
    ctx.shadowBlur = 0;
    const fs = Math.max(8, Math.round(width / 160));
    ctx.font = `${fs}px 'Courier New', monospace`;
    const names = ["stem", "blade", "frond", "ripple", "crest", "sway", "drift", "flow", "wave", "curl", "reed", "spore"];
    const dr = fs * 0.55;
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      const lx = p.x + ms + 6;
      const ly = p.y;
      if (frame) {
        const col = sampleColor(frame, p.x, p.y);
        ctx.globalAlpha = 0.9;
        ctx.fillStyle = col;
        drawDiamond(lx + dr, ly, dr);
        ctx.fill();
      }
      ctx.globalAlpha = 0.85;
      ctx.fillStyle = N4C;
      ctx.fillText(`${names[i % names.length]}_${String(i).padStart(2, "0")}`, lx + dr * 2 + 4, ly + fs * 0.35);
    }
    ctx.globalAlpha = 1;
  }
}

/* ─── Surveil Renderer ─── */

function renderSurveil(ctx, points, settings, width, height, frame, elapsed) {
  ctx.clearRect(0, 0, width, height);
  const lw = settings.survLineWeight;
  const COL = "#ff3838";
  const CS = "rgba(255,56,56,";

  if (settings.survScanLine) {
    const sy = (elapsed * 80) % (height + 120) - 60;
    const grd = ctx.createLinearGradient(0, sy - 60, 0, sy + 60);
    grd.addColorStop(0, CS + "0)");
    grd.addColorStop(0.5, CS + "0.14)");
    grd.addColorStop(1, CS + "0)");
    ctx.fillStyle = grd;
    ctx.fillRect(0, sy - 60, width, 120);
    ctx.strokeStyle = CS + "0.45)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(0, sy); ctx.lineTo(width, sy);
    ctx.stroke();
  }

  const fs = Math.max(9, Math.round(width / 180));
  const monoFont = `${fs}px 'Courier New', monospace`;
  const bigFont = `${Math.round(fs*1.15)}px 'Courier New', monospace`;

  if (settings.survHud) {
    ctx.font = bigFont;
    ctx.fillStyle = COL;
    ctx.globalAlpha = 0.95;
    const secs = elapsed;
    const hh = String(Math.floor(secs/3600)%24).padStart(2,"0");
    const mm = String(Math.floor(secs/60)%60).padStart(2,"0");
    const ss = String(Math.floor(secs)%60).padStart(2,"0");
    const cs = String(Math.floor((secs%1)*100)).padStart(2,"0");
    // Blinking REC dot
    if (Math.floor(elapsed*2) % 2 === 0) {
      ctx.beginPath(); ctx.arc(14, fs + 2, fs*0.35, 0, Math.PI*2); ctx.fill();
    }
    ctx.fillText(`REC  ${hh}:${mm}:${ss}.${cs}`, 26, fs + 6);
    ctx.fillText(`TRACKING ${String(points.length).padStart(2,"0")} TARGETS`, 26, fs*2 + 12);
    const right = `CH_01 // ISR-CAM  ${width}x${height}`;
    ctx.textAlign = "right";
    ctx.fillText(right, width - 14, fs + 6);
    ctx.fillText(`LAT 00.000  LON 00.000`, width - 14, fs*2 + 12);
    ctx.textAlign = "left";
    // Frame corner brackets
    const fb = 22, fw = lw * 1.5;
    ctx.lineWidth = fw;
    ctx.strokeStyle = COL;
    ctx.beginPath();
    ctx.moveTo(6, 6+fb); ctx.lineTo(6,6); ctx.lineTo(6+fb,6);
    ctx.moveTo(width-6-fb,6); ctx.lineTo(width-6,6); ctx.lineTo(width-6,6+fb);
    ctx.moveTo(width-6,height-6-fb); ctx.lineTo(width-6,height-6); ctx.lineTo(width-6-fb,height-6);
    ctx.moveTo(6+fb,height-6); ctx.lineTo(6,height-6); ctx.lineTo(6,height-6-fb);
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  if (points.length === 0) return;

  const bs = settings.survBracketSize;
  const cornerLen = bs * 0.32;

  if (settings.survBrackets) {
    ctx.strokeStyle = COL;
    ctx.lineWidth = lw * 1.3;
    ctx.shadowColor = COL;
    ctx.shadowBlur = 6;
    ctx.beginPath();
    for (const p of points) {
      const x0 = p.x - bs/2, y0 = p.y - bs/2;
      const x1 = p.x + bs/2, y1 = p.y + bs/2;
      ctx.moveTo(x0, y0+cornerLen); ctx.lineTo(x0,y0); ctx.lineTo(x0+cornerLen,y0);
      ctx.moveTo(x1-cornerLen,y0); ctx.lineTo(x1,y0); ctx.lineTo(x1,y0+cornerLen);
      ctx.moveTo(x1,y1-cornerLen); ctx.lineTo(x1,y1); ctx.lineTo(x1-cornerLen,y1);
      ctx.moveTo(x0+cornerLen,y1); ctx.lineTo(x0,y1); ctx.lineTo(x0,y1-cornerLen);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  if (settings.survCrosshair) {
    ctx.strokeStyle = CS + "0.85)";
    ctx.lineWidth = lw * 0.7;
    const rs = 12, gap = 3;
    ctx.beginPath();
    for (const p of points) {
      ctx.moveTo(p.x - rs, p.y); ctx.lineTo(p.x - gap, p.y);
      ctx.moveTo(p.x + gap, p.y); ctx.lineTo(p.x + rs, p.y);
      ctx.moveTo(p.x, p.y - rs); ctx.lineTo(p.x, p.y - gap);
      ctx.moveTo(p.x, p.y + gap); ctx.lineTo(p.x, p.y + rs);
      ctx.moveTo(p.x + 1.6, p.y); ctx.arc(p.x, p.y, 1.6, 0, Math.PI*2);
    }
    ctx.stroke();
  }

  if (settings.survPixelate && frame) {
    const gs = Math.max(4, Math.round(bs * 0.13));
    for (const p of points) {
      const bx = p.x + bs/2 + 6;
      const by = p.y - bs/2 - gs*3 - 4;
      for (let iy = 0; iy < 3; iy++) {
        for (let ix = 0; ix < 3; ix++) {
          const sx = p.x + (ix - 1) * 6;
          const sy = p.y + (iy - 1) * 6;
          const col = sampleColor(frame, sx, sy);
          ctx.fillStyle = col;
          ctx.globalAlpha = 0.9;
          ctx.fillRect(bx + ix*gs, by + iy*gs, gs, gs);
        }
      }
    }
    ctx.globalAlpha = 1;
  }

  if (settings.survHud) {
    ctx.font = monoFont;
    ctx.fillStyle = COL;
    ctx.globalAlpha = 0.95;
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      const id = `TGT-${String(i).padStart(3,"0")}`;
      const coord = `[${String(Math.round(p.x)).padStart(4,"0")},${String(Math.round(p.y)).padStart(4,"0")}]`;
      const conf = Math.round(58 + (p.intensity||0.5) * 41);
      ctx.fillText(id, p.x + bs/2 + 6, p.y - bs/2 + fs);
      ctx.fillStyle = "rgba(255,255,255,0.75)";
      ctx.fillText(coord, p.x + bs/2 + 6, p.y - bs/2 + fs*2 + 2);
      ctx.fillStyle = COL;
      ctx.fillText(`CONF ${conf}%`, p.x - bs/2, p.y + bs/2 + fs + 3);
    }
    ctx.globalAlpha = 1;
  }
}

/* ─── Mocap Renderer ─── */

function renderMocap(ctx, points, triangles, settings, width, height, frame) {
  ctx.clearRect(0, 0, width, height);
  const lw = settings.mocLineWeight;
  const PRI = "#ff00cc";
  const ACC = "#00ffff";
  const PS = "rgba(255,0,204,";
  const AS = "rgba(0,255,255,";

  if (settings.mocGrid) {
    const step = Math.max(28, width / 28);
    ctx.strokeStyle = AS + "0.05)";
    ctx.lineWidth = 1;
    ctx.beginPath();
    for (let x = step; x < width; x += step) { ctx.moveTo(x, 0); ctx.lineTo(x, height); }
    for (let y = step; y < height; y += step) { ctx.moveTo(0, y); ctx.lineTo(width, y); }
    ctx.stroke();
    ctx.strokeStyle = AS + "0.22)";
    ctx.beginPath();
    for (let x = step; x < width; x += step) {
      for (let y = step; y < height; y += step) {
        ctx.moveTo(x-3, y); ctx.lineTo(x+3, y);
        ctx.moveTo(x, y-3); ctx.lineTo(x, y+3);
      }
    }
    ctx.stroke();
  }

  if (points.length === 0) return;

  if (settings.mocWireframe && triangles.length > 0) {
    ctx.strokeStyle = AS + "0.35)";
    ctx.lineWidth = lw * 0.5;
    ctx.shadowColor = ACC;
    ctx.shadowBlur = 3;
    ctx.beginPath();
    for (const tri of triangles) {
      ctx.moveTo(tri.a.x, tri.a.y); ctx.lineTo(tri.b.x, tri.b.y);
      ctx.moveTo(tri.b.x, tri.b.y); ctx.lineTo(tri.c.x, tri.c.y);
      ctx.moveTo(tri.c.x, tri.c.y); ctx.lineTo(tri.a.x, tri.a.y);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  if (settings.mocVectors) {
    ctx.strokeStyle = PRI;
    ctx.lineWidth = lw * 0.9;
    ctx.shadowColor = PRI;
    ctx.shadowBlur = 5;
    const vs = settings.mocVectorScale;
    ctx.beginPath();
    for (const p of points) {
      const vx = p.vx || 0, vy = p.vy || 0;
      const mag = Math.hypot(vx, vy);
      if (mag < 0.4) continue;
      const ex = p.x + vx * vs, ey = p.y + vy * vs;
      ctx.moveTo(p.x, p.y); ctx.lineTo(ex, ey);
      const ang = Math.atan2(ey - p.y, ex - p.x);
      const ah = Math.min(9, mag * vs * 0.35);
      ctx.moveTo(ex, ey);
      ctx.lineTo(ex - Math.cos(ang - 0.45) * ah, ey - Math.sin(ang - 0.45) * ah);
      ctx.moveTo(ex, ey);
      ctx.lineTo(ex - Math.cos(ang + 0.45) * ah, ey - Math.sin(ang + 0.45) * ah);
    }
    ctx.stroke();
    ctx.shadowBlur = 0;
  }

  ctx.shadowColor = ACC;
  ctx.shadowBlur = 6;
  for (const p of points) {
    ctx.fillStyle = ACC;
    ctx.globalAlpha = 0.95;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 3.2, 0, Math.PI*2);
    ctx.fill();
    ctx.strokeStyle = PRI;
    ctx.lineWidth = lw * 0.9;
    ctx.beginPath();
    ctx.arc(p.x, p.y, 7, 0, Math.PI*2);
    ctx.stroke();
  }
  ctx.shadowBlur = 0;
  ctx.globalAlpha = 1;

  if (settings.mocRgb && frame) {
    const bs = 5;
    for (const p of points) {
      const px = samplePixel(frame, p.x, p.y);
      if (!px) continue;
      const [r, g, b] = px;
      const bx = p.x + 10, by = p.y + 10;
      ctx.fillStyle = `rgb(${r},0,0)`; ctx.fillRect(bx, by, bs, bs);
      ctx.fillStyle = `rgb(0,${g},0)`; ctx.fillRect(bx + bs, by, bs, bs);
      ctx.fillStyle = `rgb(0,0,${b})`; ctx.fillRect(bx + bs*2, by, bs, bs);
    }
  }

  if (settings.mocData) {
    const fs = Math.max(8, Math.round(width / 210));
    ctx.font = `${fs}px 'Courier New', monospace`;
    for (let i = 0; i < points.length; i++) {
      const p = points[i];
      const id = `N${String(i).padStart(3, "0")}`;
      const v = Math.hypot(p.vx || 0, p.vy || 0);
      ctx.fillStyle = PRI;
      ctx.globalAlpha = 0.95;
      ctx.fillText(id, p.x + 12, p.y - 10);
      ctx.fillStyle = "rgba(255,255,255,0.7)";
      ctx.fillText(`x:${String(Math.round(p.x)).padStart(4)} y:${String(Math.round(p.y)).padStart(4)}`, p.x + 12, p.y - 10 + fs + 1);
      ctx.fillStyle = AS + "0.85)";
      ctx.fillText(`v:${v.toFixed(1)}`, p.x + 12, p.y - 10 + fs*2 + 2);
    }
    ctx.font = `${Math.round(fs*1.35)}px 'Courier New', monospace`;
    ctx.fillStyle = PRI;
    ctx.globalAlpha = 0.95;
    ctx.fillText(`MOCAP_STREAM // ${String(points.length).padStart(2,"0")} NODES`, 10, fs*1.6 + 6);
    ctx.fillStyle = AS + "0.65)";
    ctx.fillText(`CHANNEL_XYV`, 10, fs*3 + 8);
    ctx.globalAlpha = 1;
  }
}

/* ─── Idle animation ─── */

function generateIdlePoints(time, w, h) {
  const pts = [];
  for (let i = 0; i < 40; i++) {
    const a = (i/40)*Math.PI*2+time*0.3, r = 80+Math.sin(time*0.5+i*0.3)*40;
    pts.push({ x: w/2+Math.cos(a)*r+Math.sin(time+i)*25, y: h/2+Math.sin(a)*r+Math.cos(time*0.7+i)*25, intensity: 0.5+Math.sin(time+i*0.5)*0.3, type:"idle", idx:i });
  }
  for (let i = 0; i < 20; i++) {
    const t = time*0.2+i*1.2;
    pts.push({ x: w*(0.15+0.7*((Math.sin(t)+1)/2)), y: h*(0.15+0.7*((Math.cos(t*1.3)+1)/2)), intensity: 0.3+Math.sin(t)*0.2, type:"idle", idx:40+i });
  }
  return pts;
}

/* ─── UI Controls ─── */

function Slider({ label, value, min, max, step, onChange, color }) {
  const pct = ((value-min)/(max-min))*100;
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display:"flex", justifyContent:"space-between", marginBottom:4 }}>
        <span style={{ fontSize:11, color:"#888", fontFamily:"'Courier New',monospace", textTransform:"uppercase", letterSpacing:1 }}>{label}</span>
        <span style={{ fontSize:11, color, fontFamily:"'Courier New',monospace" }}>{Number.isInteger(step)?value:value.toFixed(2)}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={e=>onChange(parseFloat(e.target.value))}
        style={{ width:"100%", height:3, WebkitAppearance:"none", appearance:"none",
          background:`linear-gradient(to right, ${color} ${pct}%, #333 ${pct}%)`, borderRadius:2, outline:"none", cursor:"pointer" }} />
    </div>
  );
}

function Toggle({ label, value, onChange, color }) {
  return (
    <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", marginBottom:10 }}>
      <span style={{ fontSize:11, color:"#888", fontFamily:"'Courier New',monospace", textTransform:"uppercase", letterSpacing:1 }}>{label}</span>
      <div onClick={()=>onChange(!value)} style={{ width:36, height:18, borderRadius:9, background:value?color:"#333", cursor:"pointer", position:"relative", transition:"background 0.2s" }}>
        <div style={{ width:14, height:14, borderRadius:7, background:"#fff", position:"absolute", top:2, left:value?20:2, transition:"left 0.2s" }} />
      </div>
    </div>
  );
}

function Section({ title, children }) {
  return (<><div style={{ height:1, background:"#1a1a1a", margin:"16px 0" }} /><div style={{ fontSize:10, color:"#555", letterSpacing:3, marginBottom:16 }}>{title}</div>{children}</>);
}

/* ═══════════════════════════════════════════════════
   MAIN APP
   ═══════════════════════════════════════════════════ */

export default function Tracer() {
  const [settings, setSettings] = useState(DEFAULTS);
  const [source, setSource] = useState("idle");
  const [recording, setRecording] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const [fps, setFps] = useState(0);
  const [pointCount, setPointCount] = useState(0);
  const [triCount, setTriCount] = useState(0);
  const [videoName, setVideoName] = useState("");
  const [loading, setLoading] = useState(false);
  const [isMobile, setIsMobile] = useState(typeof window !== "undefined" && window.innerWidth < 640);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const overlayRef = useRef(null);
  const offscreenRef = useRef(null);
  const prevGrayRef = useRef(null);
  const trackerRef = useRef({ points: [], nextId: 1 });
  const lastModeRef = useRef(null);
  const sizeRef = useRef({ w: 0, h: 0, dpr: 1 });
  const sessionStartRef = useRef(performance.now());
  const videoAspectRef = useRef(null);
  const resizeFnRef = useRef(null);
  const animRef = useRef(null);
  const recorderRef = useRef(null);
  const recordingLoopRef = useRef(null);
  const chunksRef = useRef([]);
  const fpsRef = useRef({ frames:0, last:performance.now() });
  const settingsRef = useRef(settings);
  const sourceRef = useRef(source);
  const fileInputRef = useRef(null);

  useEffect(() => { settingsRef.current = settings; }, [settings]);
  useEffect(() => { sourceRef.current = source; }, [source]);
  useEffect(() => {
    const onR = () => setIsMobile(window.innerWidth < 640);
    window.addEventListener("resize", onR);
    return () => window.removeEventListener("resize", onR);
  }, []);
  useEffect(() => { if (resizeFnRef.current) resizeFnRef.current(); }, [sidebarOpen, isMobile]);
  useEffect(() => { if (videoRef.current) { videoRef.current.muted = true; videoRef.current.playsInline = true; } }, []);

  const colors = COLORS[settings.colorScheme];
  const set = useCallback((key, val) => setSettings(s => ({ ...s, [key]: val })), []);
  const isSkx = settings.renderMode === "skeletrix";
  const isN4t = settings.renderMode === "n4ture";
  const isSurv = settings.renderMode === "surveil";

  useEffect(() => {
    const c = canvasRef.current, o = overlayRef.current;
    if (!c || !o) return;
    const resize = () => {
      const rect = c.parentElement.getBoundingClientRect();
      const containerW = Math.floor(rect.width), containerH = Math.floor(rect.height);
      let w, h, left, top;
      const aspect = videoAspectRef.current;
      if (aspect) {
        if (containerW / containerH > aspect) {
          h = containerH; w = Math.floor(h * aspect);
        } else {
          w = containerW; h = Math.floor(w / aspect);
        }
        left = Math.floor((containerW - w) / 2);
        top = Math.floor((containerH - h) / 2);
      } else {
        w = containerW; h = containerH; left = 0; top = 0;
      }
      // DPR-aware backing store for crisp lines on mobile/retina; renderers draw in CSS px
      const dpr = Math.min(window.devicePixelRatio || 1, 2);
      c.width = w * dpr; c.height = h * dpr; o.width = w * dpr; o.height = h * dpr;
      sizeRef.current = { w, h, dpr };
      [c, o].forEach(el => {
        el.style.width = w + "px"; el.style.height = h + "px";
        el.style.left = left + "px"; el.style.top = top + "px";
        el.style.right = "auto"; el.style.bottom = "auto";
      });
      // Processing canvas is capped small — detection cost is what kills mobile
      if (!offscreenRef.current) offscreenRef.current = document.createElement("canvas");
      const pw = Math.min(384, Math.max(2, w));
      const ph = Math.max(2, Math.round(pw * h / Math.max(1, w)));
      offscreenRef.current.width = pw; offscreenRef.current.height = ph;
    };
    resizeFnRef.current = resize;
    resize();
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, []);

  /* ─── Main loop ─── */
  useEffect(() => {
    let running = true;
    const loop = (time) => {
      if (!running) return;
      animRef.current = requestAnimationFrame(loop);
      const c = canvasRef.current, o = overlayRef.current, v = videoRef.current;
      if (!c || !o) return;
      const ctx = c.getContext("2d"), overlayCtx = o.getContext("2d");
      const { w, h, dpr } = sizeRef.current;
      if (!w || !h) return;
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      overlayCtx.setTransform(dpr, 0, 0, dpr, 0, 0);
      const s = settingsRef.current, col = COLORS[s.colorScheme];
      const skx = s.renderMode === "skeletrix";
      const n4t = s.renderMode === "n4ture";
      const surv = s.renderMode === "surveil";
      const moc = s.renderMode === "mocap";

      // Fresh tracker when the mode changes (each mode has its own point budget)
      if (lastModeRef.current !== s.renderMode) {
        lastModeRef.current = s.renderMode;
        trackerRef.current = { points: [], nextId: 1 };
        prevGrayRef.current = null;
      }

      fpsRef.current.frames++;
      if (time - fpsRef.current.last > 1000) { setFps(fpsRef.current.frames); fpsRef.current.frames=0; fpsRef.current.last=time; }

      if (sourceRef.current === "idle") {
        ctx.fillStyle = "#0a0a0a"; ctx.fillRect(0, 0, w, h);
        const pts = generateIdlePoints(time/1000, w, h);
        const targetN = skx ? s.skxPoints : n4t ? s.n4tPoints : surv ? s.survPoints : s.mocPoints;
        const thinned = thinPoints(pts, targetN).map((p,i)=>({...p,idx:i,vx:0,vy:0}));
        const tris = delaunay(thinned);
        setPointCount(thinned.length); setTriCount(tris.length);
        if (skx) renderSkeletrix(overlayCtx, thinned, tris, s, col, w, h, null);
        else if (n4t) renderN4ture(overlayCtx, thinned, tris, s, w, h, null);
        else if (surv) renderSurveil(overlayCtx, thinned, s, w, h, null, (time - sessionStartRef.current)/1000);
        else renderMocap(overlayCtx, thinned, tris, s, w, h, null);
        return;
      }

      if (v && v.readyState >= 2 && v.videoWidth > 0) {
        const off = offscreenRef.current;
        if (!off) return;
        const pw = off.width, ph = off.height;
        const offCtx = off.getContext("2d", { willReadFrequently: true });
        offCtx.drawImage(v, 0, 0, pw, ph);
        const img = offCtx.getImageData(0, 0, pw, ph);
        const gray = getGrayscale(img.data, pw, ph);
        const scale = w / pw;
        const frame = { data: img.data, w: pw, h: ph, scale: 1 / scale };

        if (s.background === "black") { ctx.fillStyle="#0a0a0a"; ctx.fillRect(0,0,w,h); }
        else if (s.background === "dimmed") { ctx.drawImage(v,0,0,w,h); ctx.fillStyle=`rgba(0,0,0,${s.dimAmount})`; ctx.fillRect(0,0,w,h); }
        else ctx.drawImage(v, 0, 0, w, h);

        // Guard: proc resolution changed (resize/rotate) → old gray buffer is invalid
        if (prevGrayRef.current && prevGrayRef.current.length !== pw * ph) {
          prevGrayRef.current = null;
          trackerRef.current = { points: [], nextId: 1 };
        }

        const cfg = surv ? { maxPoints: s.survPoints, threshold: s.survThreshold, step: s.sampleRate }
                  : moc  ? { maxPoints: s.mocPoints, threshold: s.mocThreshold, step: s.sampleRate }
                  : n4t  ? { maxPoints: s.n4tPoints, threshold: s.n4tThreshold, step: s.sampleRate, lumBias: true }
                  :        { maxPoints: s.skxPoints, threshold: s.motionThreshold, step: s.sampleRate };
        trackerUpdate(trackerRef.current, prevGrayRef.current, gray, pw, ph, cfg);
        prevGrayRef.current = gray;

        // Oldest (most stable) points first so labels/ids stay consistent
        const ranked = [...trackerRef.current.points].sort((a, b) => (b.age - a.age) || (b.intensity - a.intensity));
        const points = ranked.map((p, i) => ({
          x: p.x * scale, y: p.y * scale,
          vx: p.vx * scale, vy: p.vy * scale,
          intensity: p.intensity, idx: i, age: p.age,
        }));

        const tris = !surv && points.length >= 3 ? delaunay(points) : [];
        setPointCount(points.length); setTriCount(tris.length);

        if (skx) renderSkeletrix(overlayCtx, points, tris, s, col, w, h, frame);
        else if (n4t) renderN4ture(overlayCtx, points, tris, s, w, h, frame);
        else if (surv) renderSurveil(overlayCtx, points, s, w, h, frame, (time - sessionStartRef.current)/1000);
        else renderMocap(overlayCtx, points, tris, s, w, h, frame);
      } else {
        ctx.fillStyle = "#0a0a0a"; ctx.fillRect(0,0,w,h);
        overlayCtx.clearRect(0,0,w,h);
      }
    };
    animRef.current = requestAnimationFrame(loop);
    return () => { running=false; cancelAnimationFrame(animRef.current); };
  }, []);

  const startWebcam = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video:{ facingMode:{ideal:"environment"}, width:{ideal:1280}, height:{ideal:720} }, audio:false });
      const v = videoRef.current; v.srcObject=stream; v.muted=true; await v.play();
      prevGrayRef.current=null; trackerRef.current={points:[],nextId:1}; sessionStartRef.current=performance.now();
      if (v.videoWidth && v.videoHeight) { videoAspectRef.current = v.videoWidth / v.videoHeight; if (resizeFnRef.current) resizeFnRef.current(); }
      setSource("webcam"); setVideoName("Webcam");
    } catch(e) { console.error("Webcam error:", e); }
  }, []);

  const loadVideo = useCallback((file) => {
    setLoading(true);
    const url = URL.createObjectURL(file);
    const v = videoRef.current;
    if (v.srcObject) { v.srcObject.getTracks().forEach(t=>t.stop()); v.srcObject=null; }
    v.muted=true; v.playsInline=true; v.loop=true; v.preload="auto"; v.src=url;
    prevGrayRef.current=null; trackerRef.current={points:[],nextId:1}; sessionStartRef.current=performance.now(); setVideoName(file.name);
    const onReady = () => {
      v.removeEventListener("canplay",onReady); v.removeEventListener("loadeddata",onReady);
      if (v.videoWidth && v.videoHeight) {
        videoAspectRef.current = v.videoWidth / v.videoHeight;
        if (resizeFnRef.current) resizeFnRef.current();
      }
      v.play().then(()=>{setLoading(false);setSource("video")}).catch(()=>{setLoading(false);setSource("video")});
    };
    v.addEventListener("canplay",onReady); v.addEventListener("loadeddata",onReady);
    v.addEventListener("error",()=>{setLoading(false);setVideoName("")},{once:true});
    v.load();
  }, []);

  const stopSource = useCallback(() => {
    const v = videoRef.current;
    if (v.srcObject) { v.srcObject.getTracks().forEach(t=>t.stop()); v.srcObject=null; }
    if (v.src) { v.pause(); v.removeAttribute("src"); v.load(); }
    prevGrayRef.current=null; trackerRef.current={points:[],nextId:1}; sessionStartRef.current=performance.now();
    videoAspectRef.current = null; if (resizeFnRef.current) resizeFnRef.current();
    setSource("idle"); setVideoName("");
  }, []);

  const startRecording = useCallback(() => {
    const c = canvasRef.current, o = overlayRef.current;
    const comp = document.createElement("canvas"); comp.width=c.width; comp.height=c.height;
    const compCtx = comp.getContext("2d");
    let active=true;
    const compLoop = () => { if(!active) return; compCtx.drawImage(c,0,0); compCtx.drawImage(o,0,0); requestAnimationFrame(compLoop); };
    requestAnimationFrame(compLoop);
    recordingLoopRef.current = () => { active=false; };
    const stream = comp.captureStream(30);
    let mime = "video/webm;codecs=vp9";
    if (!MediaRecorder.isTypeSupported(mime)) mime = "video/webm";
    const rec = new MediaRecorder(stream, { mimeType:mime, videoBitsPerSecond:8000000 });
    chunksRef.current = [];
    rec.ondataavailable = e => { if(e.data.size>0) chunksRef.current.push(e.data); };
    rec.onstop = () => { const blob=new Blob(chunksRef.current,{type:"video/webm"}); const a=document.createElement("a"); a.href=URL.createObjectURL(blob); a.download=`tracer_${Date.now()}.webm`; a.click(); };
    recorderRef.current=rec; rec.start(); setRecording(true);
  }, []);

  const stopRecording = useCallback(() => {
    if (recordingLoopRef.current) recordingLoopRef.current();
    if (recorderRef.current) { recorderRef.current.stop(); setRecording(false); }
  }, []);

  const btnBase = {
    background:"transparent", border:`1px solid ${colors.secondary}`, color:colors.primary,
    padding:"10px 24px", fontFamily:"'Courier New',monospace", fontSize:11,
    letterSpacing:2, cursor:"pointer", textTransform:"uppercase", transition:"all 0.3s", borderRadius:0,
  };

  /* ═══ RENDER ═══ */
  return (
    <div style={{ width:"100vw", height:"100vh", background:"#0a0a0a", display:"flex", flexDirection:"column", fontFamily:"'Courier New',monospace", color:"#fff", overflow:"hidden" }}>

      {/* Header */}
      <div style={{ display:"flex", justifyContent:"space-between", alignItems:"center", padding:"12px 20px", borderBottom:"1px solid #1a1a1a", flexShrink:0, zIndex:10 }}>
        <div style={{ display:"flex", alignItems:"center", gap:12 }}>
          <span style={{ fontSize:18, fontWeight:700, letterSpacing:6, color:colors.primary, textShadow:`0 0 20px ${colors.glow}` }}>TRACER</span>
          <span style={{ fontSize:9, color:"#555", letterSpacing:2, paddingTop:2 }}>v1.3</span>
        </div>
        <div style={{ display:"flex", alignItems:"center", gap:16, fontSize:10, color:"#555" }}>
          <span style={{ color: isSkx ? "#ff44ff" : isN4t ? "#44ffaa" : isSurv ? "#ff3838" : "#ff00cc", opacity: 0.7 }}>{isSkx ? "SKX" : isN4t ? "N4T" : isSurv ? "SRV" : "MOC"}</span>
          {!isMobile && <span>{pointCount} PTS</span>}
          {!isMobile && <span>{triCount} TRI</span>}
          <span>{fps} FPS</span>
          <span style={{ color: loading?"#ffbf00":source==="idle"?"#555":colors.primary }}>
            {loading?"⟳ LOADING…":source==="idle"?"STANDBY":source==="webcam"?"● LIVE":`▶ ${videoName}`}
          </span>
          {recording && <span style={{ color:"#ff3333", animation:"blink 1s infinite" }}>● REC</span>}
        </div>
      </div>

      {/* Main */}
      <div style={{ flex:1, display:"flex", overflow:"hidden", position:"relative" }}>
        <div style={{ flex:1, position:"relative", overflow:"hidden" }}>
          <video ref={videoRef} style={{ display:"none" }} playsInline muted />
          <canvas ref={canvasRef} style={{ position:"absolute", inset:0, width:"100%", height:"100%" }} />
          <canvas ref={overlayRef} style={{ position:"absolute", inset:0, width:"100%", height:"100%", pointerEvents:"none" }} />
          {(source==="idle"||loading) && (
            <div style={{ position:"absolute", inset:0, display:"flex", flexDirection:"column", alignItems:"center", justifyContent:"center", gap:16, zIndex:5 }}>
              {loading ? (
                <div style={{ fontSize:10, color:colors.primary, letterSpacing:3, animation:"blink 1.5s infinite" }}>LOADING VIDEO…</div>
              ) : (
                <>
                  <div style={{ fontSize:10, color:"#444", letterSpacing:3, marginBottom:8 }}>SELECT INPUT SOURCE</div>
                  <div style={{ display:"flex", gap:12 }}>
                    <button onClick={startWebcam} style={btnBase}>◉ Webcam</button>
                    <button onClick={()=>fileInputRef.current?.click()} style={btnBase}>▲ Upload Video</button>
                    <input ref={fileInputRef} type="file" accept="video/*" style={{ display:"none" }} onChange={e=>{if(e.target.files[0])loadVideo(e.target.files[0])}} />
                  </div>
                </>
              )}
            </div>
          )}
        </div>

        {/* Sidebar */}
        {sidebarOpen && (
          <div style={{
            ...(isMobile
              ? { position:"absolute", top:0, right:0, bottom:0, width:"min(280px, 82vw)", zIndex:20, boxShadow:"-8px 0 30px rgba(0,0,0,0.65)", borderLeft:"1px solid #1a1a1a" }
              : { width:260, flexShrink:0, borderLeft:"1px solid #1a1a1a" }),
            background:"#0d0d0d", padding:"16px 16px 80px", overflowY:"auto",
            scrollbarWidth:"thin", scrollbarColor:"#333 transparent", WebkitOverflowScrolling:"touch",
          }}>

            {/* Render mode toggle */}
            <div style={{ fontSize:10, color:"#555", letterSpacing:3, marginBottom:12 }}>RENDER MODE</div>
            <div style={{ display:"grid", gridTemplateColumns:"1fr 1fr", gap:6, marginBottom:4 }}>
              {[
                ["skeletrix","SKELETRIX", "#ff44ff", "rgba(255,68,255,0.15)"],
                ["n4ture","N4TURE", "#44ffaa", "rgba(68,255,170,0.15)"],
                ["surveil","SURVEIL", "#ff3838", "rgba(255,56,56,0.15)"],
                ["mocap","MOCAP", "#ff00cc", "rgba(255,0,204,0.15)"],
              ].map(([mode, label, accent, glow]) => (
                <button key={mode} onClick={()=>set("renderMode",mode)} style={{
                  padding:"7px 0", fontSize:9, letterSpacing:1, textTransform:"uppercase",
                  background: settings.renderMode===mode ? glow : "transparent",
                  border: `1px solid ${settings.renderMode===mode ? accent : "#333"}`,
                  color: settings.renderMode===mode ? accent : "#666",
                  fontFamily:"'Courier New',monospace", cursor:"pointer", borderRadius:0,
                }}>{label}</button>
              ))}
            </div>

            <Section title="TRACKING">
              <Slider label="Motion Sensitivity" value={settings.motionThreshold} min={4} max={60} step={1} onChange={v=>set("motionThreshold",v)} color={colors.primary} />
              <Slider label="Sample Density" value={settings.sampleRate} min={2} max={8} step={1} onChange={v=>set("sampleRate",v)} color={colors.primary} />
            </Section>

            {/* Mode-specific controls */}
            {isSkx ? (
              <Section title="SKELETRIX">
                <Slider label="Track Points" value={settings.skxPoints} min={8} max={40} step={1} onChange={v=>set("skxPoints",v)} color="#ff44ff" />
                <Slider label="Swatch Size" value={settings.skxSwatchSize} min={8} max={40} step={2} onChange={v=>set("skxSwatchSize",v)} color="#ff44ff" />
                <Slider label="Marker Size" value={settings.skxMarkerSize} min={3} max={14} step={1} onChange={v=>set("skxMarkerSize",v)} color="#ff44ff" />
                <Toggle label="Extend Lines" value={settings.skxExtendLines} onChange={v=>set("skxExtendLines",v)} color="#ff44ff" />
                <Toggle label="Labels" value={settings.skxLabels} onChange={v=>set("skxLabels",v)} color="#ff44ff" />
                <Slider label="Line Weight" value={settings.lineWeight} min={0.3} max={3} step={0.1} onChange={v=>set("lineWeight",v)} color="#ff44ff" />
              </Section>
            ) : isN4t ? (
              <Section title="N4TURE">
                <Slider label="Track Points" value={settings.n4tPoints} min={8} max={60} step={1} onChange={v=>set("n4tPoints",v)} color="#44ffaa" />
                <Slider label="Sensitivity" value={settings.n4tThreshold} min={5} max={50} step={1} onChange={v=>set("n4tThreshold",v)} color="#44ffaa" />
                <Slider label="Organic Radius" value={settings.n4tOrganicRadius} min={4} max={20} step={1} onChange={v=>set("n4tOrganicRadius",v)} color="#44ffaa" />
                <Toggle label="Flow Lines" value={settings.n4tFlowLines} onChange={v=>set("n4tFlowLines",v)} color="#44ffaa" />
                <Toggle label="Swatches" value={settings.n4tSwatches} onChange={v=>set("n4tSwatches",v)} color="#44ffaa" />
                <Toggle label="Labels" value={settings.n4tLabels} onChange={v=>set("n4tLabels",v)} color="#44ffaa" />
                <Slider label="Line Weight" value={settings.n4tLineWeight} min={0.3} max={8} step={0.1} onChange={v=>set("n4tLineWeight",v)} color="#44ffaa" />
              </Section>
            ) : isSurv ? (
              <Section title="SURVEIL">
                <Slider label="Targets" value={settings.survPoints} min={4} max={40} step={1} onChange={v=>set("survPoints",v)} color="#ff3838" />
                <Slider label="Sensitivity" value={settings.survThreshold} min={8} max={60} step={1} onChange={v=>set("survThreshold",v)} color="#ff3838" />
                <Slider label="Bracket Size" value={settings.survBracketSize} min={16} max={120} step={2} onChange={v=>set("survBracketSize",v)} color="#ff3838" />
                <Toggle label="Corner Brackets" value={settings.survBrackets} onChange={v=>set("survBrackets",v)} color="#ff3838" />
                <Toggle label="Crosshair Reticle" value={settings.survCrosshair} onChange={v=>set("survCrosshair",v)} color="#ff3838" />
                <Toggle label="HUD Overlay" value={settings.survHud} onChange={v=>set("survHud",v)} color="#ff3838" />
                <Toggle label="Scan Line" value={settings.survScanLine} onChange={v=>set("survScanLine",v)} color="#ff3838" />
                <Toggle label="Pixel Mosaic" value={settings.survPixelate} onChange={v=>set("survPixelate",v)} color="#ff3838" />
                <Slider label="Line Weight" value={settings.survLineWeight} min={0.3} max={5} step={0.1} onChange={v=>set("survLineWeight",v)} color="#ff3838" />
              </Section>
            ) : (
              <Section title="MOCAP">
                <Slider label="Nodes" value={settings.mocPoints} min={8} max={120} step={1} onChange={v=>set("mocPoints",v)} color="#ff00cc" />
                <Slider label="Sensitivity" value={settings.mocThreshold} min={5} max={50} step={1} onChange={v=>set("mocThreshold",v)} color="#ff00cc" />
                <Slider label="Vector Scale" value={settings.mocVectorScale} min={1} max={20} step={0.5} onChange={v=>set("mocVectorScale",v)} color="#ff00cc" />
                <Toggle label="Velocity Vectors" value={settings.mocVectors} onChange={v=>set("mocVectors",v)} color="#ff00cc" />
                <Toggle label="Wireframe" value={settings.mocWireframe} onChange={v=>set("mocWireframe",v)} color="#ff00cc" />
                <Toggle label="TD Grid" value={settings.mocGrid} onChange={v=>set("mocGrid",v)} color="#ff00cc" />
                <Toggle label="Data Readouts" value={settings.mocData} onChange={v=>set("mocData",v)} color="#ff00cc" />
                <Toggle label="RGB Sampling" value={settings.mocRgb} onChange={v=>set("mocRgb",v)} color="#ff00cc" />
                <Slider label="Line Weight" value={settings.mocLineWeight} min={0.3} max={5} step={0.1} onChange={v=>set("mocLineWeight",v)} color="#ff00cc" />
              </Section>
            )}

            <Section title="DISPLAY">
              <div style={{ marginBottom:12 }}>
                <div style={{ fontSize:11, color:"#888", marginBottom:6, letterSpacing:1, textTransform:"uppercase" }}>Background</div>
                <div style={{ display:"flex", gap:6 }}>
                  {["original","dimmed","black"].map(bg => (
                    <button key={bg} onClick={()=>set("background",bg)} style={{
                      flex:1, padding:"5px 0", fontSize:9, letterSpacing:1, textTransform:"uppercase",
                      background: settings.background===bg?colors.secondary:"transparent",
                      border:`1px solid ${settings.background===bg?colors.primary:"#333"}`,
                      color: settings.background===bg?colors.primary:"#666",
                      fontFamily:"'Courier New',monospace", cursor:"pointer", borderRadius:0,
                    }}>{bg}</button>
                  ))}
                </div>
              </div>
              {settings.background==="dimmed" && (
                <Slider label="Dim Amount" value={settings.dimAmount} min={0.3} max={0.95} step={0.05} onChange={v=>set("dimAmount",v)} color={colors.primary} />
              )}
              <div style={{ marginBottom:12 }}>
                <div style={{ fontSize:11, color:"#888", marginBottom:6, letterSpacing:1, textTransform:"uppercase" }}>Colour</div>
                <div style={{ display:"flex", gap:6 }}>
                  {Object.keys(COLORS).map(scheme => (
                    <button key={scheme} onClick={()=>set("colorScheme",scheme)} style={{
                      flex:1, height:24, borderRadius:0, cursor:"pointer", background:COLORS[scheme].primary,
                      border: settings.colorScheme===scheme?"2px solid #fff":"2px solid transparent",
                      opacity: settings.colorScheme===scheme?1:0.4, transition:"all 0.2s",
                    }} />
                  ))}
                </div>
              </div>
            </Section>

            <Section title="EXPORT">
              <div style={{ display:"flex", gap:8 }}>
                {!recording ? (
                  <button onClick={startRecording} disabled={source==="idle"} style={{
                    flex:1, padding:"8px 0", fontSize:10, letterSpacing:2, textTransform:"uppercase",
                    background: source==="idle"?"#1a1a1a":"transparent",
                    border:`1px solid ${source==="idle"?"#333":"#ff4444"}`,
                    color: source==="idle"?"#444":"#ff4444",
                    fontFamily:"'Courier New',monospace", cursor:source==="idle"?"default":"pointer", borderRadius:0,
                  }}>● Record</button>
                ) : (
                  <button onClick={stopRecording} style={{
                    flex:1, padding:"8px 0", fontSize:10, letterSpacing:2, textTransform:"uppercase",
                    background:"rgba(255,0,0,0.1)", border:"1px solid #ff4444", color:"#ff4444",
                    fontFamily:"'Courier New',monospace", cursor:"pointer", borderRadius:0, animation:"blink 1s infinite",
                  }}>■ Stop & Save</button>
                )}
              </div>
              {source!=="idle" && (
                <button onClick={stopSource} style={{
                  width:"100%", padding:"8px 0", fontSize:10, letterSpacing:2, textTransform:"uppercase", marginTop:8,
                  background:"transparent", border:"1px solid #333", color:"#666",
                  fontFamily:"'Courier New',monospace", cursor:"pointer", borderRadius:0,
                }}>✕ Stop Source</button>
              )}
            </Section>
          </div>
        )}

        <button onClick={()=>setSidebarOpen(!sidebarOpen)} style={{
          position:"absolute", right:sidebarOpen?260:0, top:12, zIndex:20,
          background:"#0d0d0d", border:"1px solid #1a1a1a", borderRight:sidebarOpen?"none":undefined,
          color:"#555", padding:"6px 8px", cursor:"pointer", fontFamily:"'Courier New',monospace", fontSize:10,
          borderRadius:0, transition:"right 0.2s",
        }}>{sidebarOpen?"▸":"◂"}</button>
      </div>

      <style>{`
        @keyframes blink{0%,100%{opacity:1}50%{opacity:0.3}}
        input[type="range"]::-webkit-slider-thumb{-webkit-appearance:none;width:10px;height:10px;background:#fff;border-radius:0;cursor:pointer;border:none}
        input[type="range"]::-moz-range-thumb{width:10px;height:10px;background:#fff;border-radius:0;cursor:pointer;border:none}
        ::-webkit-scrollbar{width:4px}::-webkit-scrollbar-track{background:transparent}::-webkit-scrollbar-thumb{background:#333}
        *{box-sizing:border-box;margin:0;padding:0}
      `}</style>
    </div>
  );
}
