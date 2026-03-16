import { useState, useRef, useEffect, useCallback } from "react";

/* ═══════════════════════════════════════════════════
   TRACER v1.1 — Motion Tracking Overlay Engine
   Delaunay triangulation + TouchDesigner aesthetic
   ═══════════════════════════════════════════════════ */

const COLORS = {
  cyan:    { primary: "#00ffff", secondary: "#007777", glow: "rgba(0,255,255,0.15)", text: "#00cccc" },
  green:   { primary: "#39ff14", secondary: "#1a7a0a", glow: "rgba(57,255,20,0.15)", text: "#2bcc10" },
  amber:   { primary: "#ffbf00", secondary: "#7a5c00", glow: "rgba(255,191,0,0.15)", text: "#cc9900" },
  magenta: { primary: "#ff44ff", secondary: "#7a007a", glow: "rgba(255,68,255,0.15)", text: "#cc33cc" },
  white:   { primary: "#ffffff", secondary: "#666666", glow: "rgba(255,255,255,0.1)", text: "#999999" },
};

const DEFAULTS = {
  colorScheme: "cyan",
  trailLength: 25,
  trailDecay: 0.92,
  lineWeight: 1.0,
  glowIntensity: 12,
  pointSize: 2.5,
  showMesh: true,
  showTrails: true,
  showLabels: false,
  showEdges: true,
  showPoints: true,
  background: "dimmed",
  motionThreshold: 35,
  sampleRate: 5,
  dimAmount: 0.75,
  maxPoints: 120,
  triangleFill: 0.06,
};

/* ─── Delaunay Triangulation (Bowyer-Watson) ─── */

function delaunay(points) {
  if (points.length < 3) return [];
  let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
  for (const p of points) {
    if (p.x < minX) minX = p.x;
    if (p.y < minY) minY = p.y;
    if (p.x > maxX) maxX = p.x;
    if (p.y > maxY) maxY = p.y;
  }
  const dx = maxX - minX, dy = maxY - minY;
  const dmax = Math.max(dx, dy);
  const midX = (minX + maxX) / 2, midY = (minY + maxY) / 2;
  const p0 = { x: midX - 20 * dmax, y: midY - dmax, idx: -1 };
  const p1 = { x: midX, y: midY + 20 * dmax, idx: -2 };
  const p2 = { x: midX + 20 * dmax, y: midY - dmax, idx: -3 };
  let triangles = [{ a: p0, b: p1, c: p2 }];

  for (const p of points) {
    const good = [];
    const bad = [];
    for (const tri of triangles) {
      if (inCircumcircle(p, tri)) bad.push(tri);
      else good.push(tri);
    }
    const edges = [];
    for (const tri of bad) {
      edges.push([tri.a, tri.b]);
      edges.push([tri.b, tri.c]);
      edges.push([tri.c, tri.a]);
    }
    const boundary = [];
    for (let i = 0; i < edges.length; i++) {
      let shared = false;
      for (let j = 0; j < edges.length; j++) {
        if (i !== j && edges[i][0] === edges[j][1] && edges[i][1] === edges[j][0]) {
          shared = true; break;
        }
      }
      if (!shared) boundary.push(edges[i]);
    }
    for (const edge of boundary) good.push({ a: edge[0], b: edge[1], c: p });
    triangles = good;
  }

  return triangles.filter(tri =>
    tri.a.idx >= 0 && tri.b.idx >= 0 && tri.c.idx >= 0
  );
}

function inCircumcircle(p, tri) {
  const ax = tri.a.x - p.x, ay = tri.a.y - p.y;
  const bx = tri.b.x - p.x, by = tri.b.y - p.y;
  const cx = tri.c.x - p.x, cy = tri.c.y - p.y;
  return (ax * ax + ay * ay) * (bx * cy - cx * by) -
         (bx * bx + by * by) * (ax * cy - cx * ay) +
         (cx * cx + cy * cy) * (ax * by - bx * ay) > 0;
}

/* ─── Detection ─── */

function getGrayscale(data, w, h) {
  const gray = new Uint8Array(w * h);
  for (let i = 0; i < w * h; i++) {
    const j = i * 4;
    gray[i] = (data[j] * 0.299 + data[j + 1] * 0.587 + data[j + 2] * 0.114) | 0;
  }
  return gray;
}

function detectMotion(curr, prev, w, h, threshold, step) {
  const points = [];
  if (!prev) return points;
  for (let y = step; y < h - step; y += step) {
    for (let x = step; x < w - step; x += step) {
      const i = y * w + x;
      const diff = Math.abs(curr[i] - prev[i]);
      if (diff > threshold) {
        points.push({ x, y, intensity: Math.min(diff / 120, 1), type: "motion" });
      }
    }
  }
  return points;
}

function detectEdges(gray, w, h, step) {
  const points = [];
  for (let y = step; y < h - step; y += step * 2) {
    for (let x = step; x < w - step; x += step * 2) {
      const i = y * w + x;
      const gx = -gray[i - w - 1] + gray[i - w + 1] - 2 * gray[i - 1] + 2 * gray[i + 1] - gray[i + w - 1] + gray[i + w + 1];
      const gy = -gray[i - w - 1] - 2 * gray[i - w] - gray[i - w + 1] + gray[i + w - 1] + 2 * gray[i + w] + gray[i + w + 1];
      const mag = Math.sqrt(gx * gx + gy * gy);
      if (mag > 80) {
        points.push({ x, y, intensity: Math.min(mag / 300, 1), type: "edge" });
      }
    }
  }
  return points;
}

function thinPoints(points, maxCount) {
  if (points.length <= maxCount) return points;
  const sorted = [...points].sort((a, b) => b.intensity - a.intensity);
  const step = Math.ceil(sorted.length / maxCount);
  const result = [];
  for (let i = 0; i < sorted.length && result.length < maxCount; i += step) {
    result.push(sorted[i]);
  }
  return result;
}

/* ─── Rendering ─── */

function renderFrame(ctx, points, triangles, trailBuffer, settings, colors, width, height) {
  ctx.clearRect(0, 0, width, height);

  // Delaunay triangle mesh
  if (settings.showMesh && triangles.length > 0) {
    ctx.shadowColor = colors.primary;
    ctx.shadowBlur = settings.glowIntensity * 0.5;

    // Subtle triangle fills
    if (settings.triangleFill > 0) {
      for (const tri of triangles) {
        const avgI = (tri.a.intensity + tri.b.intensity + tri.c.intensity) / 3;
        ctx.globalAlpha = settings.triangleFill * (0.3 + avgI * 0.7);
        ctx.fillStyle = colors.primary;
        ctx.beginPath();
        ctx.moveTo(tri.a.x, tri.a.y);
        ctx.lineTo(tri.b.x, tri.b.y);
        ctx.lineTo(tri.c.x, tri.c.y);
        ctx.closePath();
        ctx.fill();
      }
    }

    // Triangle edges — the main visual
    ctx.strokeStyle = colors.primary;
    ctx.lineWidth = settings.lineWeight;
    ctx.globalAlpha = 0.5;
    ctx.shadowBlur = settings.glowIntensity * 0.7;
    ctx.beginPath();
    for (const tri of triangles) {
      ctx.moveTo(tri.a.x, tri.a.y);
      ctx.lineTo(tri.b.x, tri.b.y);
      ctx.lineTo(tri.c.x, tri.c.y);
      ctx.closePath();
    }
    ctx.stroke();

    // Inner centroid lines — secondary detail
    ctx.strokeStyle = colors.secondary;
    ctx.lineWidth = settings.lineWeight * 0.35;
    ctx.globalAlpha = 0.12;
    ctx.shadowBlur = settings.glowIntensity * 0.3;
    ctx.beginPath();
    for (const tri of triangles) {
      const cx = (tri.a.x + tri.b.x + tri.c.x) / 3;
      const cy = (tri.a.y + tri.b.y + tri.c.y) / 3;
      ctx.moveTo(cx, cy); ctx.lineTo(tri.a.x, tri.a.y);
      ctx.moveTo(cx, cy); ctx.lineTo(tri.b.x, tri.b.y);
      ctx.moveTo(cx, cy); ctx.lineTo(tri.c.x, tri.c.y);
    }
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  // Trails
  if (settings.showTrails && trailBuffer.length > 1) {
    ctx.shadowColor = colors.primary;
    ctx.shadowBlur = settings.glowIntensity * 0.6;
    ctx.lineWidth = settings.lineWeight * 0.8;
    for (let t = 1; t < trailBuffer.length; t++) {
      const age = trailBuffer.length - t;
      const alpha = Math.pow(settings.trailDecay, age) * 0.5;
      if (alpha < 0.02) continue;
      ctx.strokeStyle = colors.primary;
      ctx.globalAlpha = alpha;
      const prev = trailBuffer[t - 1];
      const curr = trailBuffer[t];
      if (!prev.length || !curr.length) continue;
      ctx.beginPath();
      const step = Math.max(1, Math.floor(curr.length / 60));
      for (let i = 0; i < curr.length; i += step) {
        const p = curr[i];
        let closest = null, closestDist = 1200;
        for (let j = 0; j < prev.length; j += step) {
          const ddx = p.x - prev[j].x, ddy = p.y - prev[j].y;
          const d = ddx * ddx + ddy * ddy;
          if (d < closestDist) { closestDist = d; closest = prev[j]; }
        }
        if (closest) { ctx.moveTo(closest.x, closest.y); ctx.lineTo(p.x, p.y); }
      }
      ctx.stroke();
    }
    ctx.globalAlpha = 1;
  }

  // Points
  if (settings.showPoints) {
    ctx.shadowColor = colors.primary;
    ctx.shadowBlur = settings.glowIntensity;
    ctx.fillStyle = colors.primary;
    for (const p of points) {
      ctx.globalAlpha = 0.5 + p.intensity * 0.5;
      ctx.beginPath();
      ctx.arc(p.x, p.y, settings.pointSize * (0.6 + p.intensity * 0.6), 0, Math.PI * 2);
      ctx.fill();
    }
    ctx.globalAlpha = 1;
  }

  // Coordinate labels
  if (settings.showLabels && points.length > 0) {
    ctx.shadowBlur = 0;
    ctx.font = `${Math.max(8, Math.round(width / 140))}px 'Courier New', monospace`;
    ctx.fillStyle = colors.text;
    ctx.globalAlpha = 0.6;
    const labelStep = Math.max(1, Math.floor(points.length / 18));
    for (let i = 0; i < points.length; i += labelStep) {
      const p = points[i];
      ctx.fillText(`${(p.x / width).toFixed(3)}, ${(p.y / height).toFixed(3)}`, p.x + 6, p.y - 4);
    }
    ctx.globalAlpha = 1;
  }
}

/* ─── Idle animation ─── */

function generateIdlePoints(time, w, h) {
  const pts = [];
  for (let i = 0; i < 40; i++) {
    const angle = (i / 40) * Math.PI * 2 + time * 0.3;
    const r = 80 + Math.sin(time * 0.5 + i * 0.3) * 40;
    pts.push({
      x: w / 2 + Math.cos(angle) * r + Math.sin(time + i) * 25,
      y: h / 2 + Math.sin(angle) * r + Math.cos(time * 0.7 + i) * 25,
      intensity: 0.5 + Math.sin(time + i * 0.5) * 0.3, type: "idle", idx: i,
    });
  }
  for (let i = 0; i < 20; i++) {
    const t = time * 0.2 + i * 1.2;
    pts.push({
      x: w * (0.15 + 0.7 * ((Math.sin(t) + 1) / 2)),
      y: h * (0.15 + 0.7 * ((Math.cos(t * 1.3) + 1) / 2)),
      intensity: 0.3 + Math.sin(t) * 0.2, type: "idle", idx: 40 + i,
    });
  }
  return pts;
}

/* ─── UI Controls ─── */

function Slider({ label, value, min, max, step, onChange, color }) {
  const pct = ((value - min) / (max - min)) * 100;
  return (
    <div style={{ marginBottom: 12 }}>
      <div style={{ display: "flex", justifyContent: "space-between", marginBottom: 4 }}>
        <span style={{ fontSize: 11, color: "#888", fontFamily: "'Courier New', monospace", textTransform: "uppercase", letterSpacing: 1 }}>{label}</span>
        <span style={{ fontSize: 11, color, fontFamily: "'Courier New', monospace" }}>{Number.isInteger(step) ? value : value.toFixed(2)}</span>
      </div>
      <input type="range" min={min} max={max} step={step} value={value} onChange={e => onChange(parseFloat(e.target.value))}
        style={{ width: "100%", height: 3, WebkitAppearance: "none", appearance: "none",
          background: `linear-gradient(to right, ${color} ${pct}%, #333 ${pct}%)`,
          borderRadius: 2, outline: "none", cursor: "pointer" }} />
    </div>
  );
}

function Toggle({ label, value, onChange, color }) {
  return (
    <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", marginBottom: 10 }}>
      <span style={{ fontSize: 11, color: "#888", fontFamily: "'Courier New', monospace", textTransform: "uppercase", letterSpacing: 1 }}>{label}</span>
      <div onClick={() => onChange(!value)}
        style={{ width: 36, height: 18, borderRadius: 9, background: value ? color : "#333", cursor: "pointer", position: "relative", transition: "background 0.2s" }}>
        <div style={{ width: 14, height: 14, borderRadius: 7, background: "#fff", position: "absolute", top: 2, left: value ? 20 : 2, transition: "left 0.2s" }} />
      </div>
    </div>
  );
}

function Section({ title, children }) {
  return (<><div style={{ height: 1, background: "#1a1a1a", margin: "16px 0" }} /><div style={{ fontSize: 10, color: "#555", letterSpacing: 3, marginBottom: 16 }}>{title}</div>{children}</>);
}

/* ═══ MAIN APP ═══ */

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

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const overlayRef = useRef(null);
  const offscreenRef = useRef(null);
  const prevGrayRef = useRef(null);
  const trailRef = useRef([]);
  const animRef = useRef(null);
  const recorderRef = useRef(null);
  const recordingLoopRef = useRef(null);
  const chunksRef = useRef([]);
  const fpsRef = useRef({ frames: 0, last: performance.now() });
  const settingsRef = useRef(settings);
  const sourceRef = useRef(source);
  const fileInputRef = useRef(null);

  useEffect(() => { settingsRef.current = settings; }, [settings]);
  useEffect(() => { sourceRef.current = source; }, [source]);
  useEffect(() => { if (videoRef.current) { videoRef.current.muted = true; videoRef.current.playsInline = true; } }, []);

  const colors = COLORS[settings.colorScheme];
  const set = useCallback((key, val) => setSettings(s => ({ ...s, [key]: val })), []);

  useEffect(() => {
    const c = canvasRef.current, o = overlayRef.current;
    if (!c || !o) return;
    const resize = () => {
      const rect = c.parentElement.getBoundingClientRect();
      const w = Math.floor(rect.width), h = Math.floor(rect.height);
      c.width = w; c.height = h; o.width = w; o.height = h;
      if (!offscreenRef.current) offscreenRef.current = document.createElement("canvas");
      offscreenRef.current.width = w; offscreenRef.current.height = h;
    };
    resize();
    window.addEventListener("resize", resize);
    return () => window.removeEventListener("resize", resize);
  }, []);

  useEffect(() => {
    let running = true;
    const loop = (time) => {
      if (!running) return;
      animRef.current = requestAnimationFrame(loop);
      const c = canvasRef.current, o = overlayRef.current, v = videoRef.current;
      if (!c || !o) return;
      const ctx = c.getContext("2d"), overlayCtx = o.getContext("2d");
      const w = c.width, h = c.height;
      const s = settingsRef.current, col = COLORS[s.colorScheme];

      fpsRef.current.frames++;
      if (time - fpsRef.current.last > 1000) {
        setFps(fpsRef.current.frames); fpsRef.current.frames = 0; fpsRef.current.last = time;
      }

      if (sourceRef.current === "idle") {
        ctx.fillStyle = "#0a0a0a"; ctx.fillRect(0, 0, w, h);
        const pts = generateIdlePoints(time / 1000, w, h);
        const tris = delaunay(pts);
        setPointCount(pts.length); setTriCount(tris.length);
        trailRef.current.push(pts);
        if (trailRef.current.length > s.trailLength) trailRef.current.shift();
        renderFrame(overlayCtx, pts, tris, trailRef.current, s, col, w, h);
        return;
      }

      if (v && v.readyState >= 2 && v.videoWidth > 0) {
        const off = offscreenRef.current;
        if (!off) return;
        const offCtx = off.getContext("2d", { willReadFrequently: true });
        offCtx.drawImage(v, 0, 0, w, h);
        const frame = offCtx.getImageData(0, 0, w, h);
        const gray = getGrayscale(frame.data, w, h);

        if (s.background === "black") { ctx.fillStyle = "#0a0a0a"; ctx.fillRect(0, 0, w, h); }
        else if (s.background === "dimmed") { ctx.drawImage(v, 0, 0, w, h); ctx.fillStyle = `rgba(0,0,0,${s.dimAmount})`; ctx.fillRect(0, 0, w, h); }
        else ctx.drawImage(v, 0, 0, w, h);

        let rawPoints = [];
        rawPoints = rawPoints.concat(detectMotion(gray, prevGrayRef.current, w, h, s.motionThreshold, s.sampleRate));
        if (s.showEdges) rawPoints = rawPoints.concat(detectEdges(gray, w, h, s.sampleRate));
        prevGrayRef.current = gray;

        const points = thinPoints(rawPoints, s.maxPoints).map((p, i) => ({ ...p, idx: i }));
        const tris = points.length >= 3 ? delaunay(points) : [];
        setPointCount(points.length); setTriCount(tris.length);

        trailRef.current.push(points);
        if (trailRef.current.length > s.trailLength) trailRef.current.shift();
        renderFrame(overlayCtx, points, tris, trailRef.current, s, col, w, h);
      } else {
        ctx.fillStyle = "#0a0a0a"; ctx.fillRect(0, 0, w, h);
        overlayCtx.clearRect(0, 0, w, h);
      }
    };
    animRef.current = requestAnimationFrame(loop);
    return () => { running = false; cancelAnimationFrame(animRef.current); };
  }, []);

  const startWebcam = useCallback(async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({ video: { width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false });
      const v = videoRef.current; v.srcObject = stream; v.muted = true; await v.play();
      prevGrayRef.current = null; trailRef.current = [];
      setSource("webcam"); setVideoName("Webcam");
    } catch (e) { console.error("Webcam error:", e); }
  }, []);

  const loadVideo = useCallback((file) => {
    setLoading(true);
    const url = URL.createObjectURL(file);
    const v = videoRef.current;
    if (v.srcObject) { v.srcObject.getTracks().forEach(t => t.stop()); v.srcObject = null; }
    v.muted = true; v.playsInline = true; v.loop = true; v.preload = "auto"; v.src = url;
    prevGrayRef.current = null; trailRef.current = []; setVideoName(file.name);
    const onReady = () => {
      v.removeEventListener("canplay", onReady); v.removeEventListener("loadeddata", onReady);
      v.play().then(() => { setLoading(false); setSource("video"); }).catch(() => { setLoading(false); setSource("video"); });
    };
    v.addEventListener("canplay", onReady); v.addEventListener("loadeddata", onReady);
    v.addEventListener("error", () => { setLoading(false); setVideoName(""); }, { once: true });
    v.load();
  }, []);

  const stopSource = useCallback(() => {
    const v = videoRef.current;
    if (v.srcObject) { v.srcObject.getTracks().forEach(t => t.stop()); v.srcObject = null; }
    if (v.src) { v.pause(); v.removeAttribute("src"); v.load(); }
    prevGrayRef.current = null; trailRef.current = [];
    setSource("idle"); setVideoName("");
  }, []);

  const startRecording = useCallback(() => {
    const c = canvasRef.current, o = overlayRef.current;
    const comp = document.createElement("canvas"); comp.width = c.width; comp.height = c.height;
    const compCtx = comp.getContext("2d");
    let active = true;
    const compLoop = () => { if (!active) return; compCtx.drawImage(c, 0, 0); compCtx.drawImage(o, 0, 0); requestAnimationFrame(compLoop); };
    requestAnimationFrame(compLoop);
    recordingLoopRef.current = () => { active = false; };
    const stream = comp.captureStream(30);
    let mime = "video/webm;codecs=vp9";
    if (!MediaRecorder.isTypeSupported(mime)) mime = "video/webm";
    const rec = new MediaRecorder(stream, { mimeType: mime, videoBitsPerSecond: 8000000 });
    chunksRef.current = [];
    rec.ondataavailable = e => { if (e.data.size > 0) chunksRef.current.push(e.data); };
    rec.onstop = () => { const blob = new Blob(chunksRef.current, { type: "video/webm" }); const a = document.createElement("a"); a.href = URL.createObjectURL(blob); a.download = `tracer_${Date.now()}.webm`; a.click(); };
    recorderRef.current = rec; rec.start(); setRecording(true);
  }, []);

  const stopRecording = useCallback(() => {
    if (recordingLoopRef.current) recordingLoopRef.current();
    if (recorderRef.current) { recorderRef.current.stop(); setRecording(false); }
  }, []);

  const btnBase = {
    background: "transparent", border: `1px solid ${colors.secondary}`, color: colors.primary,
    padding: "10px 24px", fontFamily: "'Courier New', monospace", fontSize: 11,
    letterSpacing: 2, cursor: "pointer", textTransform: "uppercase", transition: "all 0.3s", borderRadius: 0,
  };

  return (
    <div style={{ width: "100vw", height: "100vh", background: "#0a0a0a", display: "flex", flexDirection: "column", fontFamily: "'Courier New', monospace", color: "#fff", overflow: "hidden" }}>
      <div style={{ display: "flex", justifyContent: "space-between", alignItems: "center", padding: "12px 20px", borderBottom: "1px solid #1a1a1a", flexShrink: 0, zIndex: 10 }}>
        <div style={{ display: "flex", alignItems: "center", gap: 12 }}>
          <span style={{ fontSize: 18, fontWeight: 700, letterSpacing: 6, color: colors.primary, textShadow: `0 0 20px ${colors.glow}` }}>TRACER</span>
          <span style={{ fontSize: 9, color: "#555", letterSpacing: 2, paddingTop: 2 }}>v1.1</span>
        </div>
        <div style={{ display: "flex", alignItems: "center", gap: 16, fontSize: 10, color: "#555" }}>
          <span>{pointCount} PTS</span>
          <span>{triCount} TRI</span>
          <span>{fps} FPS</span>
          <span style={{ color: loading ? "#ffbf00" : source === "idle" ? "#555" : colors.primary }}>
            {loading ? "⟳ LOADING…" : source === "idle" ? "STANDBY" : source === "webcam" ? "● LIVE" : `▶ ${videoName}`}
          </span>
          {recording && <span style={{ color: "#ff3333", animation: "blink 1s infinite" }}>● REC</span>}
        </div>
      </div>

      <div style={{ flex: 1, display: "flex", overflow: "hidden", position: "relative" }}>
        <div style={{ flex: 1, position: "relative", overflow: "hidden" }}>
          <video ref={videoRef} style={{ display: "none" }} playsInline muted />
          <canvas ref={canvasRef} style={{ position: "absolute", inset: 0, width: "100%", height: "100%" }} />
          <canvas ref={overlayRef} style={{ position: "absolute", inset: 0, width: "100%", height: "100%", pointerEvents: "none" }} />
          {(source === "idle" || loading) && (
            <div style={{ position: "absolute", inset: 0, display: "flex", flexDirection: "column", alignItems: "center", justifyContent: "center", gap: 16, zIndex: 5 }}>
              {loading ? (
                <div style={{ fontSize: 10, color: colors.primary, letterSpacing: 3, animation: "blink 1.5s infinite" }}>LOADING VIDEO…</div>
              ) : (
                <>
                  <div style={{ fontSize: 10, color: "#444", letterSpacing: 3, marginBottom: 8 }}>SELECT INPUT SOURCE</div>
                  <div style={{ display: "flex", gap: 12 }}>
                    <button onClick={startWebcam} style={btnBase}>◉ Webcam</button>
                    <button onClick={() => fileInputRef.current?.click()} style={btnBase}>▲ Upload Video</button>
                    <input ref={fileInputRef} type="file" accept="video/*" style={{ display: "none" }} onChange={e => { if (e.target.files[0]) loadVideo(e.target.files[0]); }} />
                  </div>
                </>
              )}
            </div>
          )}
        </div>

        {sidebarOpen && (
          <div style={{ width: 260, background: "#0d0d0d", borderLeft: "1px solid #1a1a1a", padding: "16px 16px 80px", overflowY: "auto", flexShrink: 0, scrollbarWidth: "thin", scrollbarColor: "#333 transparent" }}>
            <Section title="TRACKING">
              <Slider label="Motion Sensitivity" value={settings.motionThreshold} min={10} max={100} step={1} onChange={v => set("motionThreshold", v)} color={colors.primary} />
              <Slider label="Sample Density" value={settings.sampleRate} min={2} max={12} step={1} onChange={v => set("sampleRate", v)} color={colors.primary} />
              <Slider label="Max Points" value={settings.maxPoints} min={30} max={300} step={10} onChange={v => set("maxPoints", v)} color={colors.primary} />
              <Toggle label="Edge Detection" value={settings.showEdges} onChange={v => set("showEdges", v)} color={colors.primary} />
            </Section>
            <Section title="TRIANGULATION">
              <Toggle label="Mesh" value={settings.showMesh} onChange={v => set("showMesh", v)} color={colors.primary} />
              <Slider label="Triangle Fill" value={settings.triangleFill} min={0} max={0.2} step={0.01} onChange={v => set("triangleFill", v)} color={colors.primary} />
              <Toggle label="Points" value={settings.showPoints} onChange={v => set("showPoints", v)} color={colors.primary} />
              <Toggle label="Trails" value={settings.showTrails} onChange={v => set("showTrails", v)} color={colors.primary} />
              <Toggle label="Coordinates" value={settings.showLabels} onChange={v => set("showLabels", v)} color={colors.primary} />
            </Section>
            <Section title="STYLE">
              <Slider label="Point Size" value={settings.pointSize} min={1} max={6} step={0.5} onChange={v => set("pointSize", v)} color={colors.primary} />
              <Slider label="Line Weight" value={settings.lineWeight} min={0.3} max={3} step={0.1} onChange={v => set("lineWeight", v)} color={colors.primary} />
              <Slider label="Glow Intensity" value={settings.glowIntensity} min={0} max={30} step={1} onChange={v => set("glowIntensity", v)} color={colors.primary} />
            </Section>
            <Section title="TRAILS">
              <Slider label="Trail Length" value={settings.trailLength} min={2} max={80} step={1} onChange={v => set("trailLength", v)} color={colors.primary} />
              <Slider label="Trail Decay" value={settings.trailDecay} min={0.8} max={0.99} step={0.01} onChange={v => set("trailDecay", v)} color={colors.primary} />
            </Section>
            <Section title="DISPLAY">
              <div style={{ marginBottom: 12 }}>
                <div style={{ fontSize: 11, color: "#888", marginBottom: 6, letterSpacing: 1, textTransform: "uppercase" }}>Background</div>
                <div style={{ display: "flex", gap: 6 }}>
                  {["original", "dimmed", "black"].map(bg => (
                    <button key={bg} onClick={() => set("background", bg)} style={{
                      flex: 1, padding: "5px 0", fontSize: 9, letterSpacing: 1, textTransform: "uppercase",
                      background: settings.background === bg ? colors.secondary : "transparent",
                      border: `1px solid ${settings.background === bg ? colors.primary : "#333"}`,
                      color: settings.background === bg ? colors.primary : "#666",
                      fontFamily: "'Courier New', monospace", cursor: "pointer", borderRadius: 0,
                    }}>{bg}</button>
                  ))}
                </div>
              </div>
              {settings.background === "dimmed" && (
                <Slider label="Dim Amount" value={settings.dimAmount} min={0.3} max={0.95} step={0.05} onChange={v => set("dimAmount", v)} color={colors.primary} />
              )}
              <div style={{ marginBottom: 12 }}>
                <div style={{ fontSize: 11, color: "#888", marginBottom: 6, letterSpacing: 1, textTransform: "uppercase" }}>Colour</div>
                <div style={{ display: "flex", gap: 6 }}>
                  {Object.keys(COLORS).map(scheme => (
                    <button key={scheme} onClick={() => set("colorScheme", scheme)} style={{
                      flex: 1, height: 24, borderRadius: 0, cursor: "pointer", background: COLORS[scheme].primary,
                      border: settings.colorScheme === scheme ? "2px solid #fff" : "2px solid transparent",
                      opacity: settings.colorScheme === scheme ? 1 : 0.4, transition: "all 0.2s",
                    }} />
                  ))}
                </div>
              </div>
            </Section>
            <Section title="EXPORT">
              <div style={{ display: "flex", gap: 8 }}>
                {!recording ? (
                  <button onClick={startRecording} disabled={source === "idle"} style={{
                    flex: 1, padding: "8px 0", fontSize: 10, letterSpacing: 2, textTransform: "uppercase",
                    background: source === "idle" ? "#1a1a1a" : "transparent",
                    border: `1px solid ${source === "idle" ? "#333" : "#ff4444"}`,
                    color: source === "idle" ? "#444" : "#ff4444",
                    fontFamily: "'Courier New', monospace", cursor: source === "idle" ? "default" : "pointer", borderRadius: 0,
                  }}>● Record</button>
                ) : (
                  <button onClick={stopRecording} style={{
                    flex: 1, padding: "8px 0", fontSize: 10, letterSpacing: 2, textTransform: "uppercase",
                    background: "rgba(255,0,0,0.1)", border: "1px solid #ff4444", color: "#ff4444",
                    fontFamily: "'Courier New', monospace", cursor: "pointer", borderRadius: 0, animation: "blink 1s infinite",
                  }}>■ Stop & Save</button>
                )}
              </div>
              {source !== "idle" && (
                <button onClick={stopSource} style={{
                  width: "100%", padding: "8px 0", fontSize: 10, letterSpacing: 2, textTransform: "uppercase", marginTop: 8,
                  background: "transparent", border: "1px solid #333", color: "#666",
                  fontFamily: "'Courier New', monospace", cursor: "pointer", borderRadius: 0,
                }}>✕ Stop Source</button>
              )}
            </Section>
          </div>
        )}

        <button onClick={() => setSidebarOpen(!sidebarOpen)} style={{
          position: "absolute", right: sidebarOpen ? 260 : 0, top: 12, zIndex: 20,
          background: "#0d0d0d", border: "1px solid #1a1a1a", borderRight: sidebarOpen ? "none" : undefined,
          color: "#555", padding: "6px 8px", cursor: "pointer", fontFamily: "'Courier New', monospace", fontSize: 10,
          borderRadius: 0, transition: "right 0.2s",
        }}>{sidebarOpen ? "▸" : "◂"}</button>
      </div>

      <style>{`
        @keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.3} }
        input[type="range"]::-webkit-slider-thumb { -webkit-appearance:none; width:10px; height:10px; background:#fff; border-radius:0; cursor:pointer; border:none; }
        input[type="range"]::-moz-range-thumb { width:10px; height:10px; background:#fff; border-radius:0; cursor:pointer; border:none; }
        ::-webkit-scrollbar { width:4px; }
        ::-webkit-scrollbar-track { background:transparent; }
        ::-webkit-scrollbar-thumb { background:#333; }
        * { box-sizing:border-box; margin:0; padding:0; }
      `}</style>
    </div>
  );
}
