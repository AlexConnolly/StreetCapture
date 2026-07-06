import { useRef } from "react";
import type { DvrSegment, DvrTimeline } from "../api";

// A 24h scrub bar. Bar HEIGHT = movement intensity (summed per-artifact motion),
// so parked cars / standing people barely register — we track how much moved,
// not what it was. Drag to scrub, or switch to select-mode to save a clip.

function clock(ts: number) {
  return new Date(ts * 1000).toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
}

export function findSeek(segments: DvrSegment[], ts: number): { seg: DvrSegment; offset: number } | null {
  if (!segments.length) return null;
  for (const s of segments) {
    if (ts >= s.start && ts < s.start + s.duration) return { seg: s, offset: ts - s.start };
  }
  let best = segments[0];
  for (const s of segments) if (Math.abs(s.start - ts) < Math.abs(best.start - ts)) best = s;
  const offset = ts < best.start ? 0 : Math.max(0, Math.min(best.duration - 0.1, ts - best.start));
  return { seg: best, offset };
}

export function Timeline({
  timeline, segments, playheadTs, live, mode, selection, onSeek, onSelect, onLive,
}: {
  timeline: DvrTimeline | null;
  segments: DvrSegment[];
  playheadTs: number;
  live: boolean;
  mode: "scrub" | "select";
  selection: { a: number; b: number } | null;
  onSeek: (ts: number) => void;
  onSelect: (sel: { a: number; b: number } | null) => void;
  onLive: () => void;
}) {
  const barRef = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const anchor = useRef<number | null>(null);

  if (!timeline) {
    return (
      <div className="border-t border-line bg-panel px-3 py-2 text-center text-[11px] text-zinc-600">
        recording… timeline appears once the DVR has footage
      </div>
    );
  }

  const { start, end } = timeline;
  const span = Math.max(1, end - start);
  const nb = timeline.motion.length || timeline.person.length;
  const heights = timeline.motion;
  const maxH = Math.max(1e-6, ...heights);

  function tsFromClientX(clientX: number): number {
    const el = barRef.current;
    if (!el) return end;
    const r = el.getBoundingClientRect();
    const f = Math.max(0, Math.min(1, (clientX - r.left) / r.width));
    return start + f * span;
  }
  const pos = (ts: number) => `${Math.max(0, Math.min(100, ((ts - start) / span) * 100))}%`;

  const onDown = (e: React.PointerEvent) => {
    dragging.current = true;
    (e.target as HTMLElement).setPointerCapture?.(e.pointerId);
    const ts = tsFromClientX(e.clientX);
    if (mode === "select") { anchor.current = ts; onSelect({ a: ts, b: ts }); }
    else onSeek(ts);
  };
  const onMove = (e: React.PointerEvent) => {
    if (!dragging.current) return;
    const ts = tsFromClientX(e.clientX);
    if (mode === "select" && anchor.current != null)
      onSelect({ a: Math.min(anchor.current, ts), b: Math.max(anchor.current, ts) });
    else if (mode === "scrub") onSeek(ts);
  };
  const onUp = () => { dragging.current = false; anchor.current = null; };

  const bands: { a: number; b: number }[] = [];
  for (const s of [...segments].sort((x, y) => x.start - y.start)) {
    const a = s.start, b = s.start + s.duration;
    const last = bands[bands.length - 1];
    if (last && a - last.b < 30) last.b = Math.max(last.b, b);
    else bands.push({ a, b });
  }

  return (
    <div className="border-t border-line bg-panel">
      <div className="flex items-center justify-between px-3 pt-2 text-[11px] text-zinc-500">
        <span>{clock(start)}</span>
        <button
          onClick={onLive}
          className={`flex items-center gap-1.5 rounded-full px-3 py-1 text-[11px] font-medium ${
            live ? "bg-red-500/15 text-red-400" : "bg-zinc-700 text-zinc-200"}`}>
          <span className={`h-2 w-2 rounded-full ${live ? "bg-red-500 animate-pulse" : "bg-zinc-400"}`} />
          {live ? "LIVE" : "Back to live"}
        </button>
        <span>{live ? "now" : clock(playheadTs)}</span>
      </div>

      <div
        ref={barRef}
        onPointerDown={onDown}
        onPointerMove={onMove}
        onPointerUp={onUp}
        className={`relative mx-3 my-2 h-14 touch-none select-none rounded-lg bg-black/40 ${
          mode === "select" ? "cursor-crosshair ring-1 ring-accent/40" : "cursor-pointer"}`}>
        {bands.map((b, i) => (
          <div key={i} className="absolute inset-y-0 bg-white/[0.04]"
            style={{ left: pos(b.a), width: `calc(${pos(b.b)} - ${pos(b.a)})` }} />
        ))}
        {/* selection band */}
        {selection && (
          <div className="absolute inset-y-0 bg-accent/25 ring-1 ring-accent"
            style={{ left: pos(selection.a), width: `calc(${pos(selection.b)} - ${pos(selection.a)})` }} />
        )}
        <div className="absolute inset-0 flex items-end">
          {Array.from({ length: nb }).map((_, i) => {
            const h = heights[i];
            return (
              <div key={i} className="flex flex-1 items-end px-[0.5px]" style={{ height: "100%" }}>
                {h > 0 && (
                  <div className="w-full rounded-t-sm bg-accent/80"
                    style={{ height: `${Math.max(6, (h / maxH) * 100)}%` }} />
                )}
              </div>
            );
          })}
        </div>
        {!live && (
          <div className="absolute inset-y-0 w-0.5 bg-white" style={{ left: pos(playheadTs) }}>
            <div className="absolute -top-1 left-1/2 h-2.5 w-2.5 -translate-x-1/2 rounded-full bg-white" />
          </div>
        )}
      </div>
      <div className="flex items-center gap-3 px-3 pb-2 text-[10px] text-zinc-500">
        <span className="flex items-center gap-1"><i className="h-2 w-2 rounded-sm bg-accent/80" />movement intensity</span>
        <span className="ml-auto">{mode === "select" ? "drag to select a clip" : "drag to scrub"}</span>
      </div>
    </div>
  );
}
