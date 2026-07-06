import { useEffect, useRef, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  dvrPlayUrl, getDvrIndex, getDvrTimeline, getStats, saveClip, streamUrl,
  type DvrIndex, type DvrTimeline,
} from "../api";
import { useAuth } from "../auth";
import { Shell } from "../components/Shell";
import { Timeline } from "../components/Timeline";
import { usePoll } from "../lib";

export default function Live() {
  const nav = useNavigate();
  const { logout } = useAuth();
  const onAuth = () => { logout(); nav("/"); };
  const { data } = usePoll(getStats, 2000, onAuth);
  const { data: dvr } = usePoll<[DvrIndex, DvrTimeline]>(
    () => Promise.all([getDvrIndex(), getDvrTimeline()]), 8000, onAuth);

  const [overlay, setOverlay] = useState(true);
  const [showLog, setShowLog] = useState(true);
  const [mode, setMode] = useState<"live" | "play">("live");
  const [playStart, setPlayStart] = useState(0);   // wall-clock the stream starts at
  const [playhead, setPlayhead] = useState(0);
  const [selMode, setSelMode] = useState(false);
  const [selection, setSelection] = useState<{ a: number; b: number } | null>(null);
  const [toast, setToast] = useState("");
  const videoRef = useRef<HTMLVideoElement>(null);
  const seekTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  const segments = dvr?.[0]?.segments ?? [];
  const timeline = dvr?.[1] ?? null;

  // Scrubbing fires continuously; debounce the actual stream (re)load so we
  // don't spawn an ffmpeg per pixel. The playhead moves instantly regardless.
  function seek(ts: number) {
    setPlayhead(ts);
    setMode("play");
    if (seekTimer.current) clearTimeout(seekTimer.current);
    seekTimer.current = setTimeout(() => setPlayStart(ts), 250);
  }
  function goLive() {
    if (seekTimer.current) clearTimeout(seekTimer.current);
    setMode("live");
    setPlayStart(0);
  }
  useEffect(() => () => { if (seekTimer.current) clearTimeout(seekTimer.current); }, []);

  async function doSave() {
    if (!selection) return;
    const secs = Math.round(selection.b - selection.a);
    const name = window.prompt(`Name this ${secs}s clip for your library`,
      `clip ${new Date(selection.a * 1000).toLocaleTimeString()}`);
    if (!name) return;
    try {
      await saveClip(selection.a, selection.b, name);
      setToast("Saved to library ✓");
      setSelection(null); setSelMode(false);
    } catch (e: any) {
      setToast(e?.message || "save failed");
    }
    setTimeout(() => setToast(""), 3000);
  }

  const stale = data?.stale_pct ?? 0;
  const chips = [
    { label: "Active now", value: data?.active ?? "—" },
    { label: "People today", value: data ? data.daily.person ?? 0 : "—" },
    { label: "Vehicles today", value: data ? data.daily.vehicle ?? 0 : "—" },
  ];

  return (
    <Shell title="Live" online={data?.online} flush>
      <div className="flex h-full flex-col">
        <div className="relative min-h-0 flex-1 bg-black">
          {mode === "live" ? (
            <img
              key={overlay ? "ov" : "clean"}
              src={streamUrl(overlay)}
              alt="live"
              className="h-full w-full object-contain"
              onError={(e) => {
                const el = e.currentTarget;
                setTimeout(() => (el.src = streamUrl(overlay)), 1500);
              }}
            />
          ) : (
            <video
              ref={videoRef}
              key={playStart}
              src={playStart ? dvrPlayUrl(playStart) : undefined}
              autoPlay
              playsInline
              className="h-full w-full object-contain"
              onTimeUpdate={() => {
                if (videoRef.current && playStart)
                  setPlayhead(playStart + videoRef.current.currentTime);
              }}
              onEnded={goLive}
            />
          )}

          {showLog && (
            <div className="pointer-events-none absolute left-3 top-3 flex flex-wrap gap-2">
              {chips.map((c) => (
                <div key={c.label} className="rounded-lg bg-black/60 px-3 py-1.5 backdrop-blur">
                  <div className="text-[10px] uppercase tracking-wide text-zinc-400">{c.label}</div>
                  <div className="text-sm font-semibold text-accent">{c.value}</div>
                </div>
              ))}
            </div>
          )}

          <div className="absolute right-3 top-3 flex gap-1.5">
            {mode === "play" ? (
              <span className="rounded-lg bg-amber-500/80 px-2.5 py-1 text-[11px] font-medium text-black">
                ⏵ replay
              </span>
            ) : (
              <>
                <Toggle on={overlay} onClick={() => setOverlay((v) => !v)}>Boxes</Toggle>
                <Toggle on={showLog} onClick={() => setShowLog((v) => !v)}>Log</Toggle>
              </>
            )}
          </div>

          {mode === "live" && data && (
            <div className="absolute right-3 bottom-3 rounded-lg bg-black/60 px-2.5 py-1 text-[10px] backdrop-blur">
              <span className="text-zinc-400">source </span>
              <span className={stale > 35 ? "text-amber-400" : "text-zinc-200"}>
                {data.source_fps ?? "—"} fps
              </span>
              {stale > 25 && <span className="text-amber-400"> · {stale}% dropped</span>}
              {!!data.idle_objects && (
                <span className="text-zinc-500"> · {data.idle_objects} idle hidden</span>
              )}
            </div>
          )}

          {showLog && mode === "live" && data?.events?.length ? (
            <div className="pointer-events-none absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 to-transparent p-3 pb-10">
              <div className="max-h-20 space-y-0.5 overflow-hidden text-[11px] text-zinc-300">
                {data.events.slice(0, 3).map((e, i) => (
                  <div key={i} className="truncate">{e}</div>
                ))}
              </div>
            </div>
          ) : null}

          {mode === "live" && data && !data.online && (
            <div className="absolute inset-0 flex items-center justify-center text-sm text-zinc-500">
              connecting to camera…
            </div>
          )}
          {toast && (
            <div className="absolute left-1/2 top-3 -translate-x-1/2 rounded-lg bg-black/80 px-3 py-1.5 text-[12px] text-accent backdrop-blur">
              {toast}
            </div>
          )}
        </div>

        {/* clip / library toolbar */}
        <div className="flex items-center gap-2 border-t border-line bg-panel px-3 py-1.5 text-[12px]">
          <button
            onClick={() => { setSelMode((v) => !v); setSelection(null); }}
            className={`rounded-lg px-2.5 py-1 font-medium ${
              selMode ? "bg-accent text-black" : "bg-panel2 text-zinc-300"}`}>
            ✂ Clip
          </button>
          {selMode && selection && (
            <button onClick={doSave} className="rounded-lg bg-accent/20 px-2.5 py-1 text-accent">
              Save {Math.round(selection.b - selection.a)}s
            </button>
          )}
          {selMode && <span className="text-[11px] text-zinc-500">drag the bar to select</span>}
          <Link to="/library" className="ml-auto rounded-lg bg-panel2 px-2.5 py-1 text-zinc-300">
            📚 Library
          </Link>
        </div>

        <Timeline
          timeline={timeline}
          segments={segments}
          playheadTs={mode === "live" ? (timeline?.end ?? 0) : playhead}
          live={mode === "live"}
          mode={selMode ? "select" : "scrub"}
          selection={selection}
          onSeek={seek}
          onSelect={setSelection}
          onLive={goLive}
        />
      </div>
    </Shell>
  );
}

function Toggle({ on, onClick, children }: {
  on: boolean; onClick: () => void; children: React.ReactNode;
}) {
  return (
    <button onClick={onClick}
      className={`rounded-lg px-2.5 py-1 text-[11px] font-medium backdrop-blur ${
        on ? "bg-accent/80 text-black" : "bg-black/60 text-zinc-400"}`}>
      {children}
    </button>
  );
}
