import { useNavigate } from "react-router-dom";
import { getStats, streamUrl } from "../api";
import { useAuth } from "../auth";
import { Shell } from "../components/Shell";
import { usePoll } from "../lib";

export default function Live() {
  const nav = useNavigate();
  const { logout } = useAuth();
  const onAuth = () => { logout(); nav("/"); };
  const { data } = usePoll(getStats, 2000, onAuth);

  const chips = [
    { label: "FPS", value: data ? data.fps.toFixed(1) : "—" },
    { label: "Active", value: data?.active ?? "—" },
    { label: "Artifacts today", value: data ? sum(data.artifacts) : "—" },
  ];

  return (
    <Shell title="Live" online={data?.online} flush>
      <div className="relative h-full w-full bg-black">
        <img
          src={streamUrl()}
          alt="live"
          className="h-full w-full object-contain"
          onError={(e) => {
            // MJPEG hiccup — force a reconnect after a beat.
            const el = e.currentTarget;
            setTimeout(() => (el.src = streamUrl()), 1500);
          }}
        />
        {/* stat chips */}
        <div className="pointer-events-none absolute left-3 top-3 flex flex-wrap gap-2">
          {chips.map((c) => (
            <div key={c.label} className="rounded-lg bg-black/60 px-3 py-1.5 backdrop-blur">
              <div className="text-[10px] uppercase tracking-wide text-zinc-400">{c.label}</div>
              <div className="text-sm font-semibold text-accent">{c.value}</div>
            </div>
          ))}
        </div>
        {/* recent events ticker */}
        {data?.events?.length ? (
          <div className="absolute inset-x-0 bottom-0 bg-gradient-to-t from-black/80 to-transparent p-3">
            <div className="max-h-24 space-y-0.5 overflow-hidden text-[11px] text-zinc-300">
              {data.events.slice(0, 4).map((e, i) => (
                <div key={i} className="truncate">{e}</div>
              ))}
            </div>
          </div>
        ) : null}
        {data && !data.online && (
          <div className="absolute inset-0 flex items-center justify-center text-sm text-zinc-500">
            connecting to camera…
          </div>
        )}
      </div>
    </Shell>
  );
}

function sum(o: Record<string, number>) {
  return Object.values(o).reduce((a, b) => a + b, 0);
}
