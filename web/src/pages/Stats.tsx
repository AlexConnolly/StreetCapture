import { useNavigate } from "react-router-dom";
import { getHourly, getStats } from "../api";
import { useAuth } from "../auth";
import { Shell } from "../components/Shell";
import { usePoll } from "../lib";

export default function Stats() {
  const nav = useNavigate();
  const { logout } = useAuth();
  const onAuth = () => { logout(); nav("/"); };
  const { data } = usePoll(getStats, 3000, onAuth);
  const people = usePoll(() => getHourly("person"), 30000, onAuth).data;
  const cars = usePoll(() => getHourly("car"), 30000, onAuth).data;

  const artToday = data ? sum(data.artifacts) : 0;
  const seenToday = data ? sum(data.daily) : 0;

  const tiles = [
    { k: "FPS", v: data ? data.fps.toFixed(1) : "—", accent: true },
    { k: "Active now", v: data?.active ?? "—" },
    { k: "Artifacts today", v: artToday },
    { k: "Seen today", v: seenToday },
    { k: "Vectors", v: data?.faiss_vectors ?? "—" },
    { k: "Uptime", v: data ? fmtUptime(data.uptime_s) : "—" },
  ];

  return (
    <Shell title="Stats" online={data?.online}>
      <div className="grid grid-cols-3 gap-2.5">
        {tiles.map((t) => (
          <div key={t.k} className="rounded-xl border border-line bg-panel p-3">
            <div className={`text-xl font-semibold ${t.accent ? "text-accent" : ""}`}>{t.v}</div>
            <div className="mt-0.5 text-[11px] text-zinc-500">{t.k}</div>
          </div>
        ))}
      </div>

      <Section title="People by hour (today)"><Bars buckets={people?.buckets} color="#34d399" /></Section>
      <Section title="Cars by hour (today)"><Bars buckets={cars?.buckets} color="#fbbf24" /></Section>

      {data && (
        <div className="mt-5 rounded-xl border border-line bg-panel p-3.5 text-[12px] text-zinc-500">
          <div>Embedding model: <span className="text-zinc-300">{data.embed_model}</span></div>
          <div className="mt-1">Day: <span className="text-zinc-300">{data.day}</span></div>
        </div>
      )}
    </Shell>
  );
}

function Section({ title, children }: { title: string; children: React.ReactNode }) {
  return (
    <div className="mt-5">
      <h2 className="mb-2 text-[13px] font-medium text-zinc-400">{title}</h2>
      {children}
    </div>
  );
}

function Bars({ buckets, color }: { buckets?: number[]; color: string }) {
  const b = buckets || new Array(24).fill(0);
  const max = Math.max(1, ...b);
  return (
    <div className="flex h-28 items-end gap-[3px] rounded-xl border border-line bg-panel p-3">
      {b.map((v, h) => (
        <div key={h} className="flex flex-1 flex-col items-center gap-1">
          <div className="w-full rounded-sm" style={{
            height: `${(v / max) * 100}%`,
            minHeight: v > 0 ? 3 : 0,
            background: color,
          }} />
          {h % 6 === 0 && <span className="text-[8px] text-zinc-600">{h}</span>}
        </div>
      ))}
    </div>
  );
}

function sum(o: Record<string, number>) {
  return Object.values(o).reduce((a, b) => a + b, 0);
}
function fmtUptime(s: number) {
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  return `${Math.floor(s / 3600)}h${Math.floor((s % 3600) / 60)}m`;
}
