import { useNavigate } from "react-router-dom";
import { getStats, getTimeseries, type Busiest, type Timeseries } from "../api";
import { useAuth } from "../auth";
import { Shell } from "../components/Shell";
import { usePoll } from "../lib";

const EMERALD = "#34d399";
const AMBER = "#fbbf24";

export default function Stats() {
  const nav = useNavigate();
  const { logout } = useAuth();
  const onAuth = () => { logout(); nav("/"); };
  const { data: ts } = usePoll(() => getTimeseries(15), 15000, onAuth);
  const { data: sys } = usePoll(getStats, 5000, onAuth);

  const totPeople = ts?.totals.person ?? 0;
  const totVeh = ts?.totals.vehicle ?? 0;

  return (
    <Shell title="Today" online={sys?.online}>
      {/* headline: what we've actually seen */}
      <div className="grid grid-cols-2 gap-2.5">
        <Kpi label="People" total={totPeople} busiest={ts?.busiest.person} color={EMERALD} />
        <Kpi label="Vehicles" total={totVeh} busiest={ts?.busiest.vehicle} color={AMBER} />
      </div>

      {/* activity over time, 15-min buckets */}
      <div className="mt-4 rounded-xl border border-line bg-panel p-3.5">
        <div className="mb-3 flex items-center justify-between">
          <h2 className="text-[13px] font-medium text-zinc-300">Activity today</h2>
          <div className="flex items-center gap-3 text-[11px] text-zinc-400">
            <Legend color={EMERALD} label="people" />
            <Legend color={AMBER} label="vehicles" />
            <span className="text-zinc-600">15-min</span>
          </div>
        </div>
        <StackedChart ts={ts} />
      </div>

      {/* muted system footer */}
      {sys && (
        <div className="mt-4 flex flex-wrap gap-x-4 gap-y-1 rounded-xl border border-line bg-panel px-3.5 py-3 text-[11px] text-zinc-600">
          <span>{sys.online ? "● live" : "○ offline"} {sys.fps.toFixed(1)} fps</span>
          <span>active {sys.active}</span>
          <span>uptime {fmtUptime(sys.uptime_s)}</span>
          <span>vectors {sys.faiss_vectors}</span>
          <span className="truncate">{sys.embed_model}</span>
        </div>
      )}
    </Shell>
  );
}

function Kpi({ label, total, busiest, color }: {
  label: string; total: number; busiest?: Busiest | null; color: string;
}) {
  return (
    <div className="rounded-xl border border-line bg-panel p-3.5">
      <div className="flex items-center gap-2">
        <span className="h-2.5 w-2.5 rounded-full" style={{ background: color }} />
        <span className="text-[13px] text-zinc-400">{label}</span>
      </div>
      <div className="mt-1 text-4xl font-semibold tabular-nums">{total}</div>
      <div className="mt-1 text-[11px] text-zinc-500">
        {busiest ? `busiest ${busiest.label} · ${busiest.count}` : "no activity yet"}
      </div>
    </div>
  );
}

function Legend({ color, label }: { color: string; label: string }) {
  return (
    <span className="flex items-center gap-1">
      <span className="h-2 w-2 rounded-sm" style={{ background: color }} />
      {label}
    </span>
  );
}

function StackedChart({ ts }: { ts: Timeseries | null }) {
  const n = ts ? ts.labels.length : 96;
  const person = ts?.person ?? new Array(n).fill(0);
  const vehicle = ts?.vehicle ?? new Array(n).fill(0);
  const max = Math.max(1, ...person.map((p, i) => p + vehicle[i]));
  const ticks = [0, 0.25, 0.5, 0.75].map((f) => Math.floor(f * n));

  return (
    <div>
      <div className="flex h-36 items-end gap-px">
        {person.map((p, i) => {
          const v = vehicle[i];
          return (
            <div key={i} className="flex h-full flex-1 flex-col-reverse" title={`${ts?.labels[i] ?? ""} · ${p}p ${v}v`}>
              <div style={{ height: `${(v / max) * 100}%`, background: AMBER }} />
              <div style={{ height: `${(p / max) * 100}%`, background: EMERALD }} />
            </div>
          );
        })}
      </div>
      <div className="mt-1 flex justify-between text-[9px] text-zinc-600">
        {ticks.map((t) => <span key={t}>{ts?.labels[t] ?? ""}</span>)}
        <span>now</span>
      </div>
    </div>
  );
}

function fmtUptime(s: number) {
  if (s < 60) return `${s}s`;
  if (s < 3600) return `${Math.floor(s / 60)}m`;
  return `${Math.floor(s / 3600)}h${Math.floor((s % 3600) / 60)}m`;
}
