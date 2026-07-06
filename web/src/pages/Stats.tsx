import { Link, useNavigate } from "react-router-dom";
import {
  getRecurring, getStats, getStatsSummary, getTimeseries, withToken,
  type RecurringEntity, type StatsSummary, type Timeseries,
} from "../api";
import { useAuth } from "../auth";
import { Shell } from "../components/Shell";
import { cat, CAT_COLOR, timeAgo, usePoll } from "../lib";

const EMERALD = "#34d399";
const AMBER = "#fbbf24";
const ZINC = "#a1a1aa";
const catColor = (c: string) => (c === "person" ? EMERALD : c === "vehicle" ? AMBER : ZINC);

export default function Stats() {
  const nav = useNavigate();
  const { logout } = useAuth();
  const onAuth = () => { logout(); nav("/"); };
  const { data: ts } = usePoll(() => getTimeseries(15), 15000, onAuth);
  const { data: sys } = usePoll(getStats, 5000, onAuth);
  const { data: sum } = usePoll(getStatsSummary, 10000, onAuth);
  const { data: recurring } = usePoll(getRecurring, 15000, onAuth);

  return (
    <Shell title="Today" online={sys?.online}>
      {/* unique things vs total sightings — the "why is it not merged" answer */}
      <div className="grid grid-cols-2 gap-2.5">
        <UniqueKpi label="People" color={EMERALD}
          unique={sum?.unique.person} sightings={sum?.cat_sightings.person} />
        <UniqueKpi label="Vehicles" color={AMBER}
          unique={sum?.unique.vehicle} sightings={sum?.cat_sightings.vehicle} />
      </div>

      {/* object mix — the breakdown of what's out there today */}
      <div className="mt-4 rounded-xl border border-line bg-panel p-3.5">
        <h2 className="mb-3 text-[13px] font-medium text-zinc-300">Object mix today</h2>
        {!sum ? (
          <p className="text-[12px] text-zinc-600">loading…</p>
        ) : sum.mix.length === 0 ? (
          <p className="text-[12px] text-zinc-500">Nothing seen yet today.</p>
        ) : (
          <div className="space-y-1.5">
            {sum.mix.slice(0, 10).map((m) => {
              const max = sum.mix[0].count || 1;
              return (
                <div key={m.class} className="flex items-center gap-2">
                  <span className="w-20 shrink-0 truncate text-[12px] capitalize text-zinc-400">{m.class}</span>
                  <div className="h-4 flex-1 overflow-hidden rounded bg-panel2">
                    <div className="h-full rounded" style={{
                      width: `${Math.max(4, (m.count / max) * 100)}%`,
                      background: catColor(cat(m.class)),
                    }} />
                  </div>
                  <span className="w-10 shrink-0 text-right text-[12px] tabular-nums text-zinc-300">{m.count}</span>
                </div>
              );
            })}
          </div>
        )}
      </div>

      {/* regulars — only NAMED things, so it reads as a dashboard not a list of
          anonymous 'person' rows (those are aggregated above) */}
      <div className="mt-4">
        <div className="mb-2 flex items-baseline justify-between">
          <h2 className="text-[13px] font-medium text-zinc-300">Your regulars</h2>
          <span className="text-[10px] text-zinc-600">named things, most-seen today</span>
        </div>
        {!recurring ? (
          <p className="text-[12px] text-zinc-600">loading…</p>
        ) : recurring.length === 0 ? (
          <p className="rounded-xl border border-line bg-panel px-3.5 py-4 text-[12px] text-zinc-500">
            No named regulars yet. Open any entity or teach a label (draw a box on an artifact),
            give it a name, and it'll show here with how often it turned up today.
          </p>
        ) : (
          <div className="space-y-2">
            {recurring.map((r) => <RecurringRow key={r.entity_id} r={r} />)}
          </div>
        )}
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

function UniqueKpi({ label, color, unique, sightings }: {
  label: string; color: string; unique?: number; sightings?: number;
}) {
  return (
    <div className="rounded-xl border border-line bg-panel p-3.5">
      <div className="flex items-center gap-2">
        <span className="h-2.5 w-2.5 rounded-full" style={{ background: color }} />
        <span className="text-[13px] text-zinc-400">{label}</span>
      </div>
      <div className="mt-1 flex items-baseline gap-1.5">
        <span className="text-4xl font-semibold tabular-nums">{unique ?? "—"}</span>
        <span className="text-[12px] text-zinc-500">unique</span>
      </div>
      <div className="mt-1 text-[11px] text-zinc-500">
        {sightings ?? 0} total sightings today
      </div>
    </div>
  );
}

function RecurringRow({ r }: { r: RecurringEntity }) {
  return (
    <Link to={`/entities/${r.entity_id}`}
      className="flex items-center gap-3 rounded-xl border border-line bg-panel p-2.5">
      <div className="flex gap-1">
        {r.samples.slice(0, 3).map((u, i) => (
          <img key={i} src={withToken(u)} className="h-12 w-12 rounded-md object-cover" />
        ))}
      </div>
      <div className="min-w-0 flex-1">
        <div className={`truncate text-sm font-medium capitalize ${CAT_COLOR[cat(r.class)]}`}>
          {r.label || r.class}
        </div>
        <div className="text-[11px] text-zinc-500">last {timeAgo(r.last)}</div>
      </div>
      <div className="text-right">
        <div className="text-lg font-semibold tabular-nums text-accent">{r.count}×</div>
        <div className="text-[10px] text-zinc-600">seen</div>
      </div>
    </Link>
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
