import { useNavigate } from "react-router-dom";
import { getEvents, type EventItem } from "../api";
import { useAuth } from "../auth";
import { Shell } from "../components/Shell";
import { clock, usePoll } from "../lib";

const STYLE: Record<string, { dot: string; label: string }> = {
  artifact_created: { dot: "bg-accent", label: "Artifact created" },
  artifact_rejected: { dot: "bg-zinc-600", label: "Rejected" },
  vehicle_passed: { dot: "bg-amber-400", label: "Vehicle passed" },
  object_stayed: { dot: "bg-violet-400", label: "Stayed" },
  object_entered: { dot: "bg-emerald-400", label: "Entered" },
  object_left: { dot: "bg-rose-400", label: "Left" },
  track_started: { dot: "bg-zinc-500", label: "Track started" },
  track_ended: { dot: "bg-zinc-500", label: "Track ended" },
};

function line(e: EventItem): string {
  const cls = e.class || "object";
  if (e.type === "artifact_created") return `${cls} · Artifact #${e.artifact_id}`;
  if (e.type === "artifact_rejected") return `${cls} · ${e.reason}`;
  if (e.duration != null) return `${cls} · ${e.duration.toFixed(0)}s`;
  return `${cls}${e.source_track_id != null ? ` · track #${e.source_track_id}` : ""}`;
}

export default function Events() {
  const nav = useNavigate();
  const { logout } = useAuth();
  const { data } = usePoll(getEvents, 3000, () => { logout(); nav("/"); });

  return (
    <Shell title="Events">
      {!data ? (
        <p className="mt-10 text-center text-sm text-zinc-600">loading…</p>
      ) : data.length === 0 ? (
        <p className="mt-10 text-center text-sm text-zinc-600">No events yet.</p>
      ) : (
        <div className="divide-y divide-line overflow-hidden rounded-xl border border-line bg-panel">
          {data.map((e, i) => {
            const s = STYLE[e.type] || { dot: "bg-zinc-500", label: e.type };
            return (
              <div key={i} className="flex items-center gap-3 px-3.5 py-2.5">
                <span className={`h-2 w-2 shrink-0 rounded-full ${s.dot}`} />
                <div className="min-w-0 flex-1">
                  <div className="text-sm">{s.label}</div>
                  <div className="truncate text-[11px] text-zinc-500">{line(e)}</div>
                </div>
                <span className="shrink-0 text-[11px] text-zinc-600">{clock(e.time)}</span>
              </div>
            );
          })}
        </div>
      )}
    </Shell>
  );
}
