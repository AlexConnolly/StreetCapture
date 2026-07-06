import { useEffect, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  getEntities, getEntityMembers, getEntityTimeline, nameEntity, splitEntityMember,
  withToken, type Artifact, type Entity, type Sighting,
} from "../api";
import { cat, CAT_COLOR, clock, IconBack } from "../lib";

export default function EntityDetail() {
  const { id } = useParams();
  const eid = Number(id);
  const nav = useNavigate();
  const [entity, setEntity] = useState<Entity | null>(null);
  const [members, setMembers] = useState<Artifact[] | null>(null);
  const [sightings, setSightings] = useState<Sighting[] | null>(null);

  async function load() {
    const [es, ms, ss] = await Promise.all([
      getEntities(), getEntityMembers(eid), getEntityTimeline(eid),
    ]);
    setEntity(es.find((e) => e.id === eid) || null);
    setMembers(ms);
    setSightings(ss);
  }
  useEffect(() => { load(); }, [eid]);

  async function split(aid: number) {
    setMembers((ms) => ms?.filter((m) => m.id !== aid) ?? ms);   // optimistic
    try { await splitEntityMember(aid); } finally { load(); }
  }

  async function rename() {
    const name = window.prompt("Name this entity (e.g. 'the white DPD van')", entity?.label || "");
    if (!name) return;
    await nameEntity(eid, name);
    load();
  }

  const title = entity?.label || entity?.class || "Entity";
  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center gap-3 border-b border-line bg-panel px-4 py-3"
        style={{ paddingTop: "max(0.75rem, env(safe-area-inset-top))" }}>
        <button onClick={() => nav(-1)} className="text-zinc-400"><IconBack className="h-5 w-5" /></button>
        <h1 className={`flex-1 truncate text-[15px] font-semibold capitalize ${entity ? CAT_COLOR[cat(entity.class)] : ""}`}>
          {title}
        </h1>
        <button onClick={rename} className="rounded-lg bg-accent px-3 py-1.5 text-sm font-medium text-black">
          Name
        </button>
      </header>

      <main className="flex-1 overflow-y-auto p-4">
        {entity && (
          <p className="mb-3 text-[13px] text-zinc-400">
            Seen <span className="text-zinc-200">{entity.occurrences}×</span>
            {entity.sightings && entity.sightings > entity.occurrences ?
              <span className="text-zinc-600"> ({entity.sightings} detections)</span> : null}
            {" · "}first {clock(entity.first_seen)} · last {clock(entity.last_seen)}
          </p>
        )}

        {sightings && sightings.length > 1 && <SightingBar sightings={sightings} />}

        <p className="mb-2 mt-4 text-[12px] text-zinc-500">
          The system thinks these are all the same thing.
          Tap <span className="text-red-400">✕</span> on any that <span className="text-zinc-300">aren't</span> — it
          splits out and won't merge back.
        </p>
        {!members ? (
          <p className="mt-6 text-center text-sm text-zinc-600">loading…</p>
        ) : (
          <div className="grid grid-cols-3 gap-1.5 sm:grid-cols-4">
            {members.map((a) => (
              <div key={a.id} className="relative">
                <Link to={`/artifacts/${a.id}`}>
                  <img src={withToken(a.images[0]?.url)}
                    className="aspect-square w-full rounded-md object-cover" />
                </Link>
                {members.length > 1 && (
                  <button
                    onClick={() => split(a.id)}
                    title="Not the same — split out"
                    className="absolute right-1 top-1 flex h-6 w-6 items-center justify-center rounded-full bg-black/70 text-[13px] text-red-400 backdrop-blur active:bg-red-500 active:text-black">
                    ✕
                  </button>
                )}
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}

function SightingBar({ sightings }: { sightings: Sighting[] }) {
  // `sightings` are already merged visits (one continuous presence = one bar).
  const lo = sightings[0].start;
  const hi = sightings[sightings.length - 1].end;
  const span = Math.max(1, hi - lo);
  const pos = (t: number) => Math.max(0, Math.min(100, ((t - lo) / span) * 100));
  return (
    <div className="rounded-xl border border-line bg-panel p-3">
      <div className="mb-2 text-[12px] font-medium text-zinc-300">Visit timeline</div>
      <div className="relative h-8 rounded-lg bg-black/40">
        {sightings.map((s) => {
          const left = pos(s.start);
          const width = Math.max(1.2, pos(s.end) - left);   // visible even for instants
          return (
            <Link key={s.artifact_id} to={`/artifacts/${s.artifact_id}`}
              title={`${new Date(s.start * 1000).toLocaleTimeString()} · ${s.class}`
                + (s.detections && s.detections > 1 ? ` · ${s.detections} detections` : "")}
              className="absolute top-1/2 h-3 -translate-y-1/2 rounded-full ring-1 ring-panel"
              style={{ left: `${left}%`, width: `${width}%`,
                background: s.class === "person" ? "#34d399" : "#fbbf24" }} />
          );
        })}
      </div>
      <div className="mt-1 flex justify-between text-[10px] text-zinc-600">
        <span>{clock(lo)}</span>
        <span>{sightings.length} visit{sightings.length === 1 ? "" : "s"}</span>
        <span>{clock(hi)}</span>
      </div>
    </div>
  );
}
