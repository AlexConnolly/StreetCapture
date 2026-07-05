import { useEffect, useState } from "react";
import { useNavigate, useParams } from "react-router-dom";
import { getArtifact, withToken, type Artifact } from "../api";
import { cat, CAT_COLOR, clock, IconBack } from "../lib";

export default function ArtifactDetail() {
  const { id } = useParams();
  const nav = useNavigate();
  const [a, setA] = useState<Artifact | null>(null);
  const [err, setErr] = useState("");

  useEffect(() => {
    getArtifact(Number(id)).then(setA).catch((e) => setErr(e.message || "error"));
  }, [id]);

  if (err) return <Center>{err}</Center>;
  if (!a) return <Center>loading…</Center>;

  const stat = (k: string, v: string | number) => (
    <div className="rounded-lg bg-panel2 px-3 py-2">
      <div className="text-[10px] uppercase tracking-wide text-zinc-500">{k}</div>
      <div className="text-sm font-medium">{v}</div>
    </div>
  );

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center gap-3 border-b border-line bg-panel px-4 py-3"
        style={{ paddingTop: "max(0.75rem, env(safe-area-inset-top))" }}>
        <button onClick={() => nav(-1)} className="text-zinc-400"><IconBack className="h-5 w-5" /></button>
        <h1 className="text-[15px] font-semibold">
          <span className={`capitalize ${CAT_COLOR[cat(a.class)]}`}>{a.class}</span>
          <span className="text-zinc-500"> · Artifact #{a.id}</span>
        </h1>
      </header>

      <main className="flex-1 overflow-y-auto p-4">
        {/* keyframes */}
        <div className="-mx-4 mb-4 flex gap-2 overflow-x-auto px-4 no-scrollbar">
          {a.images.map((im) => (
            <img key={im.rank} src={withToken(im.url)}
              className="h-56 rounded-xl border border-line object-cover" />
          ))}
        </div>

        {/* labels */}
        {a.labels.length > 0 && (
          <div className="mb-4 flex flex-wrap gap-2">
            {a.labels.map((l, i) => (
              <span key={i} className="rounded-md bg-accent/10 px-2.5 py-1 text-xs text-accent">
                {l.type}: {l.value}
              </span>
            ))}
          </div>
        )}

        {/* stats grid */}
        <div className="grid grid-cols-2 gap-2">
          {stat("Seen at", clock(a.start))}
          {stat("Duration", `${a.duration.toFixed(1)}s`)}
          {stat("Confidence", a.confidence.toFixed(2))}
          {stat("Sharpness", a.sharpness.toFixed(0))}
          {stat("Visibility", a.visibility.toFixed(2))}
          {stat("Motion", `${a.motion.toFixed(0)}px`)}
          {stat("Frames", a.frames)}
          {stat("Track ID", `#${a.track_id}`)}
          {stat("Entity", a.entity_id ?? "— (v2)")}
          {stat("Embedding", a.embedding ? `${a.embedding.dim}d` : "none")}
        </div>
        {a.embedding && (
          <p className="mt-3 text-[11px] text-zinc-600">{a.embedding.model_version}</p>
        )}
      </main>
    </div>
  );
}

function Center({ children }: { children: React.ReactNode }) {
  return <div className="flex h-full items-center justify-center text-sm text-zinc-600">{children}</div>;
}
