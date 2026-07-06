import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  getArtifact, getGroups, getSimilar, withToken, tagArtifacts,
  getTagsAutocomplete, setMemberStatus, type Artifact, type ScoredArtifact, type Group,
} from "../api";
import { cat, CAT_COLOR, clock, IconBack, IconSearch } from "../lib";

const clamp = (v: number) => Math.max(0, Math.min(1, v));

export default function ArtifactDetail() {
  const { id } = useParams();
  const nav = useNavigate();
  const [a, setA] = useState<Artifact | null>(null);
  const [err, setErr] = useState("");
  const [similar, setSimilar] = useState<ScoredArtifact[] | null>(null);
  const [groups, setGroups] = useState<Group[]>([]);
  const [removingTag, setRemovingTag] = useState<string | null>(null);

  const reload = () => {
    getArtifact(Number(id)).then(setA).catch((e) => setErr(e.message || "error"));
    getGroups().then(setGroups).catch(console.error);
  };

  useEffect(() => {
    setSimilar(null);
    getArtifact(Number(id)).then(setA).catch((e) => setErr(e.message || "error"));
    getGroups().then(setGroups).catch(console.error);
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
            {a.labels.map((l, i) => {
              const matchingGroup = groups.find(
                (g) => g.kind === "labeled" && g.tag_key?.toLowerCase() === l.type.toLowerCase() && g.tag_value?.toLowerCase() === l.value.toLowerCase()
              );
              return (
                <span key={i} className="flex items-center gap-1.5 rounded-md bg-accent/10 px-2.5 py-1 text-xs text-accent">
                  {l.type}: {l.value}
                  {matchingGroup && (
                    <button
                      type="button"
                      disabled={removingTag === `${l.type}:${l.value}`}
                      onClick={async () => {
                        const tagStr = `${l.type}:${l.value}`;
                        setRemovingTag(tagStr);
                        try {
                          await setMemberStatus(matchingGroup.id, a.id, "removed");
                          reload();
                        } catch (err) {
                          console.error(err);
                        } finally {
                          setRemovingTag(null);
                        }
                      }}
                      className="hover:text-red-400 font-medium ml-0.5 text-zinc-400 disabled:opacity-50"
                      title="Remove tag"
                    >
                      ×
                    </button>
                  )}
                </span>
              );
            })}
          </div>
        )}

        {/* Tag Manager */}
        <ArtifactTagger key={a.id} a={a} onReload={reload} />

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

        {/* find similar (embedding neighbours) */}
        <div className="mt-5">
          {!similar ? (
            <button
              onClick={() => getSimilar(a.id).then(setSimilar).catch(() => setSimilar([]))}
              className="flex items-center gap-2 rounded-xl border border-line bg-panel px-4 py-2.5 text-sm text-accent">
              <IconSearch className="h-4 w-4" /> Find similar
            </button>
          ) : similar.length === 0 ? (
            <p className="text-sm text-zinc-600">No similar artifacts found.</p>
          ) : (
            <>
              <h2 className="mb-2 text-[13px] font-medium text-zinc-400">Most similar</h2>
              <div className="grid grid-cols-3 gap-1.5 sm:grid-cols-4">
                {similar.map((s) => (
                  <Link key={s.id} to={`/artifacts/${s.id}`} className="relative">
                    <img src={withToken(s.images[0]?.url)}
                      className="aspect-square w-full rounded-md object-cover" />
                    <span className="absolute bottom-0.5 right-0.5 rounded bg-black/70 px-1 text-[9px] text-accent">
                      {s.score?.toFixed(2)}
                    </span>
                  </Link>
                ))}
              </div>
            </>
          )}
        </div>
      </main>
    </div>
  );
}

function ArtifactTagger({ a, onReload }: { a: Artifact; onReload: () => void }) {
  const [tagKey, setTagKey] = useState("");
  const [tagValue, setTagValue] = useState("");
  const [autocompleteData, setAutocompleteData] = useState<{ keys: string[]; values: Record<string, string[]> }>({ keys: [], values: {} });
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState("");

  async function loadAutocomplete() {
    try {
      const data = await getTagsAutocomplete();
      setAutocompleteData(data);
    } catch (e) {
      console.error(e);
    }
  }

  useEffect(() => {
    loadAutocomplete();
  }, []);

  async function handleAddTag(e: React.FormEvent) {
    e.preventDefault();
    const k = tagKey.trim();
    const v = tagValue.trim();
    if (!k || !v) return;
    setBusy(true);
    setMsg("");
    try {
      await tagArtifacts([a.id], [{ key: k, value: v }]);
      setTagValue("");
      setMsg("Tag added!");
      onReload();
      loadAutocomplete();
    } catch (err: any) {
      setMsg(err?.message || "Failed to add tag");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mb-4 rounded-xl border border-line bg-panel p-3">
      <div className="text-[13px] font-semibold text-zinc-200 mb-2">🏷️ Add Tag</div>
      <form onSubmit={handleAddTag} className="flex gap-2">
        <div className="flex-1">
          <input
            type="text"
            required
            value={tagKey}
            onChange={(e) => setTagKey(e.target.value)}
            placeholder="Tag Key (e.g. clothing)"
            list="artifact-tag-keys"
            className="w-full rounded-lg border border-line bg-panel2 px-3 py-1.5 text-xs text-zinc-100 focus:border-accent focus:outline-none"
          />
          <datalist id="artifact-tag-keys">
            {(autocompleteData.keys || []).map((k) => (
              <option key={k} value={k} />
            ))}
          </datalist>
        </div>

        <div className="flex-1">
          <input
            type="text"
            required
            value={tagValue}
            onChange={(e) => setTagValue(e.target.value)}
            placeholder="Tag Value (e.g. dress)"
            list="artifact-tag-values"
            className="w-full rounded-lg border border-line bg-panel2 px-3 py-1.5 text-xs text-zinc-100 focus:border-accent focus:outline-none"
          />
          <datalist id="artifact-tag-values">
            {(autocompleteData.values?.[tagKey] || []).map((v) => (
              <option key={v} value={v} />
            ))}
          </datalist>
        </div>

        <button
          type="submit"
          disabled={busy || !tagKey.trim() || !tagValue.trim()}
          className="rounded-lg bg-accent px-4 text-xs font-semibold text-black disabled:opacity-40"
        >
          {busy ? "Saving..." : "Add"}
        </button>
      </form>
      {msg && <p className="mt-1.5 text-[11px] text-accent font-medium">{msg}</p>}
    </div>
  );
}

function Center({ children }: { children: React.ReactNode }) {
  return <div className="flex h-full items-center justify-center text-sm text-zinc-600">{children}</div>;
}
