import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  createGroupFromText, getEntities, getGroups, notifyStatus, notifyTest,
  recluster, searchArtifacts, withToken, type Group, type ScoredArtifact,
} from "../api";
import { useAuth } from "../auth";
import { Shell } from "../components/Shell";
import { cat, CAT_COLOR, IconBell, IconSearch, timeAgo, usePoll } from "../lib";

export default function Groups() {
  const nav = useNavigate();
  const { logout } = useAuth();
  const onAuth = () => { logout(); nav("/"); };
  const [tab, setTab] = useState<"clusters" | "entities">("clusters");

  return (
    <Shell title="Discover">
      <div className="mb-3 flex rounded-xl border border-line bg-panel p-1 text-sm">
        {(["clusters", "entities"] as const).map((t) => (
          <button key={t} onClick={() => setTab(t)}
            className={`flex-1 rounded-lg py-1.5 capitalize ${
              tab === t ? "bg-accent text-black font-medium" : "text-zinc-400"}`}>
            {t}
          </button>
        ))}
      </div>
      {tab === "clusters" ? <GroupsView onAuth={onAuth} /> : <EntitiesView onAuth={onAuth} />}
    </Shell>
  );
}

function Thumbs({ urls, groupId }: { urls: string[]; groupId?: number }) {
  return (
    <div className="flex gap-1 overflow-hidden">
      {urls.slice(0, 4).map((u, i) => (
        <img key={i} src={withToken(groupId ? `${u}?group_id=${groupId}` : u)} className="h-16 w-16 rounded-md object-cover" />
      ))}
    </div>
  );
}

function GroupCard({ g }: { g: Group }) {
  const pending = g.pending ?? 0;
  const thumbs = g.samples.length ? g.samples.slice(0, 3) : [""];
  return (
    <Link to={`/groups/${g.id}`}
      className={`flex items-stretch gap-3 rounded-2xl border bg-panel p-2.5 transition active:scale-[0.99] ${
        pending > 0 ? "border-accent/40 shadow-[0_0_0_1px_rgba(34,211,238,0.15)]" : "border-line"}`}>
      <div className="relative flex shrink-0 gap-1">
        {thumbs.map((u, i) => (
          u ? <img key={i} src={withToken(`${u}?group_id=${g.id}`)}
                className="h-16 w-16 rounded-lg object-cover ring-1 ring-black/20" />
            : <div key={i} className="flex h-16 w-16 items-center justify-center rounded-lg bg-panel2 text-zinc-600">?</div>
        ))}
      </div>
      <div className="flex min-w-0 flex-1 flex-col justify-center">
        <div className="flex items-center gap-1.5">
          {g.notify && <IconBell className="h-3.5 w-3.5 shrink-0 text-accent" />}
          <span className="truncate font-semibold">{g.name}</span>
        </div>
        <div className="mt-0.5 text-[11px] text-zinc-500">
          {g.count} sighting{g.count === 1 ? "" : "s"}
          {g.last_seen ? <> · last seen <span className="text-zinc-400">{timeAgo(g.last_seen)}</span></> : null}
        </div>
        {pending > 0 && (
          <span className="mt-1.5 inline-flex w-fit items-center gap-1 rounded-full bg-accent px-2 py-0.5 text-[11px] font-semibold text-black">
            {pending} new to review →
          </span>
        )}
      </div>
      <span className="flex items-center text-lg text-zinc-600">›</span>
    </Link>
  );
}

function GroupsView({ onAuth }: { onAuth: () => void }) {
  const { data: groups } = usePoll(getGroups, 6000, onAuth);
  const [q, setQ] = useState("");
  const [results, setResults] = useState<ScoredArtifact[] | null>(null);
  const [busy, setBusy] = useState(false);

  async function runSearch(e: React.FormEvent) {
    e.preventDefault();
    if (!q.trim()) return;
    setBusy(true);
    try {
      setResults(await searchArtifacts(q));
    } catch (err: any) {
      if (err?.name === "AuthError") onAuth();
    } finally {
      setBusy(false);
    }
  }

  async function doRecluster() {
    setBusy(true);
    try { await recluster(); } finally { setBusy(false); }
  }

  const labeled = groups?.filter((g) => g.kind === "labeled" && g.tag_key && g.tag_value) ?? [];
  // Filter suggestions to hide uninteresting background/static classes (potted plants, ovens, etc.)
  // and show more of them (up to 30) so vehicle/bicycle groups are visible.
  const clusters = (groups?.filter((g) => {
    if (g.kind !== "cluster") return false;
    const h = g.hint ? g.hint.toLowerCase() : "";
    return !(
      h.includes("plant") ||
      h.includes("oven") ||
      h.includes("handbag") ||
      h.includes("backpack") ||
      h.includes("suitcase") ||
      h.includes("chair") ||
      h.includes("table")
    );
  }) ?? []).slice(0, 30);

  return (
    <div>
      {/* zero-shot search */}
      <form onSubmit={runSearch} className="mb-2 flex gap-2">
        <div className="flex flex-1 items-center gap-2 rounded-xl border border-line bg-panel px-3">
          <IconSearch className="h-4 w-4 text-zinc-500" />
          <input value={q} onChange={(e) => setQ(e.target.value)}
            placeholder="Find by description e.g. 'delivery van'"
            className="flex-1 bg-transparent py-3 text-[16px] outline-none" />
        </div>
        <button disabled={busy} className="rounded-xl bg-accent px-4 text-black disabled:opacity-40">Go</button>
      </form>

      {results && (
        <div className="mb-4 rounded-xl border border-line bg-panel p-3">
          <div className="mb-2 flex items-center justify-between">
            <span className="text-sm text-zinc-400">{results.length} matches for “{q}”</span>
          </div>
          <div className="grid grid-cols-3 gap-1.5 sm:grid-cols-4">
            {results.slice(0, 12).map((a) => (
              <Link key={a.id} to={`/artifacts/${a.id}`} className="relative">
                <img src={withToken(a.images[0]?.url)} className="aspect-square w-full rounded-md object-cover" />
                <span className="absolute bottom-0.5 right-0.5 rounded bg-black/70 px-1 text-[9px] text-accent">
                  {a.score?.toFixed(2)}
                </span>
              </Link>
            ))}
          </div>
          <button onClick={() => setResults(null)} className="mt-2 text-xs text-zinc-500">clear</button>
        </div>
      )}

      {/* labeled groups */}
      {labeled.length > 0 && (
        <>
          <div className="mb-2 flex items-center justify-between">
            <h2 className="text-[13px] font-medium text-zinc-400">Active Tags</h2>
            {(() => {
              const toReview = labeled.reduce((n, g) => n + (g.pending ?? 0), 0);
              return toReview > 0 ? (
                <span className="rounded-full bg-accent/15 px-2 py-0.5 text-[11px] font-medium text-accent">
                  {toReview} to review
                </span>
              ) : null;
            })()}
          </div>
          <div className="mb-4 space-y-2">
            {labeled.map((g) => <GroupCard key={g.id} g={g} />)}
          </div>
        </>
      )}

      {/* cluster suggestions */}
      <div className="mb-1 flex items-center justify-between">
        <h2 className="text-[13px] font-medium text-zinc-400">Suggested Clusters</h2>
        <button onClick={doRecluster} disabled={busy}
          className="rounded-lg border border-line px-2.5 py-1 text-xs text-zinc-400 disabled:opacity-40">
          {busy ? "…" : "↻ Recluster"}
        </button>
      </div>
      <p className="mb-2 text-[11px] text-zinc-600">
        We have grouped these similar-looking artifacts together. Tap a cluster to explore and tag them!
      </p>
      {!groups ? (
        <p className="mt-6 text-center text-sm text-zinc-600">loading…</p>
      ) : clusters.length === 0 ? (
        <p className="mt-6 text-center text-sm text-zinc-600">No suggestions yet — tap Recluster.</p>
      ) : (
        <div className="space-y-2">
          {clusters.map((g) => (
            <Link key={g.id} to={`/groups/${g.id}`}
               className="flex items-center gap-3 rounded-xl border border-line bg-panel p-2.5">
               <Thumbs urls={g.samples} groupId={g.id} />
              <div className="min-w-0 flex-1">
                <div className="truncate text-zinc-300">
                  {g.hint ? <>Suggested Cluster · <span className="font-semibold text-accent">{g.count} {g.hint}</span></> : "Suggested Cluster"}
                </div>
                <div className="text-[11px] text-zinc-500">Hey, these look similar, what are they?</div>
              </div>
              <span className="text-zinc-600">›</span>
            </Link>
          ))}
        </div>
      )}

      <NotifyCard onAuth={onAuth} />
    </div>
  );
}

function NotifyCard({ onAuth }: { onAuth: () => void }) {
  const { data } = usePoll(notifyStatus, 30000, onAuth);
  const [sent, setSent] = useState<string>("");
  if (!data) return null;
  return (
    <div className="mt-5 rounded-xl border border-line bg-panel p-3.5 text-[12px]">
      <div className="mb-1 font-medium text-zinc-300">Notifications</div>
      {data.enabled ? (
        <>
          <p className="text-zinc-500">
            Push via ntfy topic <span className="text-accent">{data.topic}</span>.
            Subscribe to it in the ntfy app, then toggle the bell on any group.
          </p>
          <button
            onClick={async () => {
              try { await notifyTest(); setSent("Sent — check your phone."); }
              catch (e: any) { setSent(e?.message || "failed"); }
            }}
            className="mt-2 rounded-lg bg-accent/20 px-3 py-1 text-accent">Send test</button>
          {sent && <span className="ml-2 text-zinc-400">{sent}</span>}
        </>
      ) : (
        <p className="text-zinc-500">
          Off. Set <code className="text-zinc-300">STREETCAPTURE_NTFY_TOPIC</code> to a private
          topic name and subscribe to it in the free ntfy app to get “DPD at the door” pushes.
        </p>
      )}
    </div>
  );
}

function EntitiesView({ onAuth }: { onAuth: () => void }) {
  const { data } = usePoll(getEntities, 6000, onAuth);
  if (!data) return <p className="mt-6 text-center text-sm text-zinc-600">loading…</p>;
  if (data.length === 0)
    return <p className="mt-6 text-center text-sm text-zinc-600">
      No repeat entities yet. Entities appear when the same-looking object is seen more than once.
    </p>;
  return (
    <div className="space-y-2">
      {data.map((e) => (
        <Link key={e.id} to={`/entities/${e.id}`}
          className="flex items-center gap-3 rounded-xl border border-line bg-panel p-2.5">
          <Thumbs urls={e.samples} />
          <div className="min-w-0 flex-1">
            <div className={`font-medium capitalize ${CAT_COLOR[cat(e.class)]}`}>
              {e.label || e.class}
            </div>
            <div className="text-[11px] text-zinc-500">
              seen {e.occurrences}× · last {timeAgo(e.last_seen)}
            </div>
          </div>
          <span className="text-zinc-600">›</span>
        </Link>
      ))}
    </div>
  );
}
