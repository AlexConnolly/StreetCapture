import { useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import {
  getGroups, notifyStatus, notifyTest, searchArtifacts, withToken,
  type Group, type ScoredArtifact,
} from "../api";
import { useAuth } from "../auth";
import { Shell } from "../components/Shell";
import { IconSearch, usePoll } from "../lib";

// --- Discover = Groups | Entities, both deep-linkable -----------------------
export function DiscoverTabs({ active }: { active: "groups" | "entities" }) {
  const base = "flex-1 rounded-lg py-1.5 text-center transition";
  const on = "bg-accent text-black font-medium";
  const off = "text-zinc-400 hover:text-zinc-200";
  return (
    <div className="mb-3 flex rounded-xl border border-line bg-panel p-1 text-sm">
      <Link to="/groups" className={`${base} ${active === "groups" ? on : off}`}>Groups</Link>
      <Link to="/entities" className={`${base} ${active === "entities" ? on : off}`}>Entities</Link>
    </div>
  );
}

// --- tag-key grouping -------------------------------------------------------
interface TagValue { value: string; count: number; groupId: number; pending: number; sample?: string }
interface TagKey { key: string; total: number; values: TagValue[] }

function groupByKey(groups: Group[]): TagKey[] {
  const m = new Map<string, TagValue[]>();
  for (const g of groups) {
    if (g.kind !== "labeled" || !g.tag_key || !g.tag_value) continue;
    if (!m.has(g.tag_key)) m.set(g.tag_key, []);
    m.get(g.tag_key)!.push({
      value: g.tag_value, count: g.count, groupId: g.id,
      pending: g.pending ?? 0, sample: g.samples[0],
    });
  }
  const keys: TagKey[] = Array.from(m.entries()).map(([key, values]) => {
    values.sort((a, b) => b.count - a.count);
    return { key, total: values.reduce((s, v) => s + v.count, 0), values };
  });
  keys.sort((a, b) => b.total - a.total);
  return keys;
}

function TagKeyCard({ tk }: { tk: TagKey }) {
  const avatars = tk.values.slice(0, 6);
  const toReview = tk.values.reduce((n, v) => n + v.pending, 0);
  return (
    <div className="flex flex-col rounded-2xl border border-line bg-panel p-3.5">
      <div className="mb-2.5 flex items-baseline justify-between gap-2">
        <h3 className="truncate text-sm font-semibold capitalize text-zinc-100">{tk.key}</h3>
        <span className="shrink-0 text-[11px] text-zinc-500">{tk.total.toLocaleString()} tagged</span>
      </div>
      <div className="mb-3 flex -space-x-2">
        {avatars.map((v) => (
          v.sample
            ? <img key={v.groupId} title={v.value} src={withToken(`${v.sample}?group_id=${v.groupId}`)}
                className="h-9 w-9 rounded-full object-cover ring-2 ring-panel" />
            : <div key={v.groupId} title={v.value}
                className="flex h-9 w-9 items-center justify-center rounded-full bg-panel2 text-[10px] text-zinc-600 ring-2 ring-panel">?</div>
        ))}
        {tk.values.length > 6 && (
          <div className="flex h-9 w-9 items-center justify-center rounded-full bg-panel2 text-[10px] text-zinc-400 ring-2 ring-panel">
            +{tk.values.length - 6}
          </div>
        )}
      </div>
      <div className="space-y-0.5">
        {tk.values.map((v) => (
          <Link key={v.groupId} to={`/groups/${v.groupId}`}
            className="flex items-center justify-between rounded-lg px-2 py-1.5 text-[13px] transition hover:bg-panel2 active:scale-[0.99]">
            <span className="flex min-w-0 items-center gap-1.5">
              <span className="truncate capitalize text-zinc-200">{v.value}</span>
              {v.pending > 0 && (
                <span className="shrink-0 rounded-full bg-accent/15 px-1.5 text-[10px] font-medium text-accent">{v.pending} new</span>
              )}
            </span>
            <span className="ml-2 shrink-0 tabular-nums text-zinc-500">{v.count.toLocaleString()}</span>
          </Link>
        ))}
      </div>
      {toReview > 0 && (
        <div className="mt-2 text-[11px] text-accent">{toReview} to review</div>
      )}
    </div>
  );
}

function TagKeyGrid({ keys }: { keys: TagKey[] }) {
  return (
    <div className="grid grid-cols-1 gap-3 sm:grid-cols-2 lg:grid-cols-3">
      {keys.map((tk) => <TagKeyCard key={tk.key} tk={tk} />)}
    </div>
  );
}

// --- search box (kept from Discover — zero-shot artifact search) ------------
function SearchBox({ onAuth }: { onAuth: () => void }) {
  const [q, setQ] = useState("");
  const [results, setResults] = useState<ScoredArtifact[] | null>(null);
  const [busy, setBusy] = useState(false);
  async function runSearch(e: React.FormEvent) {
    e.preventDefault();
    if (!q.trim()) return;
    setBusy(true);
    try { setResults(await searchArtifacts(q)); }
    catch (err: any) { if (err?.name === "AuthError") onAuth(); }
    finally { setBusy(false); }
  }
  return (
    <>
      <form onSubmit={runSearch} className="mb-3 flex gap-2">
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
          <div className="mb-2 text-sm text-zinc-400">{results.length} matches for “{q}”</div>
          <div className="grid grid-cols-3 gap-1.5 sm:grid-cols-4 lg:grid-cols-6">
            {results.slice(0, 18).map((a) => (
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
    </>
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
            Push via ntfy topic <span className="text-accent">{data.topic}</span>. Subscribe in the
            ntfy app, then toggle the bell on any group.
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
          Off. Set <code className="text-zinc-300">STREETCAPTURE_NTFY_TOPIC</code> to a private topic
          and subscribe in the free ntfy app to get “DPD at the door” pushes.
        </p>
      )}
    </div>
  );
}

// --- pages ------------------------------------------------------------------
const TOP_KEYS = 5;

export default function Groups() {
  const nav = useNavigate();
  const { logout } = useAuth();
  const onAuth = () => { logout(); nav("/"); };
  const { data: groups } = usePoll(getGroups, 6000, onAuth);
  const keys = groups ? groupByKey(groups) : null;
  const shown = keys?.slice(0, TOP_KEYS) ?? null;

  return (
    <Shell title="Discover">
      <DiscoverTabs active="groups" />
      <SearchBox onAuth={onAuth} />
      {!keys ? (
        <p className="mt-6 text-center text-sm text-zinc-600">loading…</p>
      ) : keys.length === 0 ? (
        <p className="mt-8 text-center text-sm text-zinc-600">
          No tags yet. Tag some artifacts (e.g. <span className="text-zinc-400">gender: male</span>) and
          they’ll group here by tag.
        </p>
      ) : (
        <>
          <TagKeyGrid keys={shown!} />
          {keys.length > TOP_KEYS && (
            <div className="mt-4 flex justify-center">
              <Link to="/groups/all"
                className="rounded-xl border border-line bg-panel px-6 py-2.5 text-xs font-semibold text-accent transition hover:bg-zinc-800 active:scale-95">
                View all {keys.length} tags →
              </Link>
            </div>
          )}
        </>
      )}
      <NotifyCard onAuth={onAuth} />
    </Shell>
  );
}

export function AllTags() {
  const nav = useNavigate();
  const { logout } = useAuth();
  const onAuth = () => { logout(); nav("/"); };
  const { data: groups } = usePoll(getGroups, 6000, onAuth);
  const keys = groups ? groupByKey(groups) : null;

  return (
    <Shell title="All tags">
      <Link to="/groups" className="mb-3 inline-flex items-center gap-1 text-[13px] text-zinc-400 hover:text-accent">
        ‹ Back to Groups
      </Link>
      {!keys ? (
        <p className="mt-6 text-center text-sm text-zinc-600">loading…</p>
      ) : keys.length === 0 ? (
        <p className="mt-8 text-center text-sm text-zinc-600">No tags yet.</p>
      ) : (
        <TagKeyGrid keys={keys} />
      )}
    </Shell>
  );
}
