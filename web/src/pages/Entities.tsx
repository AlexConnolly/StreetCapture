import { Link, useNavigate } from "react-router-dom";
import { getEntities, withToken, type Entity } from "../api";
import { useAuth } from "../auth";
import { Shell } from "../components/Shell";
import { cat, CAT_COLOR, timeAgo, usePoll } from "../lib";
import { DiscoverTabs } from "./Groups";

function EntityCard({ e }: { e: Entity }) {
  return (
    <Link to={`/entities/${e.id}`}
      className="group flex flex-col overflow-hidden rounded-2xl border border-line bg-panel transition active:scale-[0.99]">
      <div className="relative aspect-square w-full overflow-hidden bg-panel2">
        {e.samples[0]
          ? <img src={withToken(e.samples[0])} className="h-full w-full object-cover" />
          : <div className="flex h-full w-full items-center justify-center text-zinc-600">?</div>}
        <span className="absolute right-1.5 top-1.5 rounded-md bg-black/70 px-1.5 py-0.5 text-[10px] font-medium text-zinc-200">
          seen {e.occurrences}×
        </span>
      </div>
      <div className="p-2.5">
        <div className={`truncate text-[13px] font-medium capitalize ${CAT_COLOR[cat(e.class)]}`}>
          {e.label || e.class}
        </div>
        <div className="mt-0.5 text-[11px] text-zinc-500">last {timeAgo(e.last_seen)}</div>
      </div>
    </Link>
  );
}

export default function Entities() {
  const nav = useNavigate();
  const { logout } = useAuth();
  const onAuth = () => { logout(); nav("/"); };
  const { data } = usePoll(getEntities, 6000, onAuth);

  return (
    <Shell title="Discover">
      <DiscoverTabs active="entities" />
      {!data ? (
        <p className="mt-6 text-center text-sm text-zinc-600">loading…</p>
      ) : data.length === 0 ? (
        <p className="mt-8 text-center text-sm text-zinc-600">
          No repeat entities yet. Entities appear when the same-looking object is seen more than once.
        </p>
      ) : (
        <div className="grid grid-cols-2 gap-3 sm:grid-cols-3 lg:grid-cols-4 xl:grid-cols-5">
          {data.map((e) => <EntityCard key={e.id} e={e} />)}
        </div>
      )}
    </Shell>
  );
}
