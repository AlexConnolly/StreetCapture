import { Link } from "react-router-dom";
import { withToken, type Artifact } from "../api";
import { cat, CAT_COLOR, timeAgo } from "../lib";

export function ArtifactCard({ a }: { a: Artifact }) {
  const thumb = a.images[0];
  return (
    <Link
      to={`/artifacts/${a.id}`}
      className="overflow-hidden rounded-xl border border-line bg-panel active:opacity-80"
    >
      <div className="aspect-square w-full bg-black">
        {thumb ? (
          <img src={withToken(thumb.url)} alt={a.class} loading="lazy"
            className="h-full w-full object-cover" />
        ) : (
          <div className="flex h-full items-center justify-center text-xs text-zinc-600">no image</div>
        )}
      </div>
      <div className="p-2.5">
        <div className="flex items-center justify-between">
          <span className={`text-sm font-semibold capitalize ${CAT_COLOR[cat(a.class)]}`}>
            {a.class}
          </span>
          <span className="text-[10px] text-zinc-500">#{a.id}</span>
        </div>
        <div className="mt-0.5 text-[11px] text-zinc-500">
          {a.duration.toFixed(1)}s · {timeAgo(a.start)}
        </div>
      </div>
    </Link>
  );
}
