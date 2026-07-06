import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { getArtifacts } from "../api";
import { useAuth } from "../auth";
import { ArtifactCard } from "../components/ArtifactCard";
import { Shell } from "../components/Shell";
import { usePoll } from "../lib";

const FILTERS = [
  { label: "All", cls: undefined },
  { label: "People", cls: "person" },
  { label: "Cars", cls: "car" },
  { label: "Trucks", cls: "truck" },
  { label: "Buses", cls: "bus" },
  { label: "Motorcycles", cls: "motorcycle" },
  { label: "Motorbikes", cls: "motorbike" },
  { label: "Bicycles", cls: "bicycle" },
];

export default function Artifacts() {
  const nav = useNavigate();
  const { logout } = useAuth();
  const [cls, setCls] = useState<string | undefined>(undefined);
  const { data } = usePoll(() => getArtifacts(cls), 5000, () => { logout(); nav("/"); });

  return (
    <Shell title="Artifacts">
      <div className="-mx-4 mb-3 flex gap-2 overflow-x-auto px-4 no-scrollbar">
        {FILTERS.map((f) => (
          <button key={f.label} onClick={() => setCls(f.cls)}
            className={`whitespace-nowrap rounded-full px-3.5 py-1.5 text-sm ${
              cls === f.cls ? "bg-accent text-black" : "bg-panel text-zinc-400 border border-line"
            }`}>
            {f.label}
          </button>
        ))}
      </div>

      {!data ? (
        <p className="mt-10 text-center text-sm text-zinc-600">loading…</p>
      ) : data.length === 0 ? (
        <p className="mt-10 text-center text-sm text-zinc-600">
          No artifacts yet — let the camera watch a while.
        </p>
      ) : (
        <div className="grid grid-cols-2 gap-2.5 sm:grid-cols-3">
          {data.map((a) => <ArtifactCard key={a.id} a={a} />)}
        </div>
      )}
    </Shell>
  );
}
