import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { deleteClip, getLibrary, libraryClipUrl, type LibraryClip } from "../api";
import { IconBack, IconTrash } from "../lib";

function prettyName(name: string) {
  return name.replace(/__\d{8}-\d{6}\.mp4$/, "").replace(/_/g, " ");
}
function fmtSize(b: number) {
  return b > 1e6 ? `${(b / 1e6).toFixed(1)} MB` : `${(b / 1e3).toFixed(0)} KB`;
}

export default function Library() {
  const nav = useNavigate();
  const [clips, setClips] = useState<LibraryClip[] | null>(null);
  const [playing, setPlaying] = useState<string | null>(null);

  async function load() { setClips(await getLibrary()); }
  useEffect(() => { load(); }, []);

  async function remove(name: string) {
    if (!window.confirm("Delete this saved clip?")) return;
    await deleteClip(name);
    if (playing === name) setPlaying(null);
    load();
  }

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center gap-3 border-b border-line bg-panel px-4 py-3"
        style={{ paddingTop: "max(0.75rem, env(safe-area-inset-top))" }}>
        <button onClick={() => nav(-1)} className="text-zinc-400"><IconBack className="h-5 w-5" /></button>
        <h1 className="text-[15px] font-semibold">Library</h1>
      </header>

      <main className="flex-1 overflow-y-auto p-4">
        <p className="mb-3 text-[12px] text-zinc-500">
          Saved clips are kept forever — they're pulled out of the 24h recording prune.
        </p>
        {!clips ? (
          <p className="mt-6 text-center text-sm text-zinc-600">loading…</p>
        ) : clips.length === 0 ? (
          <p className="mt-10 text-center text-sm text-zinc-600">
            No saved clips yet. On Live, tap ✂ Clip, drag the timeline to select a moment, then Save.
          </p>
        ) : (
          <div className="space-y-3">
            {clips.map((c) => (
              <div key={c.name} className="overflow-hidden rounded-xl border border-line bg-panel">
                {playing === c.name ? (
                  <video src={libraryClipUrl(c.name)} controls autoPlay playsInline
                    className="w-full bg-black" />
                ) : (
                  <button onClick={() => setPlaying(c.name)}
                    className="flex h-28 w-full items-center justify-center bg-black/60 text-3xl text-accent">
                    ⏵
                  </button>
                )}
                <div className="flex items-center gap-2 px-3 py-2">
                  <div className="min-w-0 flex-1">
                    <div className="truncate text-sm font-medium capitalize">{prettyName(c.name)}</div>
                    <div className="text-[11px] text-zinc-500">
                      {new Date(c.saved_at * 1000).toLocaleString()} · {fmtSize(c.size)}
                    </div>
                  </div>
                  <button onClick={() => remove(c.name)} className="text-zinc-500">
                    <IconTrash className="h-5 w-5" />
                  </button>
                </div>
              </div>
            ))}
          </div>
        )}
      </main>
    </div>
  );
}
