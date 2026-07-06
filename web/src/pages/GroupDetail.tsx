import { useEffect, useRef, useState } from "react";
import { Link, useNavigate, useParams } from "react-router-dom";
import {
  deleteGroup, getGroupMembers, getGroups, nameGroup, setGroupNotify,
  setMemberStatus, setMembersBatchStatus, backfillGroup, withToken,
  tagArtifacts, getTagsAutocomplete, acceptRemainingSuggestions, type Artifact, type Group,
} from "../api";
import { IconBack, IconBell, IconTrash } from "../lib";

export default function GroupDetail() {
  const { id } = useParams();
  const gid = Number(id);
  const nav = useNavigate();
  const [group, setGroup] = useState<Group | null>(null);
  const [members, setMembers] = useState<Artifact[] | null>(null);
  const pendingRequests = useRef(0);

  async function load() {
    const [gs, ms] = await Promise.all([getGroups(), getGroupMembers(gid)]);
    setGroup(gs.find((g) => g.id === gid) || null);
    setMembers(ms);
  }
  async function loadMore() {
    if (!members) return;
    try {
      const next = await getGroupMembers(gid, 120, members.length);
      setMembers((prev) => prev ? [...prev, ...next] : next);
    } catch (e) {
      console.error(e);
    }
  }
  useEffect(() => {
    load();
    loadAutocomplete();
  }, [gid]);

  async function rename() {
    let defaultName = group?.name || "";
    if (!defaultName && group?.hint) {
      let h = group.hint;
      if (h === "people") h = "person";
      else if (h.endsWith("s")) h = h.slice(0, -1);
      defaultName = h;
    }
    const name = window.prompt("Group name", defaultName);
    if (!name) return;
    await nameGroup(gid, name);
    load();
  }
  async function toggleNotify() {
    if (!group) return;
    await setGroupNotify(gid, !group.notify);
    load();
  }
  async function remove() {
    const isLabeled = group?.kind === "labeled";
    const confirmMsg = isLabeled ? "Delete this tag group? (artifacts are kept)" : "Dismiss this cluster? It will be removed from suggestions.";
    if (!window.confirm(confirmMsg)) return;
    await deleteGroup(gid);
    nav("/groups");
  }

  // Optimistic member feedback: update the tile immediately, sync in the bg.
  // Rejected + removed both vanish from view (rejected stays a negative example
  // in the DB so it can't auto-rejoin).
  async function act(aid: number, status: "confirmed" | "rejected" | "removed") {
    setMembers((ms) => {
      if (!ms) return ms;
      if (status === "confirmed") return ms.map((m) => (m.id === aid ? { ...m, member_status: status } : m));
      return ms.filter((m) => m.id !== aid);
    });
    try {
      await setMemberStatus(gid, aid, status);
    } catch (e) {
      console.error("Failed to update member status", e);
    }
  }

  async function discardNegatives() {
    if (!members) return;
    const toReject = members.filter((m) => m.member_status !== "confirmed").map((m) => m.id);
    if (toReject.length === 0) return;
    if (!window.confirm(`Discard ${toReject.length} leftover suggestions as negative examples?`)) return;

    setMembers((ms) => {
      if (!ms) return ms;
      return ms.filter((m) => m.member_status === "confirmed");
    });

    try {
      await setMembersBatchStatus(gid, toReject, "rejected");
    } catch (e) {
      console.error("Failed to discard negatives", e);
    }
  }

  async function acceptRemaining() {
    if (!window.confirm("Accept all remaining suggestions as correct, and auto-classify future matches for this group?")) return;

    // Optimistically mark pending items as confirmed
    setMembers((ms) => {
      if (!ms) return ms;
      return ms.map((m) => (m.member_status === "rejected" ? m : { ...m, member_status: "confirmed" as const }));
    });

    try {
      await acceptRemainingSuggestions(gid);
      load();
    } catch (e) {
      console.error("Failed to accept remaining suggestions", e);
    }
  }

  const [threshold, setThreshold] = useState<number | null>(null);
  const [backfilling, setBackfilling] = useState(false);
  const [backfillMsg, setBackfillMsg] = useState("");

  const [selectedIds, setSelectedIds] = useState<Set<number>>(new Set());
  const [showTagModal, setShowTagModal] = useState(false);
  const [tagKey, setTagKey] = useState("");
  const [tagValue, setTagValue] = useState("");
  const [autocompleteData, setAutocompleteData] = useState<{ keys: string[]; values: Record<string, string[]> }>({ keys: [], values: {} });

  const [tagsList, setTagsList] = useState<Array<{ key: string; value: string }>>([]);
  const [zoomArtifactId, setZoomArtifactId] = useState<number | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [trainerTagKey, setTrainerTagKey] = useState("");
  const [trainerTagValue, setTrainerTagValue] = useState("");
  const [trainerImgLoadedId, setTrainerImgLoadedId] = useState<number | null>(null);

  async function loadAutocomplete() {
    try {
      const data = await getTagsAutocomplete();
      setAutocompleteData(data);
    } catch (e) {
      console.error(e);
    }
  }

  function toggleSelect(aid: number) {
    setSelectedIds((prev) => {
      const next = new Set(prev);
      if (next.has(aid)) {
        next.delete(aid);
      } else {
        next.add(aid);
      }
      return next;
    });
  }

  function toggleSelectAll() {
    if (!members) return;
    if (selectedIds.size === members.length) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(members.map((m) => m.id)));
    }
  }

  function addTagToList() {
    const k = tagKey.trim();
    const v = tagValue.trim();
    if (!k || !v) return;
    if (tagsList.some((t) => t.key.toLowerCase() === k.toLowerCase() && t.value.toLowerCase() === v.toLowerCase())) return;
    setTagsList((prev) => [...prev, { key: k, value: v }]);
    setTagValue("");
  }

  function removeTagFromList(idx: number) {
    setTagsList((prev) => prev.filter((_, i) => i !== idx));
  }

  async function submitTag(e: React.FormEvent) {
    if (e) e.preventDefault();
    if (selectedIds.size === 0) return;
    
    let currentTags = [...tagsList];
    const k = tagKey.trim();
    const v = tagValue.trim();
    if (k && v && !currentTags.some((t) => t.key.toLowerCase() === k.toLowerCase() && t.value.toLowerCase() === v.toLowerCase())) {
      currentTags.push({ key: k, value: v });
    }
    if (currentTags.length === 0) return;

    const ids = Array.from(selectedIds);
    try {
      await tagArtifacts(ids, currentTags, gid);
      setSelectedIds(new Set());
      setTagsList([]);
      setTagKey("");
      setTagValue("");
      setShowTagModal(false);
      load();
      loadAutocomplete();
    } catch (err) {
      console.error(err);
    }
  }

  useEffect(() => {
    if (members && threshold === null) {
      const isSeeded = members.some((m) => m.member_source === "seed");
      setThreshold(isSeeded ? 0.82 : 0.72);
    }
  }, [members]);

  async function runBackfill() {
    setBackfilling(true);
    setBackfillMsg("");
    try {
      const r = await backfillGroup(gid, threshold ?? undefined);
      if (r.matched > 0) {
        setBackfillMsg(`✅ Found ${r.matched} matching artifacts!`);
        load();
      } else {
        setBackfillMsg(`No new matches at ${threshold?.toFixed(2)} — try lowering the threshold or confirming more examples first.`);
      }
    } catch (e: any) {
      setBackfillMsg(e.message || "Search failed");
    } finally {
      setBackfilling(false);
    }
  }

  const isLabeled = group?.kind === "labeled";
  const title = group?.name || "Unnamed cluster";
  const confirmed = members?.filter((m) => m.member_status === "confirmed").length ?? 0;
  const pendingMembers = members?.filter((m) => m.member_status !== "confirmed" && m.member_status !== "rejected") ?? [];

  if (isTraining) {
    const activeItem = pendingMembers[0];
    return (
      <div className="flex h-full flex-col">
        <header className="flex items-center gap-3 border-b border-line bg-panel px-4 py-3"
          style={{ paddingTop: "max(0.75rem, env(safe-area-inset-top))" }}>
          <button onClick={() => { setIsTraining(false); load(); }} className="text-zinc-400"><IconBack className="h-5 w-5" /></button>
          <h1 className="flex-1 truncate text-[15px] font-semibold">⚡ Tinder Train: {title}</h1>
          <button
            onClick={() => { setIsTraining(false); load(); }}
            className="rounded-lg border border-line bg-panel px-3 py-1 text-xs text-zinc-400 hover:bg-zinc-800"
          >
            ✕ Exit
          </button>
        </header>

        <main className="flex-1 flex flex-col items-center justify-center p-6 bg-zinc-950/20 overflow-y-auto">
          {!activeItem ? (
            <div className="text-center p-6 bg-panel border border-line rounded-2xl max-w-sm w-full">
              <div className="text-3xl mb-3">🎉</div>
              <h2 className="text-sm font-semibold text-zinc-200 mb-1">All Caught Up!</h2>
              <p className="text-xs text-zinc-500 mb-4">No more suggested artifacts left to review for this tag group.</p>
              <button
                onClick={() => { setIsTraining(false); load(); }}
                className="w-full rounded-xl bg-accent py-2 text-xs font-semibold text-black"
              >
                Return to Group
              </button>
            </div>
          ) : (
            <div className="w-full max-w-sm flex flex-col gap-4">
              <div className="text-center text-xs text-zinc-500 font-medium">
                {pendingMembers.length} suggested {pendingMembers.length === 1 ? "item" : "items"} remaining
              </div>

              {/* Card stack container */}
              <div className="relative aspect-video w-full rounded-2xl border border-line bg-panel shadow-2xl overflow-hidden flex items-center justify-center group bg-black/30">
                <img
                  src={withToken(`/api/media/${activeItem.id}/0/full?draw_box=1`)}
                  onLoad={() => setTrainerImgLoadedId(activeItem.id)}
                  className={`w-full h-full object-contain select-none transition-all duration-250 ${
                    trainerImgLoadedId === activeItem.id ? "opacity-100 scale-100 blur-0" : "opacity-20 scale-95 blur-md"
                  }`}
                  alt="Full Frame with highlighted target"
                />
                
                {/* Preload next image frame */}
                {pendingMembers[1] && (
                  <img
                    src={withToken(`/api/media/${pendingMembers[1].id}/0/full?draw_box=1`)}
                    style={{ display: "none" }}
                    alt="Preloading next card frame"
                  />
                )}
                
                {/* Spinner Overlay */}
                {trainerImgLoadedId !== activeItem.id && (
                  <div className="absolute inset-0 bg-black/40 flex flex-col items-center justify-center gap-2 backdrop-blur-sm animate-in fade-in duration-100">
                    <div className="w-8 h-8 rounded-full border-2 border-accent/20 border-t-accent animate-spin" />
                    <span className="text-[10px] text-zinc-400 font-semibold uppercase tracking-wider">Loading DVR Frame...</span>
                  </div>
                )}
                
                {/* Auto source indicator */}
                <span className="absolute left-3 top-3 rounded-md bg-black/70 border border-line/40 px-2 py-0.5 text-[9px] font-semibold text-accent uppercase tracking-wider">
                  Inferred Match
                </span>

                {/* Zoom button */}
                <button
                  type="button"
                  onClick={() => setZoomArtifactId(activeItem.id)}
                  className="absolute bottom-3 left-3 rounded-lg bg-black/80 border border-line/45 p-2 text-xs text-zinc-300 hover:text-accent hover:bg-black/90 active:scale-90 transition shadow-lg"
                  title="Zoom to full frame context"
                >
                  🔍 Zoom
                </button>
              </div>

              {/* Action buttons */}
              <div className="grid grid-cols-3 gap-2">
                <button
                  onClick={() => act(activeItem.id, "rejected")}
                  className="flex flex-col items-center justify-center gap-1 rounded-xl border border-red-500/30 bg-red-500/10 py-3 text-red-400 hover:bg-red-500/20 active:scale-95 transition"
                >
                  <span className="text-lg">❌</span>
                  <span className="text-[10px] font-semibold uppercase tracking-wider">Decline</span>
                </button>

                <button
                  onClick={() => act(activeItem.id, "removed")}
                  className="flex flex-col items-center justify-center gap-1 rounded-xl border border-line bg-panel py-3 text-zinc-400 hover:bg-zinc-800 active:scale-95 transition"
                >
                  <span className="text-lg">➡️</span>
                  <span className="text-[10px] font-semibold uppercase tracking-wider">Skip</span>
                </button>

                <button
                  onClick={() => act(activeItem.id, "confirmed")}
                  className="flex flex-col items-center justify-center gap-1 rounded-xl border border-green-500/30 bg-green-500/10 py-3 text-green-400 hover:bg-green-500/20 active:scale-95 transition"
                >
                  <span className="text-lg"></span>
                  <span className="text-[10px] font-semibold uppercase tracking-wider">Approve</span>
                </button>
              </div>

              {/* Inline quick tag editor */}
              <div className="rounded-xl border border-line bg-panel p-3">
                <div className="text-[11px] font-semibold text-zinc-400 mb-1.5 uppercase tracking-wider">Retag this item instead:</div>
                <div className="flex gap-1.5">
                  <input
                    type="text"
                    placeholder="Key (e.g. clothing)"
                    value={trainerTagKey}
                    onChange={(e) => setTrainerTagKey(e.target.value)}
                    list="trainer-tag-keys"
                    className="flex-1 rounded-lg border border-line bg-panel2 px-2.5 py-1.5 text-xs text-zinc-100 focus:border-accent focus:outline-none"
                  />
                  <datalist id="trainer-tag-keys">
                    {autocompleteData.keys.map((k) => (
                      <option key={k} value={k} />
                    ))}
                  </datalist>

                  <input
                    type="text"
                    placeholder="Value (e.g. dress)"
                    value={trainerTagValue}
                    onChange={(e) => setTrainerTagValue(e.target.value)}
                    list="trainer-tag-values"
                    className="flex-1 rounded-lg border border-line bg-panel2 px-2.5 py-1.5 text-xs text-zinc-100 focus:border-accent focus:outline-none"
                  />
                  <datalist id="trainer-tag-values">
                    {(autocompleteData.values[trainerTagKey] || []).map((v) => (
                      <option key={v} value={v} />
                    ))}
                  </datalist>

                  <button
                    onClick={async () => {
                      const k = trainerTagKey.trim();
                      const v = trainerTagValue.trim();
                      if (!k || !v) return;
                      try {
                        await tagArtifacts([activeItem.id], [{ key: k, value: v }]);
                        await act(activeItem.id, "removed"); // remove from this group suggestions
                        setTrainerTagValue("");
                      } catch (err) {
                        console.error(err);
                      }
                    }}
                    disabled={!trainerTagKey.trim() || !trainerTagValue.trim()}
                    className="rounded-lg bg-accent px-3 text-xs font-semibold text-black disabled:opacity-40"
                  >
                    Save
                  </button>
                </div>
              </div>
            </div>
          )}
        </main>
      </div>
    );
  }

  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center gap-3 border-b border-line bg-panel px-4 py-3"
        style={{ paddingTop: "max(0.75rem, env(safe-area-inset-top))" }}>
        <button onClick={() => nav(-1)} className="text-zinc-400"><IconBack className="h-5 w-5" /></button>
        <h1 className="flex-1 truncate text-[15px] font-semibold">{title}</h1>
        {group && (
          <div className="flex items-center gap-2">
            {isLabeled ? (
              <>
                <button onClick={toggleNotify} className={group.notify ? "text-accent" : "text-zinc-500"}>
                  <IconBell className="h-5 w-5" />
                </button>
                <button onClick={remove} className="text-zinc-500 hover:text-red-400">
                  <IconTrash className="h-5 w-5" />
                </button>
              </>
            ) : (
              <button
                onClick={remove}
                className="rounded-lg border border-red-500/30 bg-red-500/10 px-2.5 py-1 text-[11px] font-medium text-red-400 hover:bg-red-500/20 active:scale-95 transition"
              >
                ✕ Close Cluster
              </button>
            )}
          </div>
        )}
      </header>

      <main className="flex-1 overflow-y-auto p-4">
        <div className="mb-3 flex flex-wrap items-center gap-2">
          <button onClick={rename} className="rounded-lg bg-accent px-3 py-1.5 text-sm font-medium text-black">
            {isLabeled ? "Rename" : "Name this group"}
          </button>
          {isLabeled && (
            <button onClick={toggleNotify}
              className={`flex items-center gap-1.5 rounded-lg border px-3 py-1.5 text-sm ${
                group?.notify ? "border-accent text-accent" : "border-line text-zinc-400"}`}>
              <IconBell className="h-4 w-4" />
              {group?.notify ? "Notifying" : "Notify me"}
            </button>
          )}
        </div>

        {isLabeled && threshold !== null && (
          <div className="mb-3 rounded-xl border border-line bg-panel p-3 text-sm">
            <div className="flex items-center justify-between mb-2">
              <span className="text-zinc-400">Search breadth:</span>
              <span className="font-medium text-accent">
                {threshold.toFixed(2)} — {threshold >= 0.82 ? "Strict 🎯" : threshold >= 0.76 ? "Balanced ⚖️" : "Relaxed 🔍"}
              </span>
            </div>
            <div className="flex items-center gap-3">
              <input type="range" min="0.60" max="0.90" step="0.02" value={threshold}
                onChange={(e) => setThreshold(parseFloat(e.target.value))}
                className="flex-1 accent-accent h-1 bg-zinc-800 rounded-lg appearance-none cursor-pointer" />
              <button onClick={runBackfill} disabled={backfilling}
                className="rounded-lg bg-accent px-4 py-1.5 text-xs font-semibold text-black disabled:opacity-50">
                {backfilling ? "Searching..." : "Find others"}
              </button>
            </div>
          </div>
        )}

        {backfillMsg && (
          <p className="mb-3 text-[12px] text-accent font-medium animate-pulse">{backfillMsg}</p>
        )}

        {isLabeled ? (
          <div className="mb-3 rounded-xl border border-line bg-panel p-3">
            <p className="text-[13px] font-medium text-zinc-200">Are these all “{title}”?</p>
            <p className="mt-1 text-[12px] text-zinc-500">
              Confirm the good ones and reject the wrong ones. The group re-learns from
              the confirmed set, so future matches and notifications get sharper.
            </p>
            <div className="mt-3 flex items-center justify-between">
              <div className="flex gap-2">
                {confirmed > 0 && (
                  <p className="text-[11px] text-accent font-medium bg-accent/10 px-2 py-1 rounded">{confirmed} confirmed</p>
                )}
                {pendingMembers.length > 0 && (
                  <button
                    onClick={() => setIsTraining(true)}
                    className="flex items-center gap-1.5 rounded-lg bg-accent px-3 py-1 text-xs font-semibold text-black hover:bg-accent/90 active:scale-95 transition"
                  >
                    ⚡ Tinder Train ({pendingMembers.length})
                  </button>
                )}
              </div>
              {members && members.some((m) => m.member_status !== "confirmed") && (
                <div className="flex gap-2">
                  <button
                    onClick={acceptRemaining}
                    className="rounded-lg border border-accent/30 bg-accent/10 px-2.5 py-1 text-[11px] font-medium text-accent hover:bg-accent/20 active:scale-95 transition"
                    title="Accept remaining suggestions and auto-classify future matches"
                  >
                    ✓ Accept Remaining
                  </button>
                  <button
                    onClick={discardNegatives}
                    className="rounded-lg border border-red-500/30 bg-red-500/10 px-2.5 py-1 text-[11px] font-medium text-red-400 hover:bg-red-500/20 active:scale-95 transition"
                    title="Mark leftover suggestions as wrong matches to train the classifier"
                  >
                    🗑️ Discard Negatives
                  </button>
                </div>
              )}
            </div>
          </div>
        ) : (
          <div className="mb-3">
            <p className="text-[12px] text-zinc-500 mb-3">
              Select the matching artifacts in this cluster and tag them to group and auto-tag them.
            </p>
            {members && (
              <div className="flex items-center gap-2">
                <button onClick={toggleSelectAll} className="rounded-lg border border-line bg-panel px-3 py-1.5 text-xs text-zinc-400">
                  {selectedIds.size === members.length ? "Deselect All" : "Select All"}
                </button>
                <button onClick={() => setShowTagModal(true)} disabled={selectedIds.size === 0}
                  className="rounded-lg bg-accent px-3 py-1.5 text-xs font-semibold text-black disabled:opacity-50">
                  🏷️ Tag Selected ({selectedIds.size})
                </button>
              </div>
            )}
          </div>
        )}

        <p className="mb-2 text-[13px] text-zinc-400">{members?.length ?? 0} artifacts</p>
        {!members ? (
          <p className="mt-6 text-center text-sm text-zinc-600">loading…</p>
        ) : (
          <div className="grid grid-cols-3 gap-2 sm:grid-cols-4">
            {members.map((a) => {
              const st = a.member_status;
              const selected = selectedIds.has(a.id);
              const ring = isLabeled
                ? (st === "confirmed" ? "ring-2 ring-accent" : st === "rejected" ? "opacity-40 ring-2 ring-red-500" : "ring-1 ring-line")
                : (selected ? "ring-4 ring-accent" : "ring-1 ring-line");
              return (
                <div key={a.id} className="flex flex-col gap-1">
                  {isLabeled ? (
                    <Link to={`/artifacts/${a.id}`} className={`relative block overflow-hidden rounded-md ${ring}`}>
                      <img src={withToken(`${a.images[0]?.url}?group_id=${id}`)} className="aspect-square w-full object-cover" />
                      {a.member_source === "auto" && (
                        <span className="absolute left-0.5 top-0.5 rounded bg-black/70 px-1 text-[8px] text-zinc-300">auto</span>
                      )}
                    </Link>
                  ) : (
                    <div onClick={() => toggleSelect(a.id)} className={`relative block overflow-hidden rounded-md cursor-pointer ${ring}`}>
                      <img src={withToken(`${a.images[0]?.url}?group_id=${id}`)} className="aspect-square w-full object-cover animate-in fade-in duration-200" />
                      {a.member_source === "auto" && (
                        <span className="absolute left-0.5 top-0.5 rounded bg-black/70 px-1 text-[8px] text-zinc-300">auto</span>
                      )}
                      <div className="absolute right-1 top-1">
                        <input type="checkbox" checked={selected} readOnly className="h-4.5 w-4.5 accent-accent pointer-events-none" />
                      </div>
                      <button
                        type="button"
                        onClick={(e) => {
                          e.stopPropagation();
                          setZoomArtifactId(a.id);
                        }}
                        className="absolute bottom-1 left-1 rounded bg-black/75 p-1 text-[10px] text-zinc-300 hover:text-accent hover:bg-black/90 active:scale-95 transition"
                        title="Zoom to full frame"
                      >
                        🔍
                      </button>
                    </div>
                  )}
                  {isLabeled && (
                    <div className="flex gap-1">
                      <button onClick={() => act(a.id, st === "confirmed" ? "removed" : "confirmed")}
                        className={`flex-1 rounded py-1 text-[13px] ${
                          st === "confirmed" ? "bg-accent text-black" : "bg-panel2 text-zinc-300"}`}>✓</button>
                      <button onClick={() => act(a.id, "rejected")}
                        className={`flex-1 rounded py-1 text-[13px] ${
                          st === "rejected" ? "bg-red-500 text-black" : "bg-panel2 text-zinc-300"}`}>✕</button>
                    </div>
                  )}
                </div>
              );
            })}
          </div>
        )}

        {/* Load More Pagination Button */}
        {members && group && members.length < group.count && (
          <div className="mt-6 flex justify-center">
            <button
              onClick={loadMore}
              className="rounded-xl border border-line bg-panel px-6 py-2.5 text-xs font-semibold text-accent hover:bg-zinc-800 transition active:scale-95"
            >
              Load More (showing {members.length} of {group.count})
            </button>
          </div>
        )}
      </main>

      {showTagModal && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 p-4">
          <div className="w-full max-w-sm rounded-2xl border border-line bg-panel p-5 shadow-2xl animate-in fade-in zoom-in duration-200">
            <h3 className="text-base font-semibold text-zinc-100">Tag {selectedIds.size} Artifacts</h3>
            
            {tagsList.length > 0 && (
              <div className="mt-2 flex flex-wrap gap-1.5 max-h-24 overflow-y-auto border border-line rounded-lg p-2 bg-panel2">
                {tagsList.map((t, idx) => (
                  <span key={idx} className="flex items-center gap-1 rounded bg-accent/10 px-2 py-0.5 text-xs text-accent">
                    {t.key}:{t.value}
                    <button type="button" onClick={() => removeTagFromList(idx)} className="text-zinc-400 hover:text-red-400 ml-1">×</button>
                  </span>
                ))}
              </div>
            )}

            <div className="mt-4 space-y-4">
              <div className="flex gap-2">
                <div className="flex-1">
                  <label className="block text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Key</label>
                  <input
                    type="text"
                    value={tagKey}
                    onChange={(e) => setTagKey(e.target.value)}
                    placeholder="e.g. clothing"
                    list="tag-keys"
                    className="mt-1 w-full rounded-lg border border-line bg-panel2 px-2.5 py-1.5 text-xs text-zinc-100 focus:border-accent focus:outline-none"
                  />
                  <datalist id="tag-keys">
                    {autocompleteData.keys.map((k) => (
                      <option key={k} value={k} />
                    ))}
                  </datalist>
                </div>

                <div className="flex-1">
                  <label className="block text-[10px] font-semibold uppercase tracking-wider text-zinc-500">Value</label>
                  <input
                    type="text"
                    value={tagValue}
                    onChange={(e) => setTagValue(e.target.value)}
                    placeholder="e.g. dress"
                    list="tag-values"
                    className="mt-1 w-full rounded-lg border border-line bg-panel2 px-2.5 py-1.5 text-xs text-zinc-100 focus:border-accent focus:outline-none"
                  />
                  <datalist id="tag-values">
                    {(autocompleteData.values[tagKey] || []).map((v) => (
                      <option key={v} value={v} />
                    ))}
                  </datalist>
                </div>

                <button
                  type="button"
                  onClick={addTagToList}
                  className="mt-5 rounded-lg border border-line bg-panel px-3 text-xs font-semibold text-zinc-300 hover:bg-zinc-800"
                >
                  + Add
                </button>
              </div>

              <div className="flex gap-2 pt-2 border-t border-line">
                <button
                  type="button"
                  onClick={() => { setShowTagModal(false); setTagsList([]); }}
                  className="flex-1 rounded-lg border border-line py-2 text-sm font-medium text-zinc-400 hover:bg-zinc-800"
                >
                  Cancel
                </button>
                <button
                  type="button"
                  onClick={(e) => submitTag(e as any)}
                  disabled={tagsList.length === 0 && (!tagKey.trim() || !tagValue.trim())}
                  className="flex-1 rounded-lg bg-accent py-2 text-sm font-semibold text-black hover:bg-accent/90 disabled:opacity-50"
                >
                  Save Tags
                </button>
              </div>
            </div>
          </div>
        </div>
      )}

      {zoomArtifactId !== null && (
        <div onClick={() => setZoomArtifactId(null)} className="fixed inset-0 z-50 flex flex-col items-center justify-center bg-black/95 p-4 cursor-pointer animate-in fade-in duration-200">
          <div className="absolute right-4 top-4 text-zinc-400 text-sm font-medium bg-black/55 px-2.5 py-1 rounded-lg border border-line">✕ Close</div>
          <div className="relative max-w-full max-h-[85vh] rounded-xl overflow-hidden border border-line bg-panel shadow-2xl" onClick={(e) => e.stopPropagation()}>
            <img src={withToken(`/api/media/${zoomArtifactId}/0/full`)} className="block max-w-full max-h-[85vh] object-contain" />
          </div>
          <p className="mt-3 text-xs text-zinc-400">Viewing uncropped full frame context</p>
        </div>
      )}
    </div>
  );
}
