// Typed API client for the StreetCapture backend.

const TOKEN_KEY = "sc_token";

export function getToken(): string {
  return localStorage.getItem(TOKEN_KEY) || "";
}
export function setToken(t: string) {
  localStorage.setItem(TOKEN_KEY, t);
}
export function clearToken() {
  localStorage.removeItem(TOKEN_KEY);
}

export class AuthError extends Error {}

async function api<T>(path: string, opts: RequestInit = {}): Promise<T> {
  const headers: Record<string, string> = {
    // Bypass the ngrok-free browser-warning interstitial on XHR requests.
    "ngrok-skip-browser-warning": "1",
    ...(opts.headers as Record<string, string>),
  };
  const token = getToken();
  if (token) headers["Authorization"] = `Bearer ${token}`;
  if (opts.body) headers["Content-Type"] = "application/json";
  const res = await fetch(path, { ...opts, headers });
  if (res.status === 401) {
    clearToken();
    throw new AuthError("unauthorized");
  }
  if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
  return res.json() as Promise<T>;
}

// -- types -----------------------------------------------------------------
export interface DvrStat {
  recording: boolean;
  segments?: number;
  bytes?: number;
  earliest?: number | null;
}
export interface Stats {
  day: string;
  active: number;
  active_by_cat: Record<string, number>;
  daily: Record<string, number>;
  artifacts: Record<string, number>;
  events: string[];
  fps: number;
  source_fps: number;
  stale_pct: number;
  idle_objects?: number;
  online: boolean;
  uptime_s: number;
  faiss_vectors: number;
  embed_model: string;
  dvr: DvrStat;
}
export interface Label {
  type: string;
  value: string;
}
export interface ArtifactImage {
  url: string;
  rank: number;
  w: number;
  h: number;
  sharpness: number;
}
export interface Artifact {
  id: number;
  class: string;
  track_id: number;
  entity_id: number | null;
  start: number;
  end: number;
  duration: number;
  confidence: number;
  sharpness: number;
  visibility: number;
  motion: number;
  frames: number;
  labels: Label[];
  embedding: { model_version: string; dim: number } | null;
  images: ArtifactImage[];
  member_status?: "confirmed" | "rejected" | null;
  member_source?: string | null;
  seeds?: { group_id: number; label: string; box: number[] }[];
}
export interface EventItem {
  type: string;
  source_track_id: number | null;
  artifact_id: number | null;
  class: string | null;
  duration: number | null;
  reason: string | null;
  time: number;
}

// -- calls -----------------------------------------------------------------
export async function login(password: string): Promise<string> {
  const r = await api<{ token: string }>("/api/login", {
    method: "POST",
    body: JSON.stringify({ password }),
  });
  setToken(r.token);
  return r.token;
}
export const getStats = () => api<Stats>("/api/stats");
export const getArtifacts = (cls?: string) =>
  api<Artifact[]>(`/api/artifacts?limit=80${cls ? `&cls=${cls}` : ""}`);
export const getArtifact = (id: number) => api<Artifact>(`/api/artifacts/${id}`);
export const getEvents = () => api<EventItem[]>("/api/events?limit=80");
export const ask = (q: string) =>
  api<{ question: string; answer: string }>(`/api/query?q=${encodeURIComponent(q)}`);
export const getHourly = (cls: string) =>
  api<{ cls: string; range: string; buckets: number[] }>(`/api/hourly?cls=${cls}`);

export interface Busiest {
  label: string;
  count: number;
}
export interface Timeseries {
  bucket: number;
  range: string;
  labels: string[];
  person: number[];
  vehicle: number[];
  totals: { person: number; vehicle: number };
  busiest: { person: Busiest | null; vehicle: Busiest | null };
}
export const getTimeseries = (bucket = 15) =>
  api<Timeseries>(`/api/timeseries?bucket=${bucket}`);

// -- v2: groups / entities / search ---------------------------------------
export interface Group {
  id: number;
  name: string | null;
  kind: "cluster" | "labeled";
  notify: boolean;
  count: number;
  hint?: string | null;   // CLIP zero-shot guess for unnamed clusters
  pending?: number;       // auto-added members awaiting approve/decline
  last_seen?: number | null;   // epoch of the most recent sighting
  samples: string[]; // media urls (need token)
  tag_key?: string | null;
  tag_value?: string | null;
}
export interface Entity {
  id: number;
  label: string | null;
  class: string;
  occurrences: number;   // now = VISITS (continuous presences)
  sightings?: number;    // raw detection/artifact count
  first_seen: number;
  last_seen: number;
  samples: string[];
}
export type ScoredArtifact = Artifact & { score?: number };

export const getGroups = () => api<Group[]>("/api/groups");
export const recluster = () =>
  api<{ clusters: number; artifacts: number }>("/api/groups/recluster", { method: "POST" });
export const createGroupFromText = (name: string, prompt: string) =>
  api<{ group_id: number; members: number }>("/api/groups/from-text", {
    method: "POST", body: JSON.stringify({ name, prompt }),
  });
export const createGroupFromArtifact = (name: string, artifact_id: number) =>
  api<{ group_id: number; members: number }>("/api/groups/from-artifact", {
    method: "POST", body: JSON.stringify({ name, artifact_id }),
  });
export const nameGroup = (id: number, name: string) =>
  api(`/api/groups/${id}/name`, { method: "POST", body: JSON.stringify({ name }) });
export const setGroupNotify = (id: number, notify: boolean) =>
  api(`/api/groups/${id}/notify`, { method: "POST", body: JSON.stringify({ notify }) });
export const deleteGroup = (id: number) => api(`/api/groups/${id}`, { method: "DELETE" });
export const autoClassifyRemaining = (groupId: number) =>
  api<{ ok: boolean; classified: number; dropped: number }>(
    `/api/groups/${groupId}/auto-classify-remaining`, { method: "POST" });
export const getGroupMembers = (id: number, limit = 120, offset = 0) => api<Artifact[]>(`/api/groups/${id}?limit=${limit}&offset=${offset}`);
export const setMemberStatus = (
  gid: number, aid: number, status: "confirmed" | "rejected" | "removed",
) =>
  api<{ ok: boolean; members: number; basis: string }>(
    `/api/groups/${gid}/members/${aid}`,
    { method: "POST", body: JSON.stringify({ status }) },
  );

export const setMembersBatchStatus = (
  gid: number, artifact_ids: number[], status: "confirmed" | "rejected" | "removed",
) =>
  api<{ ok: boolean; members: number; basis: string; matched: number }>(
    `/api/groups/${gid}/members/batch`,
    { method: "POST", body: JSON.stringify({ artifact_ids, status }) },
  );

export const tagArtifacts = (artifact_ids: number[], tags: Array<{ key: string; value: string }>, source_group_id?: number) =>
  api<{ status: string; group_ids: number[] }>("/api/groups/tag", {
    method: "POST", body: JSON.stringify({ artifact_ids, tags, source_group_id }),
  });

export const getTagsAutocomplete = () =>
  api<{ keys: string[]; values: Record<string, string[]> }>("/api/tags/autocomplete");

export const backfillGroup = (id: number, threshold?: number) => {
  const query = threshold !== undefined ? `?threshold=${threshold}` : "";
  return api<{ ok: boolean; matched: number }>(`/api/groups/${id}/backfill${query}`, { method: "POST" });
};

export const searchArtifacts = (q: string) =>
  api<ScoredArtifact[]>(`/api/search?q=${encodeURIComponent(q)}`);
export const getSimilar = (id: number) =>
  api<ScoredArtifact[]>(`/api/artifacts/${id}/similar`);
export const labelRegion = (id: number, box: number[], label: string, rank = 0) =>
  api<{ group_id: number; label: string; created: boolean; seeds: number; matched: number }>(
    `/api/artifacts/${id}/label`,
    { method: "POST", body: JSON.stringify({ box, label, rank }) });

export const getEntities = () => api<Entity[]>("/api/entities");
export const getEntityMembers = (id: number) => api<Artifact[]>(`/api/entities/${id}`);
export const nameEntity = (id: number, name: string) =>
  api(`/api/entities/${id}/name`, { method: "POST", body: JSON.stringify({ name }) });

export const notifyStatus = () =>
  api<{ enabled: boolean; server: string; topic: string }>("/api/notify/status");
export const notifyTest = () => api<{ ok: boolean }>("/api/notify/test", { method: "POST" });

// -- v2: DVR (continuous recording + scrub-back timeline) -----------------
export interface DvrSegment {
  name: string;
  start: number;   // epoch seconds
  duration: number;
  size: number;
}
export interface DvrIndex {
  now: number;
  segments: DvrSegment[];
  retention_h: number;
}
export interface DvrTimeline {
  start: number;
  end: number;
  bucket_s: number;
  person: number[];
  vehicle: number[];
  motion: number[];   // movement energy per bucket (what the scrobbler shows)
}
export const getDvrIndex = () => api<DvrIndex>("/api/dvr/index");
export const getDvrTimeline = (hours = 24, bucket_min = 10) =>
  api<DvrTimeline>(`/api/dvr/timeline?hours=${hours}&bucket_min=${bucket_min}`);

export interface LibraryClip { name: string; size: number; saved_at: number; }
export const getLibrary = () => api<LibraryClip[]>("/api/dvr/library");
export const saveClip = (start: number, end: number, name: string) =>
  api<{ name: string; duration: number; size: number }>("/api/dvr/save", {
    method: "POST", body: JSON.stringify({ start, end, name }),
  });
export const deleteClip = (name: string) =>
  api(`/api/dvr/library/${encodeURIComponent(name)}`, { method: "DELETE" });

export interface RecurringEntity {
  entity_id: number; label: string | null; class: string;
  count: number; first: number; last: number; samples: string[];
}
export const getRecurring = () => api<RecurringEntity[]>("/api/stats/recurring");

export interface StatsSummary {
  range: string;
  sightings: number;
  mix: { class: string; count: number }[];
  unique: { person: number; vehicle: number; other: number };
  cat_sightings: { person: number; vehicle: number; other: number };
}
export const getStatsSummary = () => api<StatsSummary>("/api/stats/summary");

export interface Sighting { artifact_id: number; start: number; end: number; class: string; url: string; detections?: number; }
export const getEntityTimeline = (id: number) =>
  api<Sighting[]>(`/api/entities/${id}/timeline`);
export const splitEntityMember = (artifactId: number) =>
  api<{ ok: boolean; new_entity: number; dislinks: number }>(
    `/api/entities/split/${artifactId}`, { method: "POST" });

// media & stream need the token in the URL (<img> can't send headers)
export const withToken = (url: string) =>
  `${url}${url.includes("?") ? "&" : "?"}token=${getToken()}`;
export const streamUrl = (overlay = true) =>
  withToken(`/api/stream?overlay=${overlay ? 1 : 0}`);
export const segmentUrl = (name: string) =>
  withToken(`/api/dvr/segment/${encodeURIComponent(name)}`);
export const dvrPlayUrl = (start: number) =>
  withToken(`/api/dvr/play?start=${Math.floor(start)}`);
export const libraryClipUrl = (name: string) =>
  withToken(`/api/dvr/library/${encodeURIComponent(name)}`);
