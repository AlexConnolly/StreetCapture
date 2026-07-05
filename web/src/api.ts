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
export interface Stats {
  day: string;
  active: number;
  active_by_cat: Record<string, number>;
  daily: Record<string, number>;
  artifacts: Record<string, number>;
  events: string[];
  fps: number;
  online: boolean;
  uptime_s: number;
  faiss_vectors: number;
  embed_model: string;
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

// media & stream need the token in the URL (<img> can't send headers)
export const withToken = (url: string) =>
  `${url}${url.includes("?") ? "&" : "?"}token=${getToken()}`;
export const streamUrl = () => withToken("/api/stream");
