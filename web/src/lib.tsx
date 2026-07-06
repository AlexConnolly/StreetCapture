import { useEffect, useRef, useState } from "react";

// Poll a promise-returning fn on an interval; returns latest data + error.
export function usePoll<T>(fn: () => Promise<T>, ms: number, onAuthErr?: () => void) {
  const [data, setData] = useState<T | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const saved = useRef(fn);
  saved.current = fn;
  useEffect(() => {
    let alive = true;
    const tick = async () => {
      try {
        const d = await saved.current();
        if (alive) {
          setData(d);
          setErr(null);
        }
      } catch (e: any) {
        if (e?.name === "AuthError") onAuthErr?.();
        if (alive) setErr(e?.message || "error");
      }
    };
    tick();
    const id = setInterval(tick, ms);
    return () => {
      alive = false;
      clearInterval(id);
    };
  }, [ms]);
  return { data, err };
}

export function timeAgo(epoch: number): string {
  const s = Math.max(0, Date.now() / 1000 - epoch);
  if (s < 60) return `${Math.floor(s)}s ago`;
  if (s < 3600) return `${Math.floor(s / 60)}m ago`;
  if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
  return `${Math.floor(s / 86400)}d ago`;
}

export function clock(epoch: number): string {
  return new Date(epoch * 1000).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
}

export const VEHICLES = new Set([
  "car", "truck", "bus", "motorbike", "motorcycle", "bicycle", "train",
]);
export function cat(cls: string): "person" | "vehicle" | "other" {
  if (cls === "person") return "person";
  if (VEHICLES.has(cls)) return "vehicle";
  return "other";
}
export const CAT_COLOR: Record<string, string> = {
  person: "text-emerald-400",
  vehicle: "text-amber-400",
  other: "text-zinc-400",
};
export const CAT_DOT: Record<string, string> = {
  person: "bg-emerald-400",
  vehicle: "bg-amber-400",
  other: "bg-zinc-400",
};

// -- inline icons ----------------------------------------------------------
type IP = { className?: string };
const svg = (path: React.ReactNode) => (p: IP) =>
  (
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"
      strokeLinecap="round" strokeLinejoin="round" className={p.className}>
      {path}
    </svg>
  );
export const IconLive = svg(<><circle cx="12" cy="12" r="3" /><path d="M12 5v.01M12 19v.01M5 12h.01M19 12h.01" /><circle cx="12" cy="12" r="8" /></>);
export const IconGrid = svg(<><rect x="3" y="3" width="7" height="7" rx="1" /><rect x="14" y="3" width="7" height="7" rx="1" /><rect x="3" y="14" width="7" height="7" rx="1" /><rect x="14" y="14" width="7" height="7" rx="1" /></>);
export const IconBell = svg(<><path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9" /><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0" /></>);
export const IconChat = svg(<><path d="M21 15a2 2 0 0 1-2 2H7l-4 4V5a2 2 0 0 1 2-2h14a2 2 0 0 1 2 2z" /></>);
export const IconChart = svg(<><path d="M3 3v18h18" /><rect x="7" y="10" width="3" height="7" /><rect x="12" y="6" width="3" height="11" /><rect x="17" y="13" width="3" height="4" /></>);
export const IconLogout = svg(<><path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" /><path d="M16 17l5-5-5-5" /><path d="M21 12H9" /></>);
export const IconSend = svg(<><path d="M22 2 11 13" /><path d="M22 2 15 22l-4-9-9-4z" /></>);
export const IconBack = svg(<><path d="M19 12H5" /><path d="M12 19l-7-7 7-7" /></>);
export const IconGroups = svg(<><circle cx="8" cy="8" r="4" /><circle cx="17" cy="9" r="3" /><circle cx="11" cy="17" r="4" /></>);
export const IconSearch = svg(<><circle cx="11" cy="11" r="7" /><path d="M21 21l-4.3-4.3" /></>);
export const IconBellRing = svg(<><path d="M6 8a6 6 0 0 1 12 0c0 7 3 9 3 9H3s3-2 3-9" /><path d="M10.3 21a1.94 1.94 0 0 0 3.4 0" /><path d="M4 3 2 5M20 3l2 2" /></>);
export const IconTrash = svg(<><path d="M3 6h18" /><path d="M19 6l-1 14a2 2 0 0 1-2 2H8a2 2 0 0 1-2-2L5 6" /><path d="M10 11v6M14 11v6" /><path d="M9 6V4a1 1 0 0 1 1-1h4a1 1 0 0 1 1 1v2" /></>);
