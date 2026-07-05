import { NavLink, useNavigate } from "react-router-dom";
import type { ReactNode } from "react";
import { useAuth } from "../auth";
import {
  IconBell, IconChart, IconChat, IconGrid, IconLive, IconLogout,
} from "../lib";

const TABS = [
  { to: "/live", label: "Live", Icon: IconLive },
  { to: "/artifacts", label: "Artifacts", Icon: IconGrid },
  { to: "/events", label: "Events", Icon: IconBell },
  { to: "/ask", label: "Ask", Icon: IconChat },
  { to: "/stats", label: "Stats", Icon: IconChart },
];

export function Shell({ title, online, children, flush }: {
  title: string;
  online?: boolean;
  flush?: boolean;
  children: ReactNode;
}) {
  const { logout } = useAuth();
  const nav = useNavigate();
  return (
    <div className="flex h-full flex-col">
      <header className="flex items-center justify-between border-b border-line bg-panel/80 px-4 py-3 backdrop-blur"
        style={{ paddingTop: "max(0.75rem, env(safe-area-inset-top))" }}>
        <div className="flex items-center gap-2">
          <span className="h-2.5 w-2.5 rounded-full" style={{
            background: online === false ? "#71717a" : "#22d3ee",
            boxShadow: online === false ? "none" : "0 0 8px #22d3ee",
          }} />
          <h1 className="text-[15px] font-semibold tracking-wide">{title}</h1>
        </div>
        <button onClick={() => { logout(); nav("/"); }}
          className="text-zinc-500 active:text-zinc-300">
          <IconLogout className="h-5 w-5" />
        </button>
      </header>

      <main className={`flex-1 overflow-y-auto ${flush ? "" : "p-4"}`}>{children}</main>

      <nav className="grid grid-cols-5 border-t border-line bg-panel"
        style={{ paddingBottom: "env(safe-area-inset-bottom)" }}>
        {TABS.map(({ to, label, Icon }) => (
          <NavLink key={to} to={to}
            className={({ isActive }) =>
              `flex flex-col items-center gap-1 py-2.5 text-[11px] ${
                isActive ? "text-accent" : "text-zinc-500"
              }`}>
            <Icon className="h-[22px] w-[22px]" />
            {label}
          </NavLink>
        ))}
      </nav>
    </div>
  );
}
