import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../auth";

export default function Login() {
  const { login } = useAuth();
  const nav = useNavigate();
  const [pw, setPw] = useState("");
  const [err, setErr] = useState("");
  const [busy, setBusy] = useState(false);

  async function submit(e: React.FormEvent) {
    e.preventDefault();
    setBusy(true);
    setErr("");
    try {
      await login(pw);
      nav("/live");
    } catch {
      setErr("Wrong password");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="flex h-full flex-col items-center justify-center px-6">
      <div className="w-full max-w-sm">
        <div className="mb-8 text-center">
          <div className="mx-auto mb-3 flex h-14 w-14 items-center justify-center rounded-2xl bg-accent/15 text-accent">
            <span className="text-2xl">◎</span>
          </div>
          <h1 className="text-2xl font-semibold tracking-tight">StreetCapture</h1>
          <p className="mt-1 text-sm text-zinc-500">Visual memory of your street</p>
        </div>
        <form onSubmit={submit} className="space-y-3">
          <input
            type="password"
            value={pw}
            onChange={(e) => setPw(e.target.value)}
            placeholder="Password"
            autoFocus
            className="w-full rounded-xl border border-line bg-panel px-4 py-3.5 text-[16px] outline-none focus:border-accent"
          />
          {err && <p className="text-sm text-red-400">{err}</p>}
          <button
            disabled={busy || !pw}
            className="w-full rounded-xl bg-accent py-3.5 font-semibold text-black disabled:opacity-40"
          >
            {busy ? "…" : "Enter"}
          </button>
        </form>
      </div>
    </div>
  );
}
