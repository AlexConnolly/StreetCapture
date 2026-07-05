import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { ask } from "../api";
import { useAuth } from "../auth";
import { Shell } from "../components/Shell";
import { IconSend } from "../lib";

const EXAMPLES = [
  "how many vehicles today?",
  "quietest time for foot traffic?",
  "how many people this week?",
  "show vehicles between 8am and 10am today",
  "how often do vehicles appear today?",
];

interface QA { q: string; a: string; }

export default function Ask() {
  const nav = useNavigate();
  const { logout } = useAuth();
  const [q, setQ] = useState("");
  const [busy, setBusy] = useState(false);
  const [history, setHistory] = useState<QA[]>([]);

  async function run(question: string) {
    const text = question.trim();
    if (!text) return;
    setBusy(true);
    setQ("");
    try {
      const r = await ask(text);
      setHistory((h) => [{ q: text, a: r.answer }, ...h]);
    } catch (e: any) {
      if (e?.name === "AuthError") { logout(); nav("/"); return; }
      setHistory((h) => [{ q: text, a: "error: " + (e?.message || "failed") }, ...h]);
    } finally {
      setBusy(false);
    }
  }

  return (
    <Shell title="Ask">
      <form onSubmit={(e) => { e.preventDefault(); run(q); }} className="mb-3 flex gap-2">
        <input
          value={q}
          onChange={(e) => setQ(e.target.value)}
          placeholder="Ask about your street…"
          className="flex-1 rounded-xl border border-line bg-panel px-4 py-3 text-[16px] outline-none focus:border-accent"
        />
        <button disabled={busy} className="rounded-xl bg-accent px-4 text-black disabled:opacity-40">
          <IconSend className="h-5 w-5" />
        </button>
      </form>

      {history.length === 0 && (
        <div className="flex flex-wrap gap-2">
          {EXAMPLES.map((ex) => (
            <button key={ex} onClick={() => run(ex)}
              className="rounded-full border border-line bg-panel px-3 py-1.5 text-[13px] text-zinc-400">
              {ex}
            </button>
          ))}
        </div>
      )}

      <div className="mt-2 space-y-3">
        {history.map((qa, i) => (
          <div key={i} className="rounded-xl border border-line bg-panel p-3.5">
            <div className="mb-1.5 text-sm font-medium text-zinc-300">{qa.q}</div>
            <div className="whitespace-pre-wrap text-[13px] leading-relaxed text-accent">{qa.a}</div>
          </div>
        ))}
      </div>
    </Shell>
  );
}
