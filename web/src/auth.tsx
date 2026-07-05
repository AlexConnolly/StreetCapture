import { createContext, useContext, useState, type ReactNode } from "react";
import { clearToken, getToken, login as apiLogin } from "./api";

interface AuthCtx {
  authed: boolean;
  login: (password: string) => Promise<void>;
  logout: () => void;
}

const Ctx = createContext<AuthCtx>(null!);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [authed, setAuthed] = useState(!!getToken());

  async function login(password: string) {
    await apiLogin(password);
    setAuthed(true);
  }
  function logout() {
    clearToken();
    setAuthed(false);
  }

  return <Ctx.Provider value={{ authed, login, logout }}>{children}</Ctx.Provider>;
}

export const useAuth = () => useContext(Ctx);
