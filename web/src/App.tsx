import { Navigate, Route, Routes } from "react-router-dom";
import { useAuth } from "./auth";
import Login from "./pages/Login";
import Live from "./pages/Live";
import Artifacts from "./pages/Artifacts";
import ArtifactDetail from "./pages/ArtifactDetail";
import Events from "./pages/Events";
import Ask from "./pages/Ask";
import Stats from "./pages/Stats";

export default function App() {
  const { authed } = useAuth();

  if (!authed) {
    return (
      <Routes>
        <Route path="*" element={<Login />} />
      </Routes>
    );
  }

  return (
    <Routes>
      <Route path="/live" element={<Live />} />
      <Route path="/artifacts" element={<Artifacts />} />
      <Route path="/artifacts/:id" element={<ArtifactDetail />} />
      <Route path="/events" element={<Events />} />
      <Route path="/ask" element={<Ask />} />
      <Route path="/stats" element={<Stats />} />
      <Route path="*" element={<Navigate to="/live" replace />} />
    </Routes>
  );
}
