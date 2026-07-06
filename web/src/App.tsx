import { Navigate, Route, Routes } from "react-router-dom";
import { useAuth } from "./auth";
import Login from "./pages/Login";
import Live from "./pages/Live";
import Artifacts from "./pages/Artifacts";
import ArtifactDetail from "./pages/ArtifactDetail";
import Events from "./pages/Events";
import Ask from "./pages/Ask";
import Stats from "./pages/Stats";
import Groups, { AllTags } from "./pages/Groups";
import GroupDetail from "./pages/GroupDetail";
import Entities from "./pages/Entities";
import EntityDetail from "./pages/EntityDetail";
import Library from "./pages/Library";

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
      <Route path="/library" element={<Library />} />
      <Route path="/groups" element={<Groups />} />
      <Route path="/groups/all" element={<AllTags />} />
      <Route path="/groups/:id" element={<GroupDetail />} />
      <Route path="/entities" element={<Entities />} />
      <Route path="/entities/:id" element={<EntityDetail />} />
      <Route path="/artifacts" element={<Artifacts />} />
      <Route path="/artifacts/:id" element={<ArtifactDetail />} />
      <Route path="/events" element={<Events />} />
      <Route path="/ask" element={<Ask />} />
      <Route path="/stats" element={<Stats />} />
      <Route path="*" element={<Navigate to="/live" replace />} />
    </Routes>
  );
}
