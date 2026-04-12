import { createBrowserRouter, NavLink, Outlet } from "react-router-dom";

import { OverviewPage } from "./routes/overview";
import { RunsPage } from "./routes/runs";
import { RunDetailPage } from "./routes/run-detail";
import { ComparePage } from "./routes/compare";

function Shell() {
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="sidebar__brand">
          <div className="sidebar__eyebrow">COS435 • CityLearn</div>
          <h1>Benchmark cockpit</h1>
          <p>One dashboard for launching, watching, and comparing runs.</p>
        </div>

        <nav className="sidebar__nav">
          <NavLink to="/" end>
            Overview
          </NavLink>
          <NavLink to="/runs">Runs</NavLink>
          <NavLink to="/compare">Compare</NavLink>
        </nav>

        <div className="sidebar__note">
          Completed eval runs export simulation data automatically. Held-out evaluation stays out of the default flow.
        </div>
      </aside>

      <main className="main">
        <Outlet />
      </main>
    </div>
  );
}

export const router = createBrowserRouter([
  {
    path: "/",
    element: <Shell />,
    children: [
      { index: true, element: <OverviewPage /> },
      { path: "runs", element: <RunsPage /> },
      { path: "runs/:runId", element: <RunDetailPage /> },
      { path: "compare", element: <ComparePage /> },
    ],
  },
]);
