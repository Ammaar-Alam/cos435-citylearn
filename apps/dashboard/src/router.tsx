import { createBrowserRouter, NavLink, Outlet } from "react-router-dom";

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
          <NavLink to="/monitor">Monitor</NavLink>
          <NavLink to="/runs">Runs</NavLink>
          <NavLink to="/compare">Compare</NavLink>
          <NavLink to="/artifacts">Imports</NavLink>
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

export const router = createBrowserRouter(
  [
    {
      path: "/",
      element: <Shell />,
      children: [
        {
          index: true,
          lazy: () => import("./routes/overview").then((mod) => ({ Component: mod.OverviewPage })),
        },
        {
          path: "monitor",
          lazy: () => import("./routes/monitor").then((mod) => ({ Component: mod.MonitorPage })),
        },
        {
          path: "runs",
          lazy: () => import("./routes/runs").then((mod) => ({ Component: mod.RunsPage })),
        },
        {
          path: "runs/:runId",
          lazy: () => import("./routes/run-detail").then((mod) => ({ Component: mod.RunDetailPage })),
        },
        {
          path: "compare",
          lazy: () => import("./routes/compare").then((mod) => ({ Component: mod.ComparePage })),
        },
        {
          path: "artifacts",
          lazy: () => import("./routes/artifacts").then((mod) => ({ Component: mod.ArtifactsPage })),
        },
      ],
    },
  ],
  { basename: "/dashboard" },
);
