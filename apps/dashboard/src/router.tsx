import { createBrowserRouter, NavLink, Outlet } from "react-router-dom";

function HydrateFallback() {
  return (
    <main className="main">
      <div className="page-stack">
        <div className="panel">Loading dashboard…</div>
      </div>
    </main>
  );
}

function Shell() {
  return (
    <div className="shell">
      <aside className="sidebar">
        <div className="sidebar__brand">
          <div className="sidebar__eyebrow">COS435 • CityLearn</div>
          <h1>Benchmark desk</h1>
          <p>Launch runs, review evidence, compare results.</p>
        </div>

        <nav className="sidebar__nav">
          <NavLink to="/" end>
            Overview
          </NavLink>
          <NavLink to="/monitor">Live</NavLink>
          <NavLink to="/runs">Runs</NavLink>
          <NavLink to="/compare">Compare</NavLink>
          <NavLink to="/artifacts">Artifacts</NavLink>
        </nav>
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
      hydrateFallbackElement: <HydrateFallback />,
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
