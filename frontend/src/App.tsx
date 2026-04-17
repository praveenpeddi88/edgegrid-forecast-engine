import { useEffect, useMemo, useRef, useState } from "react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  forceCenter,
  forceLink,
  forceManyBody,
  forceSimulation,
  forceX,
  forceY,
} from "d3-force";
import { select } from "d3-selection";
import { zoom, zoomIdentity } from "d3-zoom";
import {
  Activity,
  ArrowLeft,
  Battery,
  Download,
  Factory,
  Gauge,
  Grid,
  LayoutDashboard,
  Network as NetworkIcon,
  Zap,
} from "lucide-react";
import {
  CommercialResponse,
  DispatchResponse,
  FLSQuote,
  GraphNode,
  MeterForecastResponse,
  NetworkGraph,
  Portfolio,
  api,
} from "./api";

// ────────────────────── routing (minimal state machine) ──────────────────────

type Route =
  | { name: "home" }
  | { name: "meter"; msn: string }
  | { name: "dispatch"; substationId: string }
  | { name: "commercial"; substationId: string }
  | { name: "portfolio" };

// ────────────────────── top-level App ────────────────────────────────────────

export default function App() {
  const [route, setRoute] = useState<Route>({ name: "home" });
  const [version, setVersion] = useState<string>("");

  useEffect(() => {
    api.modelVersion().then((v) => setVersion(v.model_version)).catch(() => {});
  }, []);

  return (
    <div className="min-h-screen flex flex-col">
      <TopBar route={route} setRoute={setRoute} />
      <main className="flex-1 min-h-0">
        {route.name === "home" && <NetworkHome setRoute={setRoute} />}
        {route.name === "meter" && (
          <MeterDetail msn={route.msn} back={() => setRoute({ name: "home" })} />
        )}
        {route.name === "dispatch" && (
          <DispatchConsole
            substationId={route.substationId}
            goCommercial={() =>
              setRoute({ name: "commercial", substationId: route.substationId })
            }
            back={() => setRoute({ name: "home" })}
          />
        )}
        {route.name === "commercial" && (
          <CommercialBrief
            substationId={route.substationId}
            back={() => setRoute({ name: "dispatch", substationId: route.substationId })}
          />
        )}
        {route.name === "portfolio" && (
          <PortfolioView back={() => setRoute({ name: "home" })} />
        )}
      </main>
      <Footer modelVersion={version} />
    </div>
  );
}

function TopBar({ route, setRoute }: { route: Route; setRoute: (r: Route) => void }) {
  return (
    <header className="flex items-center justify-between px-6 py-3 border-b border-zinc-800 bg-zinc-950/80 backdrop-blur">
      <button
        onClick={() => setRoute({ name: "home" })}
        className="flex items-center gap-2 text-teal-500 hover:text-teal-600 transition-colors duration-fast font-semibold"
      >
        <Zap className="w-5 h-5" />
        EdgeGrid <span className="text-zinc-500 font-normal">Dispatch Console</span>
      </button>
      <nav className="flex items-center gap-1 text-sm">
        <NavBtn
          icon={<NetworkIcon className="w-4 h-4" />}
          label="Network"
          active={route.name === "home"}
          onClick={() => setRoute({ name: "home" })}
        />
        <NavBtn
          icon={<LayoutDashboard className="w-4 h-4" />}
          label="Portfolio"
          active={route.name === "portfolio"}
          onClick={() => setRoute({ name: "portfolio" })}
        />
      </nav>
    </header>
  );
}

function NavBtn(props: {
  icon: React.ReactNode;
  label: string;
  active?: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={props.onClick}
      className={
        "px-3 py-1.5 rounded-md flex items-center gap-2 transition-colors duration-fast " +
        (props.active ? "bg-zinc-800 text-white" : "text-zinc-400 hover:text-white hover:bg-zinc-900")
      }
    >
      {props.icon}
      {props.label}
    </button>
  );
}

function Footer({ modelVersion }: { modelVersion: string }) {
  return (
    <footer className="flex items-center justify-between px-6 py-2 text-xs text-zinc-500 border-t border-zinc-900">
      <span>Clean, commercially-optimal energy delivery at the last mile.</span>
      <span>
        model <code className="text-zinc-300">{modelVersion || "loading…"}</code>
      </span>
    </footer>
  );
}

// ────────────────────── Screen: Network Home (D3 force graph) ────────────────

function NetworkHome({ setRoute }: { setRoute: (r: Route) => void }) {
  const [graph, setGraph] = useState<NetworkGraph | null>(null);
  const [selected, setSelected] = useState<GraphNode | null>(null);
  const svgRef = useRef<SVGSVGElement | null>(null);
  const containerRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    api.network().then(setGraph).catch((e) => console.error(e));
  }, []);

  useEffect(() => {
    if (!graph || !svgRef.current || !containerRef.current) return;
    const svg = select(svgRef.current);
    svg.selectAll("*").remove();
    const width = containerRef.current.clientWidth;
    const height = containerRef.current.clientHeight;
    const g = svg.append("g");

    // Zoom/pan
    const z = zoom<SVGSVGElement, unknown>()
      .scaleExtent([0.2, 4])
      .on("zoom", (ev) => g.attr("transform", ev.transform));
    svg.call(z as any).call(z.transform as any, zoomIdentity);

    // Copy because d3-force mutates
    const nodes = graph.nodes.map((n) => ({ ...n }));
    const links = graph.edges.map((e) => ({
      source: e.source,
      target: e.target,
      edge_type: e.edge_type,
    }));

    const kindColor: Record<string, string> = {
      substation: "#0E7C7B",
      meter: "#38BDF8",
      feeder: "#A78BFA",
      bess: "#FB923C",
      solar: "#FBBF24",
      ci_load: "#F472B6",
      weather: "#64748B",
      iex: "#94A3B8",
    };
    const kindSize: Record<string, number> = {
      substation: 14, meter: 5, feeder: 7, bess: 10,
      solar: 8, ci_load: 7, weather: 6, iex: 8,
    };
    const meterColor = (n: any) => {
      if (n.kind !== "meter") return kindColor[n.kind] || "#64748B";
      const m = n.holdout_mape ?? 10;
      if (m < 8) return "#10B981";
      if (m < 12) return "#F59E0B";
      return "#EF4444";
    };

    const link = g.append("g").attr("stroke", "#27272A").attr("stroke-opacity", 0.6)
      .selectAll("line").data(links).join("line").attr("stroke-width", 1);

    const node = g.append("g").selectAll("circle")
      .data(nodes)
      .join("circle")
      .attr("r", (d: any) => kindSize[d.kind] || 5)
      .attr("fill", meterColor)
      .attr("stroke", "#09090B")
      .attr("stroke-width", 1.5)
      .style("cursor", "pointer")
      .on("click", (_e, d: any) => {
        setSelected(d);
      });

    node.append("title").text((d: any) => `${d.label} (${d.kind})`);

    const sim = forceSimulation(nodes as any)
      .force("link", forceLink(links as any).id((d: any) => d.id).distance(40).strength(0.4))
      .force("charge", forceManyBody().strength(-120))
      .force("center", forceCenter(width / 2, height / 2))
      .force("x", forceX(width / 2).strength(0.03))
      .force("y", forceY(height / 2).strength(0.03));

    sim.on("tick", () => {
      link
        .attr("x1", (d: any) => d.source.x)
        .attr("y1", (d: any) => d.source.y)
        .attr("x2", (d: any) => d.target.x)
        .attr("y2", (d: any) => d.target.y);
      node
        .attr("cx", (d: any) => d.x)
        .attr("cy", (d: any) => d.y);
    });

    return () => void sim.stop();
  }, [graph]);

  return (
    <div className="grid grid-cols-[1fr_340px] gap-0 h-[calc(100vh-96px)]">
      <div ref={containerRef} className="relative bg-zinc-950">
        <svg ref={svgRef} width="100%" height="100%" />
        <Legend />
      </div>
      <aside className="border-l border-zinc-800 bg-zinc-900/40 p-5 overflow-auto">
        {graph ? (
          selected ? (
            <SidePanel node={selected} setRoute={setRoute} />
          ) : (
            <div>
              <h2 className="text-lg font-semibold mb-1">The EdgeGrid network</h2>
              <p className="muted text-sm mb-4">
                {graph.meta.n_nodes} nodes · {graph.meta.n_edges} edges. Click any
                node to drill in.
              </p>
              <GraphStats graph={graph} />
            </div>
          )
        ) : (
          <p className="muted">Loading graph…</p>
        )}
      </aside>
    </div>
  );
}

function Legend() {
  const items = [
    { c: "#0E7C7B", label: "Substation" },
    { c: "#38BDF8", label: "Meter" },
    { c: "#A78BFA", label: "Feeder" },
    { c: "#FB923C", label: "BESS" },
    { c: "#FBBF24", label: "Solar" },
    { c: "#F472B6", label: "C&I Load" },
  ];
  return (
    <div className="absolute top-3 left-3 card px-3 py-2 text-xs">
      <div className="muted mb-1">Legend</div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-1">
        {items.map((i) => (
          <div key={i.label} className="flex items-center gap-1.5">
            <span
              className="w-2.5 h-2.5 rounded-full inline-block"
              style={{ background: i.c }}
            />
            <span>{i.label}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function GraphStats({ graph }: { graph: NetworkGraph }) {
  const counts = graph.nodes.reduce<Record<string, number>>((acc, n) => {
    acc[n.kind] = (acc[n.kind] || 0) + 1;
    return acc;
  }, {});
  return (
    <dl className="space-y-2 text-sm">
      {Object.entries(counts).sort().map(([k, n]) => (
        <div key={k} className="flex justify-between border-b border-zinc-800 py-1.5">
          <dt className="muted capitalize">{k.replace("_", " ")}</dt>
          <dd>{n}</dd>
        </div>
      ))}
    </dl>
  );
}

function SidePanel({ node, setRoute }: { node: GraphNode; setRoute: (r: Route) => void }) {
  return (
    <div>
      <button
        onClick={() => setRoute({ name: "home" })}
        className="text-xs muted hover:text-white mb-3 flex items-center gap-1"
      >
        <ArrowLeft className="w-3 h-3" /> deselect
      </button>
      <h2 className="text-lg font-semibold">{node.label}</h2>
      <p className="muted text-sm mb-4 capitalize">{node.kind.replace("_", " ")}</p>
      <dl className="space-y-1.5 text-sm mb-5">
        {Object.entries(node)
          .filter(([k]) => !["id", "label", "kind"].includes(k))
          .filter(([, v]) => v !== null && v !== undefined && v !== "")
          .slice(0, 10)
          .map(([k, v]) => (
            <div key={k} className="flex justify-between gap-2">
              <dt className="muted text-xs mt-0.5">{k}</dt>
              <dd className="text-right max-w-[200px] truncate">{String(v)}</dd>
            </div>
          ))}
      </dl>
      {node.kind === "meter" && (
        <button
          className="btn-primary w-full"
          onClick={() => setRoute({ name: "meter", msn: (node as any).msn })}
        >
          <Gauge className="w-4 h-4" /> View forecast
        </button>
      )}
      {node.kind === "substation" && (
        <div className="flex flex-col gap-2">
          <button
            className="btn-primary w-full"
            onClick={() => setRoute({ name: "dispatch", substationId: node.id })}
          >
            <Activity className="w-4 h-4" /> Open dispatch
          </button>
          <button
            className="btn-ghost w-full"
            onClick={() => setRoute({ name: "commercial", substationId: node.id })}
          >
            <Factory className="w-4 h-4" /> Commercial brief
          </button>
        </div>
      )}
    </div>
  );
}

// ────────────────────── Screen: Meter Detail ─────────────────────────────────

function MeterDetail({ msn, back }: { msn: string; back: () => void }) {
  const [fc, setFc] = useState<MeterForecastResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.meterForecast(msn, 48)
      .then(setFc)
      .catch((e) => setErr(String(e)));
  }, [msn]);

  const data = useMemo(() => {
    if (!fc) return [];
    return fc.rows.map((r, i) => ({
      t: r.timestamp ? r.timestamp.slice(11, 16) : `${i}`,
      forecast: r.forecast_wh,
      low: r.confidence_low,
      high: r.confidence_high,
      band: r.confidence_high - r.confidence_low,
    }));
  }, [fc]);

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-5">
      <button onClick={back} className="flex items-center gap-1 muted hover:text-white text-sm">
        <ArrowLeft className="w-4 h-4" /> back to network
      </button>
      <h1 className="text-2xl font-semibold">Meter {msn}</h1>
      {err && (
        <div className="card p-4 border-red-900 bg-red-950/30 text-red-300 text-sm">
          <b>predict() failed.</b> {err}. Synthesize a forecast at the
          substation aggregate, or ensure <code>models/v4/{msn}.joblib</code> exists
          and the raw parquets under <code>data/</code> are loaded.
        </div>
      )}
      {fc && (
        <>
          <div className="card p-4">
            <h2 className="text-sm muted uppercase tracking-wider mb-2">
              48-interval forecast (Wh per 30-min block)
            </h2>
            <ResponsiveContainer width="100%" height={320}>
              <AreaChart data={data}>
                <defs>
                  <linearGradient id="band" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="0%" stopColor="#0E7C7B" stopOpacity={0.3} />
                    <stop offset="100%" stopColor="#0E7C7B" stopOpacity={0.05} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#27272A" strokeDasharray="3 3" />
                <XAxis dataKey="t" stroke="#71717A" fontSize={11} />
                <YAxis stroke="#71717A" fontSize={11} />
                <Tooltip
                  contentStyle={{ background: "#18181B", border: "1px solid #27272A" }}
                />
                <Area
                  type="monotone"
                  dataKey="high"
                  stroke="none"
                  fill="url(#band)"
                  stackId="a"
                />
                <Line type="monotone" dataKey="forecast" stroke="#0E7C7B" strokeWidth={2} dot={false} />
              </AreaChart>
            </ResponsiveContainer>
          </div>
          <div className="text-xs muted">
            Model version <code>{fc.model_version}</code> · as of {fc.as_of}
          </div>
        </>
      )}
    </div>
  );
}

// ────────────────────── Screen: Dispatch Console ─────────────────────────────

function DispatchConsole({
  substationId,
  goCommercial,
  back,
}: {
  substationId: string;
  goCommercial: () => void;
  back: () => void;
}) {
  const [d, setD] = useState<DispatchResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.substationDispatch(substationId).then(setD).catch((e) => setErr(String(e)));
  }, [substationId]);

  const chartData = useMemo(() => {
    if (!d) return [];
    return d.schedule.map((r) => ({
      t: r.timestamp.slice(11, 16),
      demand: r.demand_kwh,
      grid: r.grid_import_kwh,
      charge: r.bess_charge_kwh,
      discharge: r.bess_discharge_kwh,
      soc: r.soc_pct,
      price: r.iex_price_inr,
    }));
  }, [d]);

  return (
    <div className="p-6 max-w-7xl mx-auto space-y-5">
      <div className="flex items-center justify-between">
        <button onClick={back} className="flex items-center gap-1 muted hover:text-white text-sm">
          <ArrowLeft className="w-4 h-4" /> network
        </button>
        <button onClick={goCommercial} className="btn-ghost">
          <Factory className="w-4 h-4" /> Commercial brief
        </button>
      </div>
      <h1 className="text-2xl font-semibold">Dispatch · {substationId}</h1>
      {err && <ErrorCard msg={err} />}
      {d && (
        <>
          <div className="grid grid-cols-4 gap-3">
            <Stat label="Net benefit (48h)" value={`₹${Math.round(d.totals.net_benefit_inr).toLocaleString()}`} />
            <Stat label="Peak kVA" value={`${d.totals.peak_kva.toFixed(0)}`} />
            <Stat label="Cycles" value={d.totals.cycles.toFixed(2)} />
            <Stat label="Solver" value={d.solver_status} />
          </div>
          <div className="grid grid-cols-[2fr_1fr] gap-5">
            <div className="card p-4">
              <h2 className="text-sm muted uppercase tracking-wider mb-2">
                48h × 15-min schedule — grid import, BESS charge/discharge, SOC
              </h2>
              <ResponsiveContainer width="100%" height={280}>
                <LineChart data={chartData}>
                  <CartesianGrid stroke="#27272A" strokeDasharray="3 3" />
                  <XAxis dataKey="t" stroke="#71717A" fontSize={10} interval={15} />
                  <YAxis yAxisId="l" stroke="#71717A" fontSize={10} />
                  <YAxis yAxisId="r" orientation="right" stroke="#FB923C" fontSize={10} domain={[0, 100]} />
                  <Tooltip contentStyle={{ background: "#18181B", border: "1px solid #27272A" }} />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <Line yAxisId="l" type="stepAfter" dataKey="grid" stroke="#94A3B8" dot={false} strokeWidth={1.3} />
                  <Line yAxisId="l" type="stepAfter" dataKey="charge" stroke="#38BDF8" dot={false} strokeWidth={1.3} />
                  <Line yAxisId="l" type="stepAfter" dataKey="discharge" stroke="#F472B6" dot={false} strokeWidth={1.3} />
                  <Line yAxisId="r" type="monotone" dataKey="soc" stroke="#FB923C" dot={false} strokeWidth={2} />
                </LineChart>
              </ResponsiveContainer>
            </div>
            <div className="card p-4 overflow-auto max-h-[360px]">
              <h2 className="text-sm muted uppercase tracking-wider mb-2">Audit column</h2>
              <ul className="space-y-2 text-[13px]">
                {d.schedule
                  .filter((r) => r.action !== "hold")
                  .slice(0, 30)
                  .map((r) => (
                    <li key={r.timestamp} className="border-l-2 border-teal-500 pl-2 py-0.5">
                      <code className="text-zinc-200">{r.audit_string}</code>
                    </li>
                  ))}
              </ul>
            </div>
          </div>
        </>
      )}
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div className="card p-3">
      <div className="text-xs muted uppercase tracking-wider">{label}</div>
      <div className="text-xl font-semibold mt-1">{value}</div>
    </div>
  );
}

function ErrorCard({ msg }: { msg: string }) {
  return (
    <div className="card p-4 border-red-900 bg-red-950/30 text-red-300 text-sm">
      <b>Backend error.</b> {msg}
    </div>
  );
}

// ────────────────────── Screen: Commercial Brief ─────────────────────────────

function CommercialBrief({ substationId, back }: { substationId: string; back: () => void }) {
  const [c, setC] = useState<CommercialResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [quote, setQuote] = useState<FLSQuote | null>(null);
  const [buyerKw, setBuyerKw] = useState<number>(500);

  useEffect(() => {
    api.substationCommercial(substationId).then(setC).catch((e) => setErr(String(e)));
  }, [substationId]);

  const onQuote = async () => {
    setQuote(
      await api.substationFlsQuote(substationId, {
        contract_id: `FLS-${substationId}-${Date.now()}`,
        buyer_kw: buyerKw,
        window_start: "18:00",
        window_end: "22:00",
      })
    );
  };

  const heatRows = useMemo(() => {
    if (!c) return [];
    return c.heatmap.capacities_mwh.flatMap((cap, i) =>
      c.heatmap.durations_h.map((dur, j) => ({
        cap, dur,
        irr: c.heatmap.irr_pct[i][j],
        payback: c.heatmap.payback_years[i][j],
      }))
    );
  }, [c]);

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-5">
      <button onClick={back} className="flex items-center gap-1 muted hover:text-white text-sm">
        <ArrowLeft className="w-4 h-4" /> dispatch
      </button>
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-semibold">Commercial brief · {substationId}</h1>
        <a
          href={api.substationBriefUrl(substationId)}
          target="_blank" rel="noreferrer"
          className="btn-primary"
        >
          <Download className="w-4 h-4" /> Export PDF (print view)
        </a>
      </div>
      {err && <ErrorCard msg={err} />}
      {c && (
        <>
          <div className="grid grid-cols-4 gap-3">
            <Stat label="Today net" value={`₹${Math.round(c.daily_net_benefit_inr).toLocaleString()}`} />
            <Stat label="Annual (proj.)" value={`₹${(c.annual_net_benefit_inr / 1e5).toFixed(1)} L`} />
            <Stat label="IRR (ref)" value={`${c.reference.irr_pct.toFixed(1)}%`} />
            <Stat label="Payback" value={`${c.reference.payback_years.toFixed(1)} yr`} />
          </div>

          <div className="card p-4">
            <h2 className="text-sm muted uppercase tracking-wider mb-3">
              IRR heatmap — BESS capacity × duration
            </h2>
            <div className="overflow-auto">
              <table className="text-sm">
                <thead>
                  <tr className="muted">
                    <th className="px-3 py-1.5 text-left">MWh \ Hours</th>
                    {c.heatmap.durations_h.map((d) => (
                      <th key={d} className="px-3 py-1.5">{d}h</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {c.heatmap.capacities_mwh.map((cap, i) => (
                    <tr key={cap}>
                      <td className="px-3 py-1.5 muted">{cap} MWh</td>
                      {c.heatmap.durations_h.map((_, j) => {
                        const irr = c.heatmap.irr_pct[i][j];
                        const bg = irr > 20 ? "bg-emerald-900/50"
                          : irr > 12 ? "bg-emerald-900/25"
                          : irr > 0 ? "bg-zinc-900"
                          : "bg-red-950/40";
                        return (
                          <td key={j} className={`px-3 py-1.5 text-center ${bg}`}>
                            <div className="font-semibold">{isFinite(irr) ? `${irr.toFixed(1)}%` : "—"}</div>
                            <div className="text-[10px] muted">
                              {c.heatmap.payback_years[i][j].toFixed(1)}y
                            </div>
                          </td>
                        );
                      })}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>

          <div className="card p-4">
            <h2 className="text-sm muted uppercase tracking-wider mb-3">FLS quote generator</h2>
            <div className="flex items-end gap-3 mb-4">
              <label className="text-sm">
                <div className="muted text-xs mb-1">Buyer kW</div>
                <input
                  type="number"
                  value={buyerKw}
                  onChange={(e) => setBuyerKw(Number(e.target.value))}
                  className="bg-zinc-900 border border-zinc-700 rounded px-2 py-1 w-28"
                />
              </label>
              <div className="text-xs muted pb-1.5">
                Window: 18:00 – 22:00 weekdays, 12-month tenor
              </div>
              <button className="btn-primary" onClick={onQuote}>
                Generate quote
              </button>
            </div>
            {quote && (
              <div className="grid grid-cols-3 gap-3">
                <Stat label="Offered ₹/kWh" value={`₹${quote.offered_price_inr_per_kwh}`} />
                <Stat label="Firmness" value={`${quote.firmness_pct}%`} />
                <Stat label="Peak MAPE" value={`${quote.underlying_mape_pct}%`} />
                <div className="col-span-3 text-sm muted">{quote.rationale}</div>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}

// ────────────────────── Screen: Portfolio ────────────────────────────────────

function PortfolioView({ back }: { back: () => void }) {
  const [p, setP] = useState<Portfolio | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    api.portfolio().then(setP).catch((e) => setErr(String(e)));
  }, []);

  const healthData = useMemo(() => {
    if (!p) return [];
    return [
      { name: "green", n: p.meter_health.green, fill: "#10B981" },
      { name: "amber", n: p.meter_health.amber, fill: "#F59E0B" },
      { name: "red", n: p.meter_health.red, fill: "#EF4444" },
    ];
  }, [p]);

  return (
    <div className="p-6 max-w-6xl mx-auto space-y-5">
      <button onClick={back} className="flex items-center gap-1 muted hover:text-white text-sm">
        <ArrowLeft className="w-4 h-4" /> network
      </button>
      <h1 className="text-2xl font-semibold">Portfolio</h1>
      {err && <ErrorCard msg={err} />}
      {p && (
        <>
          <div className="grid grid-cols-4 gap-3">
            <Stat label="Substations" value={`${p.n_substations}`} />
            <Stat label="Meters" value={`${p.n_meters}`} />
            <Stat label="BESS units" value={`${p.n_bess}`} />
            <Stat label="Model" value={p.model_version} />
          </div>
          <div className="grid grid-cols-2 gap-5">
            <div className="card p-4">
              <h2 className="text-sm muted uppercase tracking-wider mb-2">Meter health (MAPE)</h2>
              <ResponsiveContainer width="100%" height={240}>
                <BarChart data={healthData}>
                  <CartesianGrid stroke="#27272A" strokeDasharray="3 3" />
                  <XAxis dataKey="name" stroke="#71717A" />
                  <YAxis stroke="#71717A" />
                  <Tooltip contentStyle={{ background: "#18181B", border: "1px solid #27272A" }} />
                  <Bar dataKey="n" />
                </BarChart>
              </ResponsiveContainer>
            </div>
            <div className="card p-4">
              <h2 className="text-sm muted uppercase tracking-wider mb-2">Substations</h2>
              <table className="text-sm w-full">
                <thead className="muted text-xs">
                  <tr>
                    <th className="text-left py-1">Substation</th>
                    <th className="text-right py-1">Meters</th>
                    <th className="text-right py-1">BESS</th>
                    <th className="text-right py-1">₹/kWh landed</th>
                  </tr>
                </thead>
                <tbody>
                  {p.substations.map((s) => (
                    <tr key={s.id} className="border-t border-zinc-800">
                      <td className="py-1.5">
                        <div>{s.label}</div>
                        <code className="text-[10px] muted">{s.id}</code>
                      </td>
                      <td className="text-right">{s.n_meters}</td>
                      <td className="text-right">{s.n_bess}</td>
                      <td className="text-right">₹{s.landed_cost_inr_per_kwh.toFixed(2)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
