import { useEffect, useMemo, useRef, useState } from "react";
import {
  Area,
  Bar,
  BarChart,
  CartesianGrid,
  ComposedChart,
  Legend,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";
import {
  ArrowLeft,
  Calendar,
  ChevronRight,
  Gauge,
  Layers,
  LayoutDashboard,
  Network as NetworkIcon,
  Zap,
} from "lucide-react";
import {
  FleetReplayResponse,
  FleetSummary,
  MeterHistoryResponse,
  MeterReplay,
  MeterSummary,
  showcase,
} from "./showcase-api";

// ────────────────────── routing ──────────────────────────────────────────────

type Route =
  | { name: "showcase" }
  | { name: "meter"; msn: string }
  | { name: "network" }
  | { name: "portfolio" };

// ────────────────────── App shell ────────────────────────────────────────────

export default function App() {
  const [route, setRoute] = useState<Route>({ name: "showcase" });
  const [version, setVersion] = useState<string>("");
  const [builtAt, setBuiltAt] = useState<string>("");

  useEffect(() => {
    fetch("/api/model/version")
      .then((r) => r.json())
      .then((v) => {
        setVersion(v.model_version);
        setBuiltAt(v.built_at || "");
      })
      .catch(() => {});
  }, []);

  return (
    <div className="min-h-screen flex flex-col">
      <TopBar route={route} setRoute={setRoute} />
      <main className="flex-1 min-h-0">
        {route.name === "showcase" && <ForecastShowcase setRoute={setRoute} />}
        {route.name === "meter" && (
          <MeterDetail msn={route.msn} back={() => setRoute({ name: "showcase" })} />
        )}
        {route.name === "network" && (
          <NetworkPlaceholder back={() => setRoute({ name: "showcase" })} />
        )}
        {route.name === "portfolio" && (
          <PortfolioView back={() => setRoute({ name: "showcase" })} />
        )}
      </main>
      <Footer modelVersion={version} builtAt={builtAt} />
    </div>
  );
}

function TopBar({ route, setRoute }: { route: Route; setRoute: (r: Route) => void }) {
  return (
    <header className="flex items-center justify-between px-6 py-3 border-b border-zinc-800 bg-zinc-950/80 backdrop-blur sticky top-0 z-20">
      <button
        onClick={() => setRoute({ name: "showcase" })}
        className="flex items-center gap-2 text-teal-500 hover:text-teal-600 transition-colors duration-fast font-semibold"
      >
        <Zap className="w-5 h-5" />
        EdgeGrid <span className="text-zinc-500 font-normal hidden sm:inline">Forecast Engine</span>
      </button>
      <nav className="flex items-center gap-1 text-sm">
        <NavBtn
          icon={<Gauge className="w-4 h-4" />}
          label="Accuracy"
          active={route.name === "showcase" || route.name === "meter"}
          onClick={() => setRoute({ name: "showcase" })}
        />
        <NavBtn
          icon={<NetworkIcon className="w-4 h-4" />}
          label="Network"
          active={route.name === "network"}
          onClick={() => setRoute({ name: "network" })}
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
        (props.active
          ? "bg-zinc-800 text-white"
          : "text-zinc-400 hover:text-white hover:bg-zinc-900")
      }
    >
      {props.icon}
      {props.label}
    </button>
  );
}

function Footer({ modelVersion, builtAt }: { modelVersion: string; builtAt: string }) {
  const trained = builtAt
    ? new Date(builtAt).toLocaleDateString(undefined, { month: "short", day: "numeric" })
    : "";
  return (
    <footer className="flex items-center justify-between px-6 py-2 text-xs text-zinc-500 border-t border-zinc-900">
      <span>v4 forecasting engine · 42 meters · APEPDCL Visakhapatnam</span>
      <span>
        {modelVersion ? (
          <>
            Model <code className="text-zinc-300">{modelVersion}</code>
            {trained ? ` · trained ${trained}` : ""}
          </>
        ) : (
          "loading model…"
        )}
      </span>
    </footer>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// SCREEN 1 — ForecastShowcase (landing)
// "See the v4 engine forecasting accurately, live on 42 real meters."
// ═══════════════════════════════════════════════════════════════════════════

function ForecastShowcase({ setRoute }: { setRoute: (r: Route) => void }) {
  const [summary, setSummary] = useState<FleetSummary | null>(null);
  const [meters, setMeters] = useState<MeterSummary[]>([]);
  const [replay, setReplay] = useState<FleetReplayResponse | null>(null);
  const [asOf, setAsOf] = useState<string>("");
  const [availableRange, setAvailableRange] =
    useState<{ min: string; max: string } | null>(null);
  const [tierFilter, setTierFilter] = useState<string | null>(null);
  const [loading, setLoading] = useState<"init" | "replay" | null>("init");
  const [error, setError] = useState<string | null>(null);
  // Monotonic generation counter — only the latest replay request is allowed
  // to set state. StrictMode double-invokes effects in dev, and the naive
  // `cancelled` boolean loses races between responses.
  const replayGenRef = useRef(0);

  useEffect(() => {
    let cancelled = false;
    (async () => {
      try {
        const [s, m, r] = await Promise.all([
          showcase.fleetSummary(),
          showcase.meters(),
          showcase.actualsRange(),
        ]);
        if (cancelled) return;
        setSummary(s);
        setMeters(m.meters);
        if (r.max) {
          const maxTs = new Date(r.max);
          const defaultAsOf = new Date(maxTs.getTime() - 48 * 3600 * 1000);
          setAsOf(defaultAsOf.toISOString());
          setAvailableRange({ min: r.min || "", max: r.max });
        }
      } catch (e: any) {
        setError(e.message || String(e));
        setLoading(null);
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  // Stable join key — keyed on MSN list rather than the meters-array identity,
  // so StrictMode re-mounts don't spuriously re-trigger the 12s replay.
  const msnsKey = useMemo(() => meters.map((m) => m.msn).join(","), [meters]);

  useEffect(() => {
    if (!asOf || !msnsKey) return;
    const myGen = ++replayGenRef.current;
    setLoading((cur) => (cur === "init" ? "init" : "replay"));
    showcase
      .fleetReplay(msnsKey.split(","), asOf, 48)
      .then((r) => {
        if (myGen !== replayGenRef.current) return;
        setReplay(r);
        setLoading(null);
      })
      .catch((e) => {
        if (myGen !== replayGenRef.current) return;
        setError(e.message || String(e));
        setLoading(null);
      });
  }, [asOf, msnsKey]);

  const tiles = useMemo(() => {
    if (meters.length === 0) return [];
    const byMsn: Record<string, MeterReplay> = replay
      ? Object.fromEntries(replay.meters.map((m) => [m.msn, m]))
      : {};
    return meters
      .map((m) => ({ meta: m, replay: byMsn[m.msn] }))
      .filter((t) => !tierFilter || t.meta.tier === tierFilter);
  }, [replay, meters, tierFilter]);

  if (error && !summary) {
    return <ErrorState msg={error} retry={() => window.location.reload()} />;
  }
  if (loading === "init" && !summary) {
    return <InitialLoading />;
  }

  return (
    <div className="px-4 md:px-6 py-5 max-w-[1600px] mx-auto">
      <ProofStrip
        summary={summary}
        replayMape={replay?.fleet_mape ?? null}
        loading={loading === "replay"}
      />

      <div className="card p-4 mb-5 flex flex-wrap items-center justify-between gap-4">
        <TimeScrubber
          asOf={asOf}
          setAsOf={setAsOf}
          available={availableRange}
          disabled={loading === "replay"}
        />
        <TierFilter
          active={tierFilter}
          setActive={setTierFilter}
          counts={summary?.tier_counts ?? {}}
        />
      </div>

      <section>
        <div className="flex items-baseline justify-between mb-3">
          <h2 className="text-sm muted uppercase tracking-wider">
            Live forecast vs actual · {tiles.length} meters
          </h2>
          <p className="text-xs muted">Click any tile for the full chart →</p>
        </div>

        {tiles.length === 0 ? (
          <EmptyTiles tierFilter={tierFilter} reset={() => setTierFilter(null)} />
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-3">
            {tiles.map((t) => (
              <MeterTile
                key={t.meta.msn}
                meta={t.meta}
                replay={t.replay}
                onClick={() => setRoute({ name: "meter", msn: t.meta.msn })}
              />
            ))}
          </div>
        )}
      </section>
    </div>
  );
}

function ProofStrip({
  summary,
  replayMape,
  loading,
}: {
  summary: FleetSummary | null;
  replayMape: number | null;
  loading: boolean;
}) {
  if (!summary) return null;
  const mainNumber = replayMape != null ? replayMape : summary.mean_mape_pct;
  const mainLabel = replayMape != null ? "this window" : "training holdout";
  return (
    <section className="mb-5">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-3 mb-3">
        <div>
          <h1 className="text-2xl md:text-3xl font-semibold leading-tight">
            Forecasting accuracy, live
          </h1>
          <p className="muted text-sm mt-1 max-w-2xl">
            The v4 engine predicts energy demand on {summary.n_meters} real APEPDCL meters
            at 30-minute granularity. Pick any date below to replay the forecast against
            what actually happened.
          </p>
        </div>
        <div className="flex items-center gap-2 text-xs muted self-start md:self-auto">
          <Layers className="w-3.5 h-3.5" />
          LightGBM · 36 features · per-tier regularization
        </div>
      </div>

      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        <BigStat
          value={`${mainNumber.toFixed(2)}%`}
          label={`Mean accuracy error · ${mainLabel}`}
          sub={loading ? "Recomputing…" : `${summary.n_meters} meters live`}
          highlight
        />
        <BigStat
          value={`${summary.median_mape_pct.toFixed(2)}%`}
          label="Median (half the fleet is better)"
          sub="Training holdout"
        />
        <BigStat
          value={
            summary.ht_peak_median_mape_pct != null
              ? `${summary.ht_peak_median_mape_pct.toFixed(2)}%`
              : "—"
          }
          label="HT peak-hour accuracy"
          sub="Evening 6–10 PM · commercial window"
        />
        <HealthBreakdown counts={summary.health_counts} />
      </div>
    </section>
  );
}

function BigStat({
  value,
  label,
  sub,
  highlight,
}: {
  value: string;
  label: string;
  sub?: string;
  highlight?: boolean;
}) {
  return (
    <div
      className={
        "card p-4 " +
        (highlight
          ? "bg-gradient-to-br from-teal-900/40 to-zinc-900 border-teal-800/60"
          : "")
      }
    >
      <div className="text-3xl md:text-4xl font-semibold tracking-tight">{value}</div>
      <div className="text-sm mt-1 font-medium">{label}</div>
      {sub && <div className="text-xs muted mt-0.5">{sub}</div>}
    </div>
  );
}

function HealthBreakdown({
  counts,
}: {
  counts: { green: number; amber: number; red: number };
}) {
  const total = counts.green + counts.amber + counts.red;
  const pct = (n: number) => (total > 0 ? (n / total) * 100 : 0);
  return (
    <div className="card p-4">
      <div className="flex items-baseline justify-between">
        <div className="text-3xl md:text-4xl font-semibold tracking-tight">
          {counts.green}
          <span className="text-lg muted font-normal">/{total}</span>
        </div>
        <div className="text-xs muted">under 8% error</div>
      </div>
      <div className="text-sm mt-1 font-medium">Fleet health</div>
      <div className="mt-2 flex h-1.5 rounded-full overflow-hidden bg-zinc-800">
        <div className="bg-emerald-500" style={{ width: `${pct(counts.green)}%` }} />
        <div className="bg-amber-500" style={{ width: `${pct(counts.amber)}%` }} />
        <div className="bg-rose-500" style={{ width: `${pct(counts.red)}%` }} />
      </div>
      <div className="flex gap-3 text-[10px] mt-1.5 muted">
        <span>
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-emerald-500 mr-1" />
          good
        </span>
        <span>
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-amber-500 mr-1" />
          watch {counts.amber}
        </span>
        <span>
          <span className="inline-block w-1.5 h-1.5 rounded-full bg-rose-500 mr-1" />
          retrain {counts.red}
        </span>
      </div>
    </div>
  );
}

function TimeScrubber({
  asOf,
  setAsOf,
  available,
  disabled,
}: {
  asOf: string;
  setAsOf: (s: string) => void;
  available: { min: string; max: string } | null;
  disabled: boolean;
}) {
  if (!available) return null;
  const currentDate = asOf ? new Date(asOf) : null;
  const maxDate = new Date(available.max);
  const minDate = new Date(available.min);
  const maxReplay = new Date(maxDate.getTime() - 48 * 3600 * 1000);

  const toInput = (d: Date) => {
    const tzOffset = d.getTimezoneOffset() * 60000;
    return new Date(d.getTime() - tzOffset).toISOString().slice(0, 16);
  };

  return (
    <div className="flex flex-col sm:flex-row sm:items-center gap-2">
      <div className="flex items-center gap-2 text-sm">
        <Calendar className="w-4 h-4 muted" />
        <span className="muted">Replay as of</span>
      </div>
      <input
        type="datetime-local"
        value={currentDate ? toInput(currentDate) : ""}
        min={toInput(minDate)}
        max={toInput(maxReplay)}
        disabled={disabled}
        onChange={(e) => {
          if (e.target.value) setAsOf(new Date(e.target.value).toISOString());
        }}
        className="bg-zinc-900 border border-zinc-700 rounded-md px-2 py-1.5 text-sm text-zinc-100 [color-scheme:dark] focus:border-teal-500 focus:outline-none"
      />
      <div className="flex gap-1">
        {[
          { label: "7 days ago", days: 7 },
          { label: "14 days ago", days: 14 },
          { label: "30 days ago", days: 30 },
        ].map((p) => (
          <button
            key={p.days}
            disabled={disabled}
            onClick={() => {
              const d = new Date(maxDate.getTime() - p.days * 86400000);
              setAsOf(d.toISOString());
            }}
            className="text-xs px-2 py-1 rounded border border-zinc-800 hover:border-teal-700 hover:text-teal-400 muted transition-colors duration-fast disabled:opacity-50"
          >
            {p.label}
          </button>
        ))}
      </div>
    </div>
  );
}

function TierFilter({
  active,
  setActive,
  counts,
}: {
  active: string | null;
  setActive: (t: string | null) => void;
  counts: Record<string, number>;
}) {
  const entries = Object.entries(counts);
  return (
    <div className="flex gap-1 flex-wrap">
      <Chip active={active === null} onClick={() => setActive(null)}>
        All {entries.reduce((s, [, n]) => s + n, 0)}
      </Chip>
      {entries.map(([tier, n]) => (
        <Chip key={tier} active={active === tier} onClick={() => setActive(tier)}>
          {tier.split(" ")[0]} {n}
        </Chip>
      ))}
    </div>
  );
}

function Chip({
  active,
  onClick,
  children,
}: {
  active: boolean;
  onClick: () => void;
  children: React.ReactNode;
}) {
  return (
    <button
      onClick={onClick}
      className={
        "text-xs px-2.5 py-1 rounded-full border transition-colors duration-fast " +
        (active
          ? "bg-teal-900/40 border-teal-700 text-teal-100"
          : "bg-zinc-900 border-zinc-800 muted hover:text-white hover:border-zinc-700")
      }
    >
      {children}
    </button>
  );
}

// ─── The meter tile — the visual heart of the product ──────────────────────

function MeterTile({
  meta,
  replay,
  onClick,
}: {
  meta: MeterSummary;
  replay: MeterReplay | undefined;
  onClick: () => void;
}) {
  const healthBorder = {
    green: "border-emerald-700/50 hover:border-emerald-500",
    amber: "border-amber-700/50 hover:border-amber-500",
    red: "border-rose-800/50 hover:border-rose-500",
  }[meta.health];
  const dot = {
    green: "bg-emerald-500",
    amber: "bg-amber-500",
    red: "bg-rose-500",
  }[meta.health];
  const tierShort = meta.tier.split(" ")[0];
  const liveMape = replay?.mape;

  return (
    <button
      onClick={onClick}
      className={
        "card text-left p-3 transition-all duration-fast hover:-translate-y-0.5 " +
        healthBorder
      }
    >
      <div className="flex items-center justify-between mb-1.5">
        <div className="flex items-center gap-1.5 min-w-0">
          <span className={"w-1.5 h-1.5 rounded-full flex-shrink-0 " + dot} />
          <span className="text-xs font-medium">{tierShort}</span>
          <span className="text-[10px] muted truncate">· {meta.phase}</span>
        </div>
        <span className="text-[10px] muted">#{meta.msn.slice(-4)}</span>
      </div>
      <TileSparkline replay={replay} health={meta.health} />
      <div className="flex items-baseline justify-between mt-2">
        <div>
          <div className="text-xl font-semibold leading-none">
            {liveMape != null
              ? `${liveMape.toFixed(1)}%`
              : replay?.status === "no_actuals"
                ? "—"
                : "…"}
          </div>
          <div className="text-[10px] muted mt-0.5">
            {liveMape != null
              ? "error this window"
              : replay?.status === "no_actuals"
                ? "no actuals yet"
                : "loading"}
          </div>
        </div>
        <ChevronRight className="w-4 h-4 muted" />
      </div>
    </button>
  );
}

function TileSparkline({
  replay,
  health,
}: {
  replay: MeterReplay | undefined;
  health: string;
}) {
  const data = useMemo(() => {
    if (!replay || !replay.series?.length) return [];
    return replay.series.map((p, i) => ({ i, f: p.forecast_wh, a: p.actual_wh }));
  }, [replay]);

  const stroke =
    { green: "#10B981", amber: "#F59E0B", red: "#EF4444" }[health] || "#0E7C7B";

  if (data.length === 0) {
    return <div className="h-14 w-full bg-zinc-950/50 rounded animate-pulse" />;
  }

  return (
    <div className="h-14 w-full">
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={data} margin={{ top: 2, right: 2, bottom: 2, left: 2 }}>
          <YAxis hide domain={["dataMin", "dataMax"]} />
          <Line
            dataKey="f"
            stroke={stroke}
            strokeWidth={1.5}
            dot={false}
            isAnimationActive={false}
          />
          <Line
            dataKey="a"
            stroke="#FAFAFA"
            strokeWidth={1.2}
            strokeDasharray="2 2"
            dot={false}
            isAnimationActive={false}
            connectNulls={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}

// ─── Loading / empty / error states (first-class per feedback) ─────────────

function InitialLoading() {
  return (
    <div className="px-6 py-16 max-w-4xl mx-auto">
      <div className="card p-8 flex items-center gap-4">
        <Gauge className="w-8 h-8 text-teal-500 animate-pulse" />
        <div>
          <h1 className="text-lg font-semibold">Loading forecast fleet</h1>
          <p className="muted text-sm">
            Warming up the v4 engine on 42 real meters · replaying against
            held-out actuals. Takes about 5 seconds on first load.
          </p>
        </div>
      </div>
    </div>
  );
}

function EmptyTiles({
  tierFilter,
  reset,
}: {
  tierFilter: string | null;
  reset: () => void;
}) {
  return (
    <div className="card p-8 text-center">
      <p className="muted text-sm mb-3">
        No meters matching {tierFilter ? `"${tierFilter}"` : "your filter"}.
      </p>
      <button className="btn-ghost" onClick={reset}>
        Show all meters
      </button>
    </div>
  );
}

function ErrorState({ msg, retry }: { msg: string; retry: () => void }) {
  return (
    <div className="px-6 py-16 max-w-2xl mx-auto">
      <div className="card p-6 border-rose-900/50 bg-rose-950/20">
        <h2 className="text-lg font-semibold mb-1">
          Couldn't load the forecast fleet.
        </h2>
        <p className="text-sm muted mb-1">
          The backend didn't respond. Most likely the API isn't running on
          port 8000, or the model bundles at <code>models/v4/</code> aren't on disk.
        </p>
        <code className="text-xs text-rose-300 block mt-2 mb-4">{msg}</code>
        <button onClick={retry} className="btn-primary">
          Try again
        </button>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// SCREEN 2 — MeterDetail
// ═══════════════════════════════════════════════════════════════════════════

function MeterDetail({ msn, back }: { msn: string; back: () => void }) {
  const [h, setH] = useState<MeterHistoryResponse | null>(null);
  const [err, setErr] = useState<string | null>(null);

  useEffect(() => {
    showcase.meterHistory(msn, undefined, 96).then(setH).catch((e) => setErr(e.message));
  }, [msn]);

  const chartData = useMemo(() => {
    if (!h) return [];
    return h.series.map((p) => ({
      ts: new Date(p.ts).toLocaleString(undefined, {
        month: "short",
        day: "numeric",
        hour: "2-digit",
        minute: "2-digit",
      }),
      forecast: p.forecast_wh,
      actual: p.actual_wh,
      low: p.confidence_low,
      high: p.confidence_high,
    }));
  }, [h]);

  const blockBars = useMemo(() => {
    if (!h) return [];
    return (["night", "morning", "solar", "peak"] as const).map((b) => ({
      block: b === "peak" ? "Peak 6-10 PM" : b.charAt(0).toUpperCase() + b.slice(1),
      trained: h.block_mape_trained[b],
      replay: h.block_mape[b],
    }));
  }, [h]);

  return (
    <div className="px-6 py-5 max-w-[1400px] mx-auto">
      <button
        onClick={back}
        className="flex items-center gap-1 muted hover:text-white text-sm mb-3"
      >
        <ArrowLeft className="w-4 h-4" /> Back to fleet
      </button>

      {err && (
        <div className="card p-4 border-rose-900/50 bg-rose-950/20 text-sm">
          <b>Couldn't load this meter's history.</b>{" "}
          <span className="text-rose-300">{err}</span>
        </div>
      )}

      {!h && !err && (
        <div className="card p-8 text-center muted text-sm">
          Loading forecast history…
        </div>
      )}

      {h && (
        <>
          <div className="flex flex-col md:flex-row md:items-end justify-between gap-3 mb-4">
            <div>
              <h1 className="text-2xl font-semibold">
                {h.tier.split(" ")[0]} meter · forecast vs actual
              </h1>
              <p className="muted text-sm mt-1">
                <code>MSN {h.msn}</code> · {h.tier} · {h.phase} · replay as of{" "}
                {new Date(h.as_of).toLocaleString()}
              </p>
            </div>
            <div className="flex gap-4">
              <div className="text-right">
                <div className="text-[10px] muted uppercase tracking-wider">
                  Training
                </div>
                <div className="text-2xl font-semibold">
                  {h.holdout_mape_trained.toFixed(2)}%
                </div>
              </div>
              <div className="text-right">
                <div className="text-[10px] muted uppercase tracking-wider">
                  This window
                </div>
                <div className="text-2xl font-semibold text-teal-400">
                  {h.replay_mape_actual != null
                    ? `${h.replay_mape_actual.toFixed(2)}%`
                    : "—"}
                </div>
              </div>
            </div>
          </div>

          <div className="card p-4 mb-5">
            <h2 className="text-sm muted uppercase tracking-wider mb-3">
              48-hour overlay · white dashed line is the truth
            </h2>
            <ResponsiveContainer width="100%" height={340}>
              <ComposedChart data={chartData}>
                <defs>
                  <linearGradient id="bandFill" x1="0" x2="0" y1="0" y2="1">
                    <stop offset="0%" stopColor="#0E7C7B" stopOpacity={0.35} />
                    <stop offset="100%" stopColor="#0E7C7B" stopOpacity={0.05} />
                  </linearGradient>
                </defs>
                <CartesianGrid stroke="#27272A" strokeDasharray="3 3" />
                <XAxis
                  dataKey="ts"
                  stroke="#71717A"
                  fontSize={10}
                  interval="preserveStartEnd"
                  minTickGap={60}
                />
                <YAxis
                  stroke="#71717A"
                  fontSize={10}
                  tickFormatter={(v) => `${Math.round(Number(v) / 1000)}k`}
                />
                <Tooltip
                  contentStyle={{ background: "#18181B", border: "1px solid #27272A" }}
                  formatter={(v: any, name: any) => [
                    typeof v === "number" ? `${v.toFixed(0)} Wh` : v,
                    name === "forecast"
                      ? "Forecast"
                      : name === "actual"
                        ? "Actual"
                        : name,
                  ]}
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Area
                  type="monotone"
                  dataKey="high"
                  stroke="none"
                  fill="url(#bandFill)"
                  name="Confidence band"
                />
                <Line
                  type="monotone"
                  dataKey="forecast"
                  stroke="#0E7C7B"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                  name="Forecast"
                />
                <Line
                  type="monotone"
                  dataKey="actual"
                  stroke="#FAFAFA"
                  strokeWidth={1.5}
                  strokeDasharray="3 3"
                  dot={false}
                  isAnimationActive={false}
                  connectNulls={false}
                  name="Actual"
                />
              </ComposedChart>
            </ResponsiveContainer>
          </div>

          <div className="card p-4">
            <h2 className="text-sm muted uppercase tracking-wider mb-3">
              Accuracy by time of day · training vs this window
            </h2>
            <ResponsiveContainer width="100%" height={220}>
              <BarChart data={blockBars}>
                <CartesianGrid stroke="#27272A" strokeDasharray="3 3" />
                <XAxis dataKey="block" stroke="#71717A" fontSize={11} />
                <YAxis stroke="#71717A" fontSize={11} unit="%" />
                <Tooltip
                  contentStyle={{ background: "#18181B", border: "1px solid #27272A" }}
                  formatter={(v: any) =>
                    typeof v === "number" ? `${v.toFixed(2)}%` : v
                  }
                />
                <Legend wrapperStyle={{ fontSize: 11 }} />
                <Bar dataKey="trained" fill="#27272A" name="Trained" />
                <Bar dataKey="replay" fill="#0E7C7B" name="This window" />
              </BarChart>
            </ResponsiveContainer>
            <p className="text-xs muted mt-3">
              Peak block (6–10 PM) is the commercial window — tight accuracy here
              backs firm-power contracts.
            </p>
          </div>
        </>
      )}
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// SCREEN 3 — Network (placeholder; demoted)
// ═══════════════════════════════════════════════════════════════════════════

function NetworkPlaceholder({ back }: { back: () => void }) {
  return (
    <div className="px-6 py-16 max-w-2xl mx-auto">
      <button
        onClick={back}
        className="flex items-center gap-1 muted hover:text-white text-sm mb-4"
      >
        <ArrowLeft className="w-4 h-4" /> Back
      </button>
      <div className="card p-8">
        <NetworkIcon className="w-8 h-8 text-teal-500 mb-3" />
        <h1 className="text-xl font-semibold mb-2">Network view</h1>
        <p className="muted text-sm mb-4">
          The substation-meter-BESS graph lives here — useful for dispatch
          simulation; less useful for the core accuracy proof, so it's moved
          out of the primary path.
        </p>
        <p className="text-xs muted">
          Coming in v3: embedded into the Accuracy view as an optional "see by
          substation" lens.
        </p>
      </div>
    </div>
  );
}

// ═══════════════════════════════════════════════════════════════════════════
// SCREEN 4 — Portfolio
// ═══════════════════════════════════════════════════════════════════════════

function PortfolioView({ back }: { back: () => void }) {
  const [summary, setSummary] = useState<FleetSummary | null>(null);
  const [meters, setMeters] = useState<MeterSummary[]>([]);

  useEffect(() => {
    Promise.all([showcase.fleetSummary(), showcase.meters()])
      .then(([s, m]) => {
        setSummary(s);
        setMeters(m.meters);
      })
      .catch(() => {});
  }, []);

  return (
    <div className="px-6 py-5 max-w-6xl mx-auto">
      <button
        onClick={back}
        className="flex items-center gap-1 muted hover:text-white text-sm mb-3"
      >
        <ArrowLeft className="w-4 h-4" /> Back
      </button>
      <h1 className="text-2xl font-semibold mb-5">Meter fleet</h1>
      {summary && (
        <div className="grid grid-cols-3 gap-3 mb-5">
          <BigStat value={String(summary.n_meters)} label="Meters" />
          <BigStat
            value={`${summary.mean_mape_pct.toFixed(2)}%`}
            label="Mean accuracy error"
          />
          <BigStat
            value={`${summary.health_counts.green}/${summary.n_meters}`}
            label="Under 8% error"
          />
        </div>
      )}
      <div className="card overflow-hidden">
        <table className="w-full text-sm">
          <thead className="bg-zinc-900/60 text-xs muted uppercase tracking-wider">
            <tr>
              <th className="px-4 py-2 text-left">Meter</th>
              <th className="px-4 py-2 text-left">Tier</th>
              <th className="px-4 py-2 text-right">Mean error</th>
              <th className="px-4 py-2 text-right">Peak error</th>
              <th className="px-4 py-2 text-left">Health</th>
            </tr>
          </thead>
          <tbody>
            {meters.map((m) => (
              <tr
                key={m.msn}
                className="border-t border-zinc-900 hover:bg-zinc-900/40 transition-colors duration-fast"
              >
                <td className="px-4 py-2">
                  <code className="text-zinc-300">{m.msn}</code>
                </td>
                <td className="px-4 py-2 muted">{m.tier}</td>
                <td className="px-4 py-2 text-right">{m.holdout_mape.toFixed(2)}%</td>
                <td className="px-4 py-2 text-right">{m.peak_mape.toFixed(2)}%</td>
                <td className="px-4 py-2">
                  <span
                    className={
                      "chip " +
                      (m.health === "green"
                        ? "bg-emerald-900/40 text-emerald-300"
                        : m.health === "amber"
                          ? "bg-amber-900/40 text-amber-300"
                          : "bg-rose-900/40 text-rose-300")
                    }
                  >
                    <span
                      className={
                        "w-1.5 h-1.5 rounded-full " +
                        (m.health === "green"
                          ? "bg-emerald-500"
                          : m.health === "amber"
                            ? "bg-amber-500"
                            : "bg-rose-500")
                      }
                    />
                    {m.health}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
