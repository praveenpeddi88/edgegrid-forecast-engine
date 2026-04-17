// Typed client additions for the Forecast Showcase endpoints.
// Kept separate from api.ts so the dispatch endpoints stay untouched.

const BASE = "/api";

async function get<T>(path: string): Promise<T> {
  const r = await fetch(`${BASE}${path}`);
  if (!r.ok) throw new Error(`${path} → ${r.status} ${r.statusText}`);
  return (await r.json()) as T;
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const r = await fetch(`${BASE}${path}`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${path} → ${r.status}`);
  return (await r.json()) as T;
}

export interface FleetSummary {
  n_meters: number;
  mean_mape_pct: number;
  median_mape_pct: number;
  peak_block_mean_mape_pct: number | null;
  peak_block_median_mape_pct: number | null;
  ht_peak_median_mape_pct: number | null;
  model_version: string;
  built_at: string | null;
  health_counts: { green: number; amber: number; red: number };
  tier_counts: Record<string, number>;
}

export interface MeterSummary {
  msn: string;
  tier: string;
  phase: string;
  holdout_mape: number;
  night_mape: number;
  morning_mape: number;
  solar_mape: number;
  peak_mape: number;
  bias_gate: string;
  health: "green" | "amber" | "red";
}

export interface ReplayPoint {
  ts: string;
  forecast_wh: number;
  confidence_low: number;
  confidence_high: number;
  actual_wh: number | null;
}

export interface MeterReplay {
  msn: string;
  mape: number | null;
  status: "ok" | "no_forecast" | "no_actuals";
  n_points: number;
  forecast_mean_wh: number;
  actual_mean_wh: number | null;
  tier: string;
  peak_mape: number;
  series: ReplayPoint[];
}

export interface FleetReplayResponse {
  as_of: string;
  horizon: number;
  fleet_mape: number | null;
  fleet_mean_wh: number;
  meters: MeterReplay[];
}

export interface ActualsRange {
  min: string | null;
  max: string | null;
  n_meters: number;
}

export interface MeterHistoryResponse {
  msn: string;
  tier: string;
  phase: string;
  as_of: string;
  horizon: number;
  holdout_mape_trained: number;
  replay_mape_actual: number | null;
  block_mape: Record<string, number | null>;
  block_mape_trained: Record<string, number>;
  series: Array<ReplayPoint & { block_label: string }>;
}

export const showcase = {
  fleetSummary: () => get<FleetSummary>("/fleet/summary"),
  meters: () => get<{ n: number; meters: MeterSummary[] }>("/meters"),
  actualsRange: () => get<ActualsRange>("/fleet/actuals-range"),
  fleetReplay: (msns: string[], asOf?: string, horizon = 48) =>
    post<FleetReplayResponse>("/fleet/replay", { msns, as_of: asOf, horizon, include_actuals: true }),
  meterHistory: (msn: string, asOf?: string, horizon = 96) => {
    const qs = new URLSearchParams();
    if (asOf) qs.set("as_of", asOf);
    qs.set("horizon", String(horizon));
    return get<MeterHistoryResponse>(`/meter/${msn}/history?${qs}`);
  },
};
