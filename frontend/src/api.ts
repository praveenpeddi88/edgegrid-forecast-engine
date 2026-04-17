// Typed client for the EdgeGrid backend. Vite proxies /api → localhost:8000.

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

// ─── types (loose — backend is source of truth) ─────────────────────────────

export type NodeKind =
  | "meter" | "feeder" | "substation" | "bess" | "solar"
  | "ci_load" | "weather" | "iex";

export interface GraphNode {
  id: string;
  label: string;
  kind: NodeKind;
  [k: string]: any;
}

export interface GraphEdge {
  source: string;
  target: string;
  edge_type: string;
  attrs?: Record<string, any>;
}

export interface NetworkGraph {
  nodes: GraphNode[];
  edges: GraphEdge[];
  meta: { n_nodes: number; n_edges: number; kinds: string[] };
}

export interface DispatchRow {
  timestamp: string;
  demand_kwh: number;
  iex_price_inr: number;
  landed_cost_inr: number;
  bess_charge_kwh: number;
  bess_discharge_kwh: number;
  grid_import_kwh: number;
  solar_curtail_kwh: number;
  soc_kwh: number;
  soc_pct: number;
  block_label: string;
  historical_block_mape: number;
  confidence_weight: number;
  action: string;
  audit_string: string;
}

export interface DispatchResponse {
  substation_id: string;
  bess_id: string;
  as_of: string;
  solver_status: string;
  totals: {
    revenue_inr: number;
    cost_inr: number;
    net_benefit_inr: number;
    peak_kva: number;
    cycles: number;
  };
  meta: Record<string, any>;
  schedule: DispatchRow[];
}

export interface CommercialResponse {
  substation_id: string;
  daily_net_benefit_inr: number;
  annual_net_benefit_inr: number;
  reference: {
    capacity_mwh: number;
    duration_h: number;
    capex_inr: number;
    irr_pct: number;
    payback_years: number;
    npv_inr: number;
  };
  heatmap: {
    capacities_mwh: number[];
    durations_h: number[];
    irr_pct: number[][];
    payback_years: number[][];
  };
  sensitivity: Record<string, { delta_pct: number; irr_pct: number; payback_years: number }[]>;
}

export interface FLSQuote {
  contract_id: string;
  buyer_kw: number;
  window_start: string;
  window_end: string;
  weekdays_only: boolean;
  landed_cost_inr_per_kwh: number;
  offered_price_inr_per_kwh: number;
  firmness_pct: number;
  underlying_mape_pct: number;
  tenor_months: number;
  rationale: string;
}

export interface Portfolio {
  n_substations: number;
  n_meters: number;
  n_bess: number;
  meter_health: { green: number; amber: number; red: number };
  substations: Array<{
    id: string;
    label: string;
    n_meters: number;
    n_bess: number;
    landed_cost_inr_per_kwh: number;
  }>;
  canonical_substation_id: string;
  model_version: string;
}

export interface MeterForecastResponse {
  msn: string;
  as_of: string;
  horizon: number;
  model_version: string;
  rows: Array<{
    timestamp?: string;
    forecast_wh: number;
    confidence_low: number;
    confidence_high: number;
    block_label: string;
    historical_block_mape: number;
  }>;
}

// ─── endpoints ──────────────────────────────────────────────────────────────

export const api = {
  modelVersion: () => get<{ model_version: string; built_at: string | null; n_models: number }>("/model/version"),
  network: () => get<NetworkGraph>("/network"),
  portfolio: () => get<Portfolio>("/portfolio"),
  meterForecast: (msn: string, horizon = 48) => get<MeterForecastResponse>(`/meter/${msn}/forecast?horizon=${horizon}`),
  substationDispatch: (id: string) => get<DispatchResponse>(`/substation/${id}/dispatch`),
  substationCommercial: (id: string, capMwh = 1.0, durH = 4) =>
    get<CommercialResponse>(`/substation/${id}/commercial?capacity_mwh=${capMwh}&duration_h=${durH}`),
  substationFlsQuote: (id: string, body: {
    contract_id: string; buyer_kw: number; window_start: string;
    window_end: string; weekdays_only?: boolean; tenor_months?: number;
  }) => post<FLSQuote>(`/substation/${id}/fls-quote`, body),
  substationBriefUrl: (id: string) => `${BASE}/substation/${id}/brief.html`,
};
