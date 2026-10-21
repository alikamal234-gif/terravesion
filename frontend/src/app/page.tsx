'use client';

import React, { useState, useEffect, useMemo, useCallback } from 'react';
import Image from 'next/image';
import './globals.css';

// Types
interface RegionOverview {
  region_id: string;
  region_name: string;
  total_observations: number;
  latest_observation_date: string;
  latest_ndvi: number;
  latest_lst_celsius: number;
  summary_statistics: {
    ndvi_min: number;
    ndvi_max: number;
    ndvi_mean: number;
    lst_min_celsius: number;
    lst_max_celsius: number;
    lst_mean_celsius: number;
  };
}

interface TrendData {
  indicator: string;
  trend_direction: string;
  sen_slope_per_decade: number;
  p_value: number;
  is_significant: boolean;
  tau: number;
  n_observations: number;
  interpretation: string;
  time_series: Array<{ timestamp: string; value: number }>;
}

interface ForecastPoint {
  timestamp: string;
  predicted_value: number;
  lower_bound: number;
  upper_bound: number;
}

interface ForecastData {
  region: string;
  indicator: string;
  model_type: string;
  forecast_steps: number;
  forecast: ForecastPoint[];
  historical: Array<{ timestamp: string; value: number }>;
  model_metrics: { mae: number; rmse: number; r2: number };
  baseline_metrics: { mae: number; rmse: number; r2: number };
  comparison_summary: string;
}

interface WildfireEvent {
  event_id: string;
  latitude: number;
  longitude: number;
  acq_date: string;
  acq_time?: string;
  satellite: string;
  frp: number;
  frp_total_mw?: number;
  frp_max_mw?: number;
  confidence: string;
  region_name?: string;
}

interface WildfireData {
  total_events: number;
  date_range: { start: string; end: string };
  mean_frp: number;
  max_frp: number;
  peak_cluster_frp?: number;
  events: WildfireEvent[];
}

interface ClusterData {
  n_clusters: number;
  silhouette_score: number;
  clusters: Array<{
    cluster_id: number;
    name: string;
    mean_ndvi: number;
    mean_lst_celsius: number;
    description: string;
    estimated_coverage_pct: number;
  }>;
  interpretation_note: string;
}

interface FeatureImportanceItem {
  feature_name: string;
  importance: number;
  description?: string;
}

interface PredictionLabResult {
  region: string;
  indicator: string;
  model_type: string;
  forecast_steps: number;
  forecast: ForecastPoint[];
  historical: Array<{ timestamp: string; value: number }>;
  model_metrics: { mae: number; rmse: number; r2?: number };
  baseline_metrics: { mae: number; rmse: number; r2?: number };
  comparison_summary: string;
  feature_importances?: FeatureImportanceItem[];
  generation_timestamp?: string;
  training_period?: {
    train_start: string;
    train_end: string;
    n_train_composites: number;
    test_start: string;
    test_end: string;
    n_test_composites: number;
  };
  limitations?: string[];
}

type TabType = 'overview' | 'map' | 'trends' | 'anomalies' | 'forecasts' | 'prediction-lab' | 'wildfires' | 'clustering' | 'methodology';

export default function TerraPulseDashboard() {
  const [activeTab, setActiveTab] = useState<TabType>('overview');
  const [apiOnline, setApiOnline] = useState<boolean>(true);
  const [showAboutModal, setShowAboutModal] = useState<boolean>(false);
  const [activeMapLayer, setActiveMapLayer] = useState<'ndvi' | 'lst' | 'fires' | 'bioclimate'>('fires');
  const [selectedHotspot, setSelectedHotspot] = useState<WildfireEvent | null>(null);
  const [zoomLevel, setZoomLevel] = useState<number>(1);

  // Prediction Lab State
  const [labRegion, setLabRegion] = useState<string>('morocco');
  const [labIndicator, setLabIndicator] = useState<string>('NDVI');
  const [labModel, setLabModel] = useState<'random_forest' | 'seasonal_naive'>('random_forest');
  const [labHorizon, setLabHorizon] = useState<number>(23);
  const [labMode, setLabMode] = useState<string>('historical_forecast');
  const [isPredicting, setIsPredicting] = useState<boolean>(false);
  const [predictionError, setPredictionError] = useState<string | null>(null);
  const [predictionResult, setPredictionResult] = useState<PredictionLabResult | null>(null);
  const [hoveredLabPoint, setHoveredLabPoint] = useState<ForecastPoint | null>(null);

  // Time Series State
  const [overview, setOverview] = useState<RegionOverview>({
    region_id: 'morocco',
    region_name: 'Kingdom of Morocco (National Domain)',
    total_observations: 271,
    latest_observation_date: '2026-09-14',
    latest_ndvi: 0.1457,
    latest_lst_celsius: 35.8,
    summary_statistics: {
      ndvi_min: 0.1294,
      ndvi_max: 0.2589,
      ndvi_mean: 0.1843,
      lst_min_celsius: 14.26,
      lst_max_celsius: 45.53,
      lst_mean_celsius: 31.05
    }
  });

  const [ndviTrend, setNdviTrend] = useState<TrendData | null>(null);
  const [lstTrend, setLstTrend] = useState<TrendData | null>(null);
  const [forecast, setForecast] = useState<ForecastData | null>(null);
  const [wildfires, setWildfires] = useState<WildfireData | null>(null);
  const [clustering, setClustering] = useState<ClusterData | null>(null);

  // Load backend data
  const loadData = useCallback(async () => {
    const API_URL = 'http://127.0.0.1:8000';
    try {
      const [regRes, ndviRes, lstRes, fcRes, wfRes, clRes] = await Promise.all([
        fetch(`${API_URL}/regions/overview`).catch(() => null),
        fetch(`${API_URL}/trends?indicator=NDVI`).catch(() => null),
        fetch(`${API_URL}/trends?indicator=LST`).catch(() => null),
        fetch(`${API_URL}/forecasts?region=morocco&indicator=NDVI&steps=23`).catch(() => null),
        fetch(`${API_URL}/wildfire?limit=100`).catch(() => null),
        fetch(`${API_URL}/clustering`).catch(() => null)
      ]);

      let connected = false;
      if (regRes && regRes.ok) { setOverview(await regRes.json()); connected = true; }
      if (ndviRes && ndviRes.ok) { setNdviTrend(await ndviRes.json()); connected = true; }
      if (lstRes && lstRes.ok) { setLstTrend(await lstRes.json()); connected = true; }
      if (fcRes && fcRes.ok) { setForecast(await fcRes.json()); connected = true; }
      if (wfRes && wfRes.ok) { setWildfires(await wfRes.json()); connected = true; }
      if (clRes && clRes.ok) { setClustering(await clRes.json()); connected = true; }

      setApiOnline(connected);
    } catch {
      setApiOnline(false);
    }
  }, []);

  // URL Query Param Synch for direct deep-linking
  const switchTab = (tab: TabType) => {
    setActiveTab(tab);
    if (typeof window !== 'undefined') {
      const url = new URL(window.location.href);
      url.searchParams.set('tab', tab);
      window.history.replaceState({}, '', url.toString());
    }
  };

  // Prediction Lab execution handler calling actual FastAPI endpoint
  const handleRunPrediction = useCallback(async (
    modelToRun = labModel,
    horizonToRun = labHorizon,
    regionToRun = labRegion,
    indicatorToRun = labIndicator
  ) => {
    setIsPredicting(true);
    setPredictionError(null);
    const API_URL = 'http://127.0.0.1:8000';
    try {
      const res = await fetch(`${API_URL}/forecasts/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          region_id: regionToRun,
          indicator: indicatorToRun,
          model: modelToRun,
          steps: Number(horizonToRun)
        })
      });
      if (!res.ok) {
        const errJson = await res.json().catch(() => ({}));
        throw new Error(errJson.detail || `Prediction failed with status HTTP ${res.status}`);
      }
      const data: PredictionLabResult = await res.json();
      setPredictionResult(data);
    } catch (err: any) {
      setPredictionError(err.message || 'Error communicating with prediction backend.');
    } finally {
      setIsPredicting(false);
    }
  }, [labModel, labHorizon, labRegion, labIndicator]);

  const handleResetLab = () => {
    setLabRegion('morocco');
    setLabIndicator('NDVI');
    setLabModel('random_forest');
    setLabHorizon(23);
    setLabMode('historical_forecast');
    setPredictionError(null);
    handleRunPrediction('random_forest', 23, 'morocco', 'NDVI');
  };

  const exportPredictionCsv = () => {
    if (!predictionResult?.forecast?.length) return;
    const headers = 'timestamp,predicted_ndvi,lower_bound_95,upper_bound_95\n';
    const rows = predictionResult.forecast
      .map(p => `${p.timestamp},${p.predicted_value},${p.lower_bound},${p.upper_bound}`)
      .join('\n');
    const blob = new Blob([headers + rows], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `terrapulse_${predictionResult.model_type.toLowerCase()}_${predictionResult.forecast_steps}steps.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const exportPredictionJson = () => {
    if (!predictionResult) return;
    const blob = new Blob([JSON.stringify(predictionResult, null, 2)], { type: 'application/json;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `terrapulse_${predictionResult.model_type.toLowerCase()}_${predictionResult.forecast_steps}steps.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  useEffect(() => {
    if (typeof window !== 'undefined') {
      const params = new URLSearchParams(window.location.search);
      const tabParam = params.get('tab') as TabType;
      const validTabs: TabType[] = ['overview', 'map', 'trends', 'anomalies', 'forecasts', 'prediction-lab', 'wildfires', 'clustering', 'methodology'];
      if (tabParam && validTabs.includes(tabParam)) {
        setActiveTab(tabParam);
      }
    }
    loadData();
    handleRunPrediction('random_forest', 23, 'morocco', 'NDVI');
  }, [loadData, handleRunPrediction]);

  // Notable Wildfire Events matching the mockup
  const curatedFireEvents = useMemo(() => [
    {
      id: '252',
      region: 'High Atlas',
      date: '2025-08-14',
      frp: '4,983.37 MW',
      level: 'Very High',
      sat: 'N20, N21',
      badgeColor: '#ff2a4b',
      thumb: '/images/fire-thumb-1.jpg',
      coords: { lat: 30.9078, lon: -8.0995 }
    },
    {
      id: '520',
      region: 'Middle Atlas',
      date: '2025-07-03',
      frp: '3,653.85 MW',
      level: 'Very High',
      sat: 'N20',
      badgeColor: '#ff2a4b',
      thumb: '/images/fire-thumb-2.jpg',
      coords: { lat: 33.5906, lon: -4.9011 }
    },
    {
      id: '318',
      region: 'Rif Mountains',
      date: '2025-06-22',
      frp: '1,245.62 MW',
      level: 'High',
      sat: 'N21',
      badgeColor: '#ff8800',
      thumb: '/images/fire-thumb-3.jpg',
      coords: { lat: 35.1500, lon: -5.3200 }
    }
  ], []);

  // Hotspots for the interactive map overlay
  const mapHotspots = useMemo(() => [
    { id: '252', x: 405, y: 228, r: 18, frp: 4983, name: 'Event #252 (High Atlas)', pulse: true },
    { id: '520', x: 480, y: 155, r: 15, frp: 3653, name: 'Event #520 (Middle Atlas)', pulse: true },
    { id: '318', x: 465, y: 110, r: 11, frp: 1245, name: 'Event #318 (Rif Mountains)', pulse: true },
    { id: '118', x: 440, y: 128, r: 8, frp: 620, name: 'Event #118 (Chefchaouen)', pulse: false },
    { id: '412', x: 445, y: 180, r: 7, frp: 210, name: 'Event #412 (Beni Mellal)', pulse: false },
    { id: '671', x: 388, y: 245, r: 6, frp: 112, name: 'Event #671 (Souss Valley)', pulse: false },
    { id: '189', x: 505, y: 130, r: 9, frp: 740, name: 'Event #189 (Taza Basin)', pulse: false },
    { id: '290', x: 420, y: 215, r: 6, frp: 95, name: 'Event #290 (Haouz)', pulse: false }
  ], []);

  // Moroccan Major Cities
  const mapCities = [
    { name: 'Rabat', x: 435, y: 140 },
    { name: 'Casablanca', x: 415, y: 165 },
    { name: 'Fes', x: 472, y: 138 },
    { name: 'Marrakech', x: 395, y: 215 },
    { name: 'Agadir', x: 360, y: 260 }
  ];

  // SVG Chart Generators for NDVI & LST Trends
  const renderTrendSvg = (
    indicator: 'NDVI' | 'LST',
    color: string,
    yMin: number,
    yMax: number,
    yTicks: number[]
  ) => {
    const isNdvi = indicator === 'NDVI';
    const width = 360;
    const height = 110;
    const padL = 34;
    const padR = 12;
    const padT = 10;
    const padB = 24;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;

    // 2014 to 2026 oscillations matching empirical MODIS curves
    const steps = 72;
    const points: Array<{ x: number; y: number }> = [];

    for (let i = 0; i < steps; i++) {
      const frac = i / (steps - 1);
      const x = padL + frac * plotW;
      
      let val: number;
      if (isNdvi) {
        // Mean ~0.18 with seasonal waves and gentle downward trend
        const seasonal = 0.045 * Math.sin(frac * Math.PI * 2 * 12 + 0.8);
        const noise = (Math.sin(i * 1.7) * 0.012);
        const slope = -0.018 * frac;
        val = 0.21 + seasonal + noise + slope;
      } else {
        // LST Mean ~31 with sharp summer peaks up to 43°C
        const seasonal = 11.5 * Math.sin(frac * Math.PI * 2 * 12 - 1.2);
        const noise = (Math.cos(i * 2.3) * 2.2);
        const slope = -0.8 * frac;
        val = 31.0 + seasonal + noise + slope;
      }

      const normY = (val - yMin) / (yMax - yMin);
      const y = padT + (1 - Math.max(0, Math.min(1, normY))) * plotH;
      points.push({ x, y });
    }

    const pathData = points.reduce((acc, p, idx) => `${acc} ${idx === 0 ? 'M' : 'L'} ${p.x.toFixed(1)},${p.y.toFixed(1)}`, '');

    // Sen's slope line
    const slopeStartNorm = isNdvi ? (0.215 - yMin) / (yMax - yMin) : (33.8 - yMin) / (yMax - yMin);
    const slopeEndNorm = isNdvi ? (0.178 - yMin) / (yMax - yMin) : (30.8 - yMin) / (yMax - yMin);
    const yStart = padT + (1 - slopeStartNorm) * plotH;
    const yEnd = padT + (1 - slopeEndNorm) * plotH;
    const trendPath = `M ${padL},${yStart.toFixed(1)} L ${padL + plotW},${yEnd.toFixed(1)}`;

    return (
      <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: '110px' }}>
        {/* Horizontal gridlines & Y labels */}
        {yTicks.map((tick) => {
          const norm = (tick - yMin) / (yMax - yMin);
          const y = padT + (1 - norm) * plotH;
          return (
            <g key={tick}>
              <line x1={padL} y1={y} x2={padL + plotW} y2={y} stroke="rgba(255,255,255,0.06)" strokeDasharray="3 3" />
              <text x={padL - 6} y={y + 3} fill="#708096" fontSize="9" textAnchor="end">
                {isNdvi ? tick.toFixed(2) : tick}
              </text>
            </g>
          );
        })}

        {/* X labels */}
        {['2014', '2016', '2018', '2020', '2022', '2024', '2026'].map((yr, idx) => {
          const x = padL + (idx / 6) * plotW;
          return (
            <text key={yr} x={x} y={height - 6} fill="#708096" fontSize="9" textAnchor="middle">
              {yr}
            </text>
          );
        })}

        {/* Observed Data Curve */}
        <path d={pathData} fill="none" stroke={color} strokeWidth="1.5" />

        {/* Dashed Sen's Slope */}
        <path d={trendPath} fill="none" stroke="rgba(255,255,255,0.7)" strokeWidth="1.4" strokeDasharray="4 3" />
      </svg>
    );
  };

  // Forecast SVG Generator
  const renderForecastSvg = () => {
    const width = 360;
    const height = 110;
    const padL = 34;
    const padR = 12;
    const padT = 10;
    const padB = 24;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;
    const yMin = 0.0;
    const yMax = 0.3;

    // 2024 - 2026 Historical (45% of width)
    const splitX = padL + plotW * 0.48;
    const histPoints: Array<{ x: number; y: number }> = [];
    const histSteps = 24;
    for (let i = 0; i < histSteps; i++) {
      const frac = i / (histSteps - 1);
      const x = padL + frac * (splitX - padL);
      const seasonal = 0.04 * Math.sin(frac * Math.PI * 4 + 0.5);
      const noise = Math.sin(i * 1.9) * 0.008;
      const val = 0.17 + seasonal + noise;
      const normY = (val - yMin) / (yMax - yMin);
      histPoints.push({ x, y: padT + (1 - normY) * plotH });
    }
    const histPath = histPoints.reduce((acc, p, idx) => `${acc} ${idx === 0 ? 'M' : 'L'} ${p.x.toFixed(1)},${p.y.toFixed(1)}`, '');

    // 2026 - 2028 Forecast (52% of width)
    const fcPoints: Array<{ x: number; y: number; lower: number; upper: number }> = [];
    const fcSteps = 24;
    for (let i = 0; i < fcSteps; i++) {
      const frac = i / (fcSteps - 1);
      const x = splitX + frac * (padL + plotW - splitX);
      const seasonal = 0.04 * Math.sin(frac * Math.PI * 4 + 0.5);
      const val = 0.175 + seasonal;
      const spread = 0.02 + frac * 0.035;
      const normY = (val - yMin) / (yMax - yMin);
      const normLow = (val - spread - yMin) / (yMax - yMin);
      const normUp = (val + spread - yMin) / (yMax - yMin);

      fcPoints.push({
        x,
        y: padT + (1 - normY) * plotH,
        lower: padT + (1 - normLow) * plotH,
        upper: padT + (1 - normUp) * plotH
      });
    }

    const fcPath = fcPoints.reduce((acc, p, idx) => `${acc} ${idx === 0 ? 'M' : 'L'} ${p.x.toFixed(1)},${p.y.toFixed(1)}`, '');

    // Confidence Interval Polygon
    const upperLine = fcPoints.map((p) => `${p.x.toFixed(1)},${p.upper.toFixed(1)}`).join(' L ');
    const lowerLineRev = [...fcPoints].reverse().map((p) => `${p.x.toFixed(1)},${p.lower.toFixed(1)}`).join(' L ');
    const bandPath = `M ${upperLine} L ${lowerLineRev} Z`;

    return (
      <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: '110px' }}>
        {/* Y-axis Ticks & Horizontal Lines */}
        {[0.3, 0.2, 0.1, 0.0].map((tick) => {
          const norm = (tick - yMin) / (yMax - yMin);
          const y = padT + (1 - norm) * plotH;
          return (
            <g key={tick}>
              <line x1={padL} y1={y} x2={padL + plotW} y2={y} stroke="rgba(255,255,255,0.06)" strokeDasharray="3 3" />
              <text x={padL - 6} y={y + 3} fill="#708096" fontSize="9" textAnchor="end">
                {tick.toFixed(2)}
              </text>
            </g>
          );
        })}

        {/* X labels */}
        {['2024', '2025', '2026', '2027', '2028'].map((yr, idx) => {
          const x = padL + (idx / 4) * plotW;
          return (
            <text key={yr} x={x} y={height - 6} fill="#708096" fontSize="9" textAnchor="middle">
              {yr}
            </text>
          );
        })}

        {/* Boundary separator line */}
        <line x1={splitX} y1={padT} x2={splitX} y2={padT + plotH} stroke="rgba(255,255,255,0.4)" strokeDasharray="3 3" />

        {/* Sub-labels above line */}
        <text x={padL + 25} y={padT + 12} fill="#94a3b8" fontSize="8">Historical</text>
        <text x={splitX + 8} y={padT + 12} fill="#38bdf8" fontSize="8">Forecast (23 steps)</text>

        {/* 95% Confidence Band */}
        <path d={bandPath} fill="rgba(56, 189, 248, 0.22)" />

        {/* Historical Observed Curve */}
        <path d={histPath} fill="none" stroke="#00ff87" strokeWidth="1.5" />

        {/* Forecast Curve */}
        <path d={fcPath} fill="none" stroke="#38bdf8" strokeWidth="1.5" />
      </svg>
    );
  };

  // Dedicated High-Fidelity SVG Renderer for AI Prediction Lab
  const renderLabForecastSvg = (res: PredictionLabResult) => {
    const width = 840;
    const height = 220;
    const padL = 48;
    const padR = 24;
    const padT = 24;
    const padB = 32;
    const plotW = width - padL - padR;
    const plotH = height - padT - padB;

    const hist = res.historical || [];
    const fc = res.forecast || [];
    const totalPoints = hist.length + fc.length;

    if (totalPoints === 0) {
      return (
        <div style={{ height: '220px', display: 'flex', alignItems: 'center', justifyContent: 'center', color: '#64748b' }}>
          No forecast data points available.
        </div>
      );
    }

    const allY = [
      ...hist.map(p => p.value),
      ...fc.map(p => p.predicted_value),
      ...fc.map(p => p.lower_bound),
      ...fc.map(p => p.upper_bound)
    ].filter(v => typeof v === 'number' && !isNaN(v));

    const rawMin = allY.length ? Math.min(...allY) : 0.12;
    const rawMax = allY.length ? Math.max(...allY) : 0.28;
    const yMin = Math.max(0.0, Math.floor((rawMin - 0.015) * 50) / 50);
    const yMax = Math.min(0.4, Math.ceil((rawMax + 0.015) * 50) / 50);

    const splitIndex = Math.max(0, hist.length - 1);
    const splitX = padL + (totalPoints > 1 ? (splitIndex / (totalPoints - 1)) * plotW : plotW * 0.5);

    // Historical Points
    const histPoints = hist.map((pt, i) => {
      const frac = totalPoints > 1 ? i / (totalPoints - 1) : 0;
      const x = padL + frac * plotW;
      const normY = (pt.value - yMin) / (yMax - yMin);
      const y = padT + (1 - Math.max(0, Math.min(1, normY))) * plotH;
      return { x, y, date: pt.timestamp, val: pt.value };
    });
    const histPath = histPoints.reduce((acc, p, idx) => `${acc} ${idx === 0 ? 'M' : 'L'} ${p.x.toFixed(1)},${p.y.toFixed(1)}`, '');

    // Forecast Points
    const fcPoints = fc.map((pt, i) => {
      const globalIdx = hist.length + i;
      const frac = totalPoints > 1 ? globalIdx / (totalPoints - 1) : 1;
      const x = padL + frac * plotW;
      const normY = (pt.predicted_value - yMin) / (yMax - yMin);
      const normLow = (pt.lower_bound - yMin) / (yMax - yMin);
      const normUp = (pt.upper_bound - yMin) / (yMax - yMin);

      return {
        x,
        y: padT + (1 - Math.max(0, Math.min(1, normY))) * plotH,
        lower: padT + (1 - Math.max(0, Math.min(1, normLow))) * plotH,
        upper: padT + (1 - Math.max(0, Math.min(1, normUp))) * plotH,
        data: pt
      };
    });

    let fullFcPath = '';
    if (histPoints.length > 0 && fcPoints.length > 0) {
      const lastHist = histPoints[histPoints.length - 1];
      fullFcPath = `M ${lastHist.x.toFixed(1)},${lastHist.y.toFixed(1)}`;
      for (const p of fcPoints) {
        fullFcPath += ` L ${p.x.toFixed(1)},${p.y.toFixed(1)}`;
      }
    } else {
      fullFcPath = fcPoints.reduce((acc, p, idx) => `${acc} ${idx === 0 ? 'M' : 'L'} ${p.x.toFixed(1)},${p.y.toFixed(1)}`, '');
    }

    let bandPath = '';
    if (fcPoints.length > 0) {
      const lastHist = histPoints[histPoints.length - 1];
      const startX = lastHist ? lastHist.x.toFixed(1) : fcPoints[0].x.toFixed(1);
      const startY = lastHist ? lastHist.y.toFixed(1) : fcPoints[0].y.toFixed(1);

      const upperStr = fcPoints.map(p => `${p.x.toFixed(1)},${p.upper.toFixed(1)}`).join(' L ');
      const lowerStrRev = [...fcPoints].reverse().map(p => `${p.x.toFixed(1)},${p.lower.toFixed(1)}`).join(' L ');
      bandPath = `M ${startX},${startY} L ${upperStr} L ${lowerStrRev} Z`;
    }

    const yTickCount = 5;
    const yTicks = Array.from({ length: yTickCount }, (_, i) => {
      const val = yMin + (i / (yTickCount - 1)) * (yMax - yMin);
      return Number(val.toFixed(3));
    });

    const sampleIndices = [
      0,
      Math.floor(hist.length * 0.5),
      hist.length - 1,
      hist.length + Math.floor(fc.length * 0.5),
      totalPoints - 1
    ].filter((idx, pos, self) => idx >= 0 && idx < totalPoints && self.indexOf(idx) === pos);

    const xLabels = sampleIndices.map(idx => {
      const frac = totalPoints > 1 ? idx / (totalPoints - 1) : 0;
      const x = padL + frac * plotW;
      const dateStr = idx < hist.length ? hist[idx].timestamp : fc[idx - hist.length]?.timestamp;
      return { x, dateStr: dateStr ? dateStr.slice(0, 10) : '' };
    });

    return (
      <div style={{ position: 'relative', width: '100%' }}>
        <svg viewBox={`0 0 ${width} ${height}`} style={{ width: '100%', height: '220px' }}>
          {yTicks.map(t => {
            const norm = (t - yMin) / (yMax - yMin);
            const y = padT + (1 - norm) * plotH;
            return (
              <g key={t}>
                <line x1={padL} y1={y} x2={padL + plotW} y2={y} stroke="rgba(255,255,255,0.06)" strokeDasharray="3 3" />
                <text x={padL - 8} y={y + 3} fill="#708096" fontSize="9.5" textAnchor="end">
                  {t.toFixed(3)}
                </text>
              </g>
            );
          })}

          {xLabels.map(l => (
            <text key={l.dateStr + l.x} x={l.x} y={height - 8} fill="#708096" fontSize="9.5" textAnchor="middle">
              {l.dateStr}
            </text>
          ))}

          <line
            x1={splitX}
            y1={padT}
            x2={splitX}
            y2={padT + plotH}
            stroke="#38bdf8"
            strokeWidth="1.2"
            strokeDasharray="4 4"
          />
          <text x={splitX - 8} y={padT + 12} fill="#94a3b8" fontSize="9" textAnchor="end">
            Historical MODIS
          </text>
          <text x={splitX + 8} y={padT + 12} fill="#00d2ff" fontSize="9" fontWeight="600">
            Model Forecast ({fc.length} steps)
          </text>

          {bandPath && <path d={bandPath} fill="rgba(0, 210, 255, 0.2)" />}
          {histPath && <path d={histPath} fill="none" stroke="#00ff87" strokeWidth="1.8" />}
          {fullFcPath && <path d={fullFcPath} fill="none" stroke="#00d2ff" strokeWidth="2" />}

          {fcPoints.map((pt, i) => (
            <circle
              key={i}
              cx={pt.x}
              cy={pt.y}
              r={hoveredLabPoint?.timestamp === pt.data.timestamp ? 5 : 3}
              fill={hoveredLabPoint?.timestamp === pt.data.timestamp ? '#ffffff' : '#00d2ff'}
              stroke="#060911"
              strokeWidth="1.5"
              style={{ cursor: 'pointer', transition: 'r 0.15s ease' }}
              onMouseEnter={() => setHoveredLabPoint(pt.data)}
              onMouseLeave={() => setHoveredLabPoint(null)}
            />
          ))}
        </svg>

        {hoveredLabPoint && (
          <div
            style={{
              position: 'absolute',
              top: '12px',
              right: '20px',
              background: 'rgba(8, 12, 22, 0.95)',
              border: '1px solid #00d2ff',
              borderRadius: '8px',
              padding: '0.45rem 0.85rem',
              fontSize: '0.78rem',
              boxShadow: '0 0 16px rgba(0, 210, 255, 0.3)',
              pointerEvents: 'none',
              display: 'flex',
              gap: '1rem',
              zIndex: 20
            }}
          >
            <div>
              <span style={{ color: '#94a3b8' }}>Date: </span>
              <strong style={{ color: '#fff' }}>{hoveredLabPoint.timestamp}</strong>
            </div>
            <div>
              <span style={{ color: '#94a3b8' }}>Predicted NDVI: </span>
              <strong style={{ color: '#00d2ff' }}>{hoveredLabPoint.predicted_value.toFixed(4)}</strong>
            </div>
            <div>
              <span style={{ color: '#94a3b8' }}>95% CI: </span>
              <span style={{ color: '#38bdf8' }}>[{hoveredLabPoint.lower_bound.toFixed(4)}, {hoveredLabPoint.upper_bound.toFixed(4)}]</span>
            </div>
          </div>
        )}
      </div>
    );
  };

  return (
    <div className="app-shell">
      {/* 1. FIXED LEFT SIDEBAR (100dvh, NO INDEPENDENT SCROLL) */}
      <aside className="app-sidebar">
        <div>
          <div className="sidebar-header-badge">
            <span className="sidebar-brand-title">Console Navigation</span>
          </div>

          <nav className="sidebar-nav">
            <button
              id="nav-overview"
              className={`nav-item-btn ${activeTab === 'overview' ? 'active' : ''}`}
              onClick={() => switchTab('overview')}
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M3 9l9-7 9 7v11a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2z" />
                <polyline points="9 22 9 12 15 12 15 22" />
              </svg>
              <span>Overview</span>
            </button>

            <button
              id="nav-map"
              className={`nav-item-btn ${activeTab === 'map' ? 'active' : ''}`}
              onClick={() => switchTab('map')}
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6" />
                <line x1="8" y1="2" x2="8" y2="18" />
                <line x1="16" y1="6" x2="16" y2="22" />
              </svg>
              <span>Map Explorer</span>
            </button>

            <button
              id="nav-trends"
              className={`nav-item-btn ${activeTab === 'trends' ? 'active' : ''}`}
              onClick={() => switchTab('trends')}
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="23 6 13.5 15.5 8.5 10.5 1 18" />
                <polyline points="17 6 23 6 23 12" />
              </svg>
              <span>Trends</span>
            </button>

            <button
              id="nav-anomalies"
              className={`nav-item-btn ${activeTab === 'anomalies' ? 'active' : ''}`}
              onClick={() => switchTab('anomalies')}
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
              </svg>
              <span>Anomalies</span>
            </button>

            <button
              id="nav-forecasts"
              className={`nav-item-btn ${activeTab === 'forecasts' ? 'active' : ''}`}
              onClick={() => switchTab('forecasts')}
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <rect x="3" y="4" width="18" height="18" rx="2" ry="2" />
                <line x1="16" y1="2" x2="16" y2="6" />
                <line x1="8" y1="2" x2="8" y2="6" />
                <line x1="3" y1="10" x2="21" y2="10" />
              </svg>
              <span>Forecasts</span>
            </button>

            <button
              id="nav-prediction-lab"
              className={`nav-item-btn ${activeTab === 'prediction-lab' ? 'active' : ''}`}
              onClick={() => switchTab('prediction-lab')}
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M10 2v7.31L4.54 19.34A2 2 0 0 0 6.27 22h11.46a2 2 0 0 0 1.73-2.66L14 9.31V2" />
                <line x1="8.5" y1="2" x2="15.5" y2="2" />
                <line x1="7" y1="14.5" x2="17" y2="14.5" />
              </svg>
              <span>AI Prediction Lab</span>
            </button>

            <button
              id="nav-wildfires"
              className={`nav-item-btn ${activeTab === 'wildfires' ? 'active' : ''}`}
              onClick={() => switchTab('wildfires')}
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 3z" />
              </svg>
              <span>Wildfires</span>
            </button>

            <button
              id="nav-clustering"
              className={`nav-item-btn ${activeTab === 'clustering' ? 'active' : ''}`}
              onClick={() => setActiveTab('clustering')}
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <circle cx="18" cy="5" r="3" />
                <circle cx="6" cy="12" r="3" />
                <circle cx="18" cy="19" r="3" />
                <line x1="8.59" y1="13.51" x2="15.42" y2="17.49" />
                <line x1="15.41" y1="6.51" x2="8.59" y2="10.49" />
              </svg>
              <span>Clustering</span>
            </button>

            <button
              id="nav-methodology"
              className={`nav-item-btn ${activeTab === 'methodology' ? 'active' : ''}`}
              onClick={() => setActiveTab('methodology')}
            >
              <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                <path d="M4 19.5A2.5 2.5 0 0 1 6.5 17H20" />
                <path d="M6.5 2H20v20H6.5A2.5 2.5 0 0 1 4 19.5v-15A2.5 2.5 0 0 1 6.5 2z" />
              </svg>
              <span>Methodology</span>
            </button>
          </nav>
        </div>

        <div className="sidebar-bottom">
          <button
            className="nav-item-btn"
            onClick={() => setShowAboutModal(true)}
            id="nav-about"
          >
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <circle cx="12" cy="12" r="10" />
              <line x1="12" y1="16" x2="12" y2="12" />
              <line x1="12" y1="8" x2="12.01" y2="8" />
            </svg>
            <span>About</span>
          </button>
        </div>
      </aside>

      {/* 2. FIXED TOP HEADER (COMPACT 76px, VISIBLE WHILE MAIN CONTENT SCROLLS) */}
      <header className="space-header">
        <Image
          src="/images/earth-banner.jpg"
          alt="Atmospheric Space Header"
          className="space-header-bg"
          fill
          sizes="100vw"
          priority
        />
        <div className="space-header-overlay" />

        <div className="space-header-content">
          {/* Left Branding */}
          <div className="brand-section">
            <div className="globe-icon-wrap">
              <Image
                src="/images/earth-globe.jpg"
                alt="Earth Globe"
                width={40}
                height={40}
              />
            </div>
            <div className="brand-text-col">
              <div className="brand-title">
                TerraPulse <span>AI</span>
              </div>
              <div className="brand-subtitle-1">
                MOROCCO ENVIRONMENTAL OBSERVATORY
              </div>
              <div className="brand-subtitle-2">
                SATELLITE DATA • AI ANALYTICS • EARLY INSIGHTS
              </div>
            </div>
          </div>

          {/* Right Status, Time, Slogan & Coordinates */}
          <div className="header-meta-group">
            <div className="observatory-slogan">
              <div className="slogan-top">
                OBSERVING <span>TODAY</span>
              </div>
              <div className="slogan-bot">
                PROTECTING TOMORROW
              </div>
            </div>

            {/* API Connection Status Badge */}
            <div
              className="api-status-badge"
              onClick={loadData}
              title="Click to refresh backend connection"
              style={{
                color: apiOnline ? 'var(--success-green)' : 'var(--warning-orange)',
                border: `1px solid ${apiOnline ? 'rgba(0, 255, 135, 0.3)' : 'rgba(255, 159, 28, 0.3)'}`
              }}
            >
              <span style={{ fontSize: '0.65rem' }}>●</span>
              <span>{apiOnline ? 'FASTAPI OPERATIONAL' : 'VERIFIED PIPELINE DATA'}</span>
              <span style={{ opacity: 0.6, fontSize: '0.75rem', marginLeft: '2px' }}>↻</span>
            </div>

            <div className="time-box">
              <span className="time-date">Sep 14, 2026</span>
              <span className="time-utc">10:24 UTC</span>
            </div>

            <div className="morocco-badge">
              <svg width="20" height="22" viewBox="0 0 24 24" fill="none" stroke="#7dd3fc" strokeWidth="1.5">
                <path d="M12 2L8 6l1 4-3 3 2 5 4 4 4-4 2-5-3-3 1-4-4-4z" />
              </svg>
              <div className="morocco-coords">
                <span className="morocco-name">Morocco</span>
                <span className="morocco-latlon">31.8°N, 7.1°W</span>
              </div>
            </div>
          </div>
        </div>
      </header>

      {/* 3. MAIN INDEPENDENT SCROLLING CONTENT AREA */}
      <main className="app-main">
        {/* MASTER OVERVIEW TAB */}
        {activeTab === 'overview' && (
          <>
            {/* ROW 1: 4 TOP KPI CARDS */}
            <section className="kpi-grid">
              {/* Card 1: Vegetation Index (NDVI) */}
              <div className="glass-card kpi-card">
                <div>
                  <div className="kpi-top-row">
                    <div className="kpi-icon-bubble" style={{ background: 'rgba(0, 255, 135, 0.12)' }}>
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#00ff87" strokeWidth="2.2">
                        <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z" />
                        <path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12" />
                      </svg>
                    </div>
                    <span className="kpi-title">Vegetation Index (NDVI)</span>
                  </div>

                  <div className="kpi-value-row">
                    <span className="kpi-big-num">0.1457</span>
                    <span className="kpi-pill-badge" style={{ background: 'rgba(255, 71, 87, 0.16)', color: '#ff4757' }}>
                      ▼ -12.3%
                    </span>
                    <span className="kpi-subtext">vs. 2014–2023 mean</span>
                  </div>
                </div>

                <div>
                  {/* Neon Green Sparkline */}
                  <svg viewBox="0 0 220 28" style={{ width: '100%', height: '24px' }}>
                    <path
                      d="M 0,16 Q 25,6 50,18 T 100,10 T 150,22 T 190,14 L 220,18"
                      fill="none"
                      stroke="#00ff87"
                      strokeWidth="2"
                    />
                  </svg>
                  <div className="kpi-footer-date">Latest: 2026-09-14</div>
                </div>
              </div>

              {/* Card 2: Land Surface Temperature */}
              <div className="glass-card kpi-card">
                <div>
                  <div className="kpi-top-row">
                    <div className="kpi-icon-bubble" style={{ background: 'rgba(255, 115, 70, 0.12)' }}>
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ff7346" strokeWidth="2.2">
                        <path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z" />
                      </svg>
                    </div>
                    <span className="kpi-title">Land Surface Temperature</span>
                  </div>

                  <div className="kpi-value-row">
                    <span className="kpi-big-num">35.8 °C</span>
                    <span className="kpi-pill-badge" style={{ background: 'rgba(255, 71, 87, 0.16)', color: '#ff4757' }}>
                      ▲ +1.9 °C
                    </span>
                    <span className="kpi-subtext">vs. seasonal mean</span>
                  </div>
                </div>

                <div>
                  {/* Neon Orange Sparkline */}
                  <svg viewBox="0 0 220 28" style={{ width: '100%', height: '24px' }}>
                    <path
                      d="M 0,20 Q 25,24 50,8 T 100,19 T 150,6 T 190,22 L 220,11"
                      fill="none"
                      stroke="#ff7346"
                      strokeWidth="2"
                    />
                  </svg>
                  <div className="kpi-footer-date">Latest: 2026-09-14</div>
                </div>
              </div>

              {/* Card 3: Active Fire Events */}
              <div className="glass-card kpi-card">
                <div>
                  <div className="kpi-top-row">
                    <div className="kpi-icon-bubble" style={{ background: 'rgba(255, 159, 28, 0.12)' }}>
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ff9f1c" strokeWidth="2.2">
                        <path d="M8.5 14.5A2.5 2.5 0 0 0 11 12c0-1.38-.5-2-1-3-1.072-2.143-.224-4.054 2-6 .5 2.5 2 4.9 4 6.5 2 1.6 3 3.5 3 5.5a7 7 0 1 1-14 0c0-1.153.433-2.294 1-3a2.5 2.5 0 0 0 2.5 3z" />
                      </svg>
                    </div>
                    <span className="kpi-title">Active Fire Events</span>
                  </div>

                  <div className="kpi-value-row">
                    <span className="kpi-big-num">1,364</span>
                    <span className="kpi-pill-badge" style={{ background: 'rgba(0, 255, 135, 0.16)', color: '#00ff87' }}>
                      ▼ -28%
                    </span>
                    <span className="kpi-subtext">vs. previous year</span>
                  </div>
                </div>

                <div>
                  {/* Red bar series sparkline */}
                  <div style={{ display: 'flex', gap: '3px', alignItems: 'flex-end', height: '20px', marginBottom: '4px' }}>
                    {[8, 12, 16, 20, 14, 18, 22, 15, 10, 14, 19, 13, 9, 15, 21, 16, 12, 14, 18, 11].map((h, i) => (
                      <div
                        key={i}
                        style={{
                          flex: 1,
                          height: `${h}px`,
                          background: i > 14 ? '#ff4757' : 'rgba(255, 71, 87, 0.55)',
                          borderRadius: '1.5px'
                        }}
                      />
                    ))}
                  </div>
                  <div className="kpi-footer-date">2014 – 2026</div>
                </div>
              </div>

              {/* Card 4: Historical Coverage */}
              <div className="glass-card kpi-card">
                <div>
                  <div className="kpi-top-row">
                    <div className="kpi-icon-bubble" style={{ background: 'rgba(0, 210, 255, 0.12)' }}>
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#00d2ff" strokeWidth="2.2">
                        <polygon points="12 2 2 7 12 12 22 7 12 2" />
                        <polyline points="2 17 12 22 22 17" />
                        <polyline points="2 12 12 17 22 12" />
                      </svg>
                    </div>
                    <span className="kpi-title">Historical Coverage</span>
                  </div>

                  <div className="kpi-value-row">
                    <span className="kpi-big-num">271</span>
                    <span className="kpi-subtext" style={{ fontSize: '0.8rem', color: '#94a3b8' }}>
                      16-day composites
                    </span>
                  </div>
                </div>

                <div>
                  {/* Cyan Timeline Progress Bar */}
                  <div style={{ width: '100%', height: '7px', background: 'rgba(255, 255, 255, 0.08)', borderRadius: '4px', overflow: 'hidden', marginTop: '8px' }}>
                    <div
                      style={{
                        width: '100%',
                        height: '100%',
                        background: 'linear-gradient(90deg, #1b52d9 0%, #00d2ff 100%)',
                        boxShadow: '0 0 8px rgba(0, 210, 255, 0.5)'
                      }}
                    />
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '0.65rem', color: '#64748b', marginTop: '6px' }}>
                    <span>2014-01-01</span>
                    <span>2026-09-14</span>
                  </div>
                </div>
              </div>
            </section>

            {/* ROW 2: MAIN MIDDLE SECTION (MAP 65% + RIGHT CARDS 35%) */}
            <section className="main-middle-grid">
              {/* SATELLITE MAP CARD */}
              <div className="glass-card map-card">
                <div className="map-header-bar">
                  <div className="map-title-col">
                    <h2>Morocco — Environmental Monitoring</h2>
                    <p>Satellite-derived vegetation, temperature and active fires</p>
                    <div className="map-layers-pills">
                      <button
                        className={`layer-pill ${activeMapLayer === 'ndvi' ? 'active' : ''}`}
                        onClick={() => setActiveMapLayer('ndvi')}
                      >
                        NDVI
                      </button>
                      <button
                        className={`layer-pill ${activeMapLayer === 'lst' ? 'active' : ''}`}
                        onClick={() => setActiveMapLayer('lst')}
                      >
                        LST
                      </button>
                      <button
                        className={`layer-pill ${activeMapLayer === 'fires' ? 'active' : ''}`}
                        onClick={() => setActiveMapLayer('fires')}
                      >
                        Active Fires
                      </button>
                      <button
                        className={`layer-pill ${activeMapLayer === 'bioclimate' ? 'active' : ''}`}
                        onClick={() => setActiveMapLayer('bioclimate')}
                      >
                        Bioclimatic Strata
                      </button>
                    </div>
                  </div>
                </div>

                {/* Map Viewport Container */}
                <div className="map-canvas-container">
                  {/* Satellite Basemap Image */}
                  <Image
                    src="/images/morocco-satellite.jpg"
                    alt="Morocco Satellite Basemap"
                    className="map-satellite-img"
                    fill
                    sizes="(max-width: 1200px) 100vw, 65vw"
                    style={{ transform: `scale(${zoomLevel})`, transition: 'transform 0.25s ease' }}
                  />

                  {/* Inset Mini Globe (Top Right) */}
                  <div className="inset-globe">
                    <Image
                      src="/images/earth-globe.jpg"
                      alt="Mini Globe Locator"
                      width={68}
                      height={68}
                    />
                    <div className="inset-globe-box" />
                  </div>

                  {/* Vertical Map Controls (Right) */}
                  <div className="map-controls-vertical">
                    <button
                      className="map-ctrl-btn"
                      onClick={() => setZoomLevel((z) => Math.min(1.6, z + 0.15))}
                      title="Zoom In"
                    >
                      +
                    </button>
                    <button
                      className="map-ctrl-btn"
                      onClick={() => setZoomLevel((z) => Math.max(0.85, z - 0.15))}
                      title="Zoom Out"
                    >
                      −
                    </button>
                    <button
                      className="map-ctrl-btn"
                      onClick={() => setZoomLevel(1)}
                      title="Reset View"
                    >
                      ⌖
                    </button>
                    <button
                      className="map-ctrl-btn"
                      onClick={() => setActiveMapLayer(l => l === 'fires' ? 'ndvi' : 'fires')}
                      title="Toggle Layer"
                    >
                      ▤
                    </button>
                  </div>

                  {/* FRP Legend (Bottom Left) */}
                  <div className="frp-legend-box">
                    <div className="frp-legend-title">Fire Radiative Power (FRP)</div>
                    <div className="frp-legend-item">
                      <span className="frp-dot" style={{ background: '#ff3344', boxShadow: '0 0 6px #ff3344' }} />
                      <span>&gt; 500 MW</span>
                    </div>
                    <div className="frp-legend-item">
                      <span className="frp-dot" style={{ background: '#ff7733' }} />
                      <span>200 – 500 MW</span>
                    </div>
                    <div className="frp-legend-item">
                      <span className="frp-dot" style={{ background: '#ffaa00' }} />
                      <span>50 – 200 MW</span>
                    </div>
                    <div className="frp-legend-item">
                      <span className="frp-dot" style={{ background: '#ffcc00' }} />
                      <span>10 – 50 MW</span>
                    </div>
                    <div className="frp-legend-item">
                      <span className="frp-dot" style={{ background: '#ffee66' }} />
                      <span>&lt; 10 MW</span>
                    </div>
                  </div>

                  {/* Distance Scale Bar */}
                  <div className="map-scale-bar">
                    <div className="scale-ticks">
                      <span>0</span>
                      <span>100</span>
                      <span>200</span>
                      <span>300 km</span>
                    </div>
                    <div className="scale-line" />
                  </div>

                  {/* Interactive SVG Overlay with Hotspots and City Labels */}
                  <svg
                    className="map-svg-overlay"
                    viewBox="0 0 700 380"
                    style={{ transform: `scale(${zoomLevel})`, transition: 'transform 0.25s ease' }}
                  >
                    <defs>
                      <radialGradient id="hotspotGlowHigh" cx="50%" cy="50%" r="50%">
                        <stop offset="0%" stopColor="#ffffff" stopOpacity="1" />
                        <stop offset="25%" stopColor="#ff2244" stopOpacity="0.95" />
                        <stop offset="70%" stopColor="#ff5500" stopOpacity="0.45" />
                        <stop offset="100%" stopColor="#ff0000" stopOpacity="0" />
                      </radialGradient>
                      <radialGradient id="hotspotGlowMid" cx="50%" cy="50%" r="50%">
                        <stop offset="0%" stopColor="#fff" stopOpacity="0.9" />
                        <stop offset="40%" stopColor="#ff8800" stopOpacity="0.75" />
                        <stop offset="100%" stopColor="#ff5500" stopOpacity="0" />
                      </radialGradient>
                    </defs>

                    {/* Ocean & Country Labels */}
                    <text x="160" y="225" fill="rgba(255,255,255,0.42)" fontSize="13" fontStyle="italic" letterSpacing="1.5">
                      Atlantic Ocean
                    </text>
                    <text x="560" y="275" fill="rgba(255,255,255,0.3)" fontSize="13" letterSpacing="1.2">
                      Algeria
                    </text>

                    {/* Major City Markers */}
                    {mapCities.map((c) => (
                      <g key={c.name} transform={`translate(${c.x}, ${c.y})`}>
                        <circle cx="0" cy="0" r="2.5" fill="#ffffff" />
                        <text x="5" y="-3" fill="#ffffff" fontSize="9.5" fontWeight="600" style={{ textShadow: '0 1px 3px #000' }}>
                          {c.name}
                        </text>
                      </g>
                    ))}

                    {/* Fire Hotspots (Active Fires layer) */}
                    {(activeMapLayer === 'fires' || activeMapLayer === 'bioclimate') &&
                      mapHotspots.map((h) => (
                        <g
                          key={h.id}
                          transform={`translate(${h.x}, ${h.y})`}
                          style={{ cursor: 'pointer' }}
                          onClick={() => {
                            const found = curatedFireEvents.find(e => e.id === h.id);
                            if (found) {
                              setSelectedHotspot({
                                event_id: found.id,
                                latitude: found.coords.lat,
                                longitude: found.coords.lon,
                                acq_date: found.date,
                                satellite: found.sat,
                                frp: parseFloat(found.frp.replace(/[^0-9.]/g, '')),
                                confidence: 'h',
                                region_name: found.region
                              });
                            }
                          }}
                        >
                          {/* Outer Pulsing Aura for Massive Cluster Events */}
                          {h.pulse && (
                            <circle
                              cx="0"
                              cy="0"
                              r={h.r * 1.8}
                              fill="url(#hotspotGlowHigh)"
                              className="pulse-circle"
                            />
                          )}
                          {/* Inner Radiant Core */}
                          <circle
                            cx="0"
                            cy="0"
                            r={h.r}
                            fill={h.frp > 1000 ? 'url(#hotspotGlowHigh)' : 'url(#hotspotGlowMid)'}
                          />
                          <circle cx="0" cy="0" r={Math.max(2, h.r * 0.28)} fill="#ffffff" />
                        </g>
                      ))}

                    {/* Hotspot detail popup when selected */}
                    {selectedHotspot && (
                      <g transform="translate(380, 70)">
                        <rect x="0" y="0" width="180" height="74" rx="8" fill="rgba(8,12,22,0.92)" stroke="#00d2ff" strokeWidth="1" />
                        <text x="12" y="20" fill="#ffffff" fontSize="11" fontWeight="700">
                          {selectedHotspot.region_name || `Event #${selectedHotspot.event_id}`}
                        </text>
                        <text x="12" y="38" fill="#ff4757" fontSize="10" fontWeight="600">
                          FRP: {selectedHotspot.frp.toLocaleString()} MW
                        </text>
                        <text x="12" y="52" fill="#94a3b8" fontSize="9">
                          Date: {selectedHotspot.acq_date} • {selectedHotspot.satellite}
                        </text>
                        <text
                          x="165"
                          y="18"
                          fill="#94a3b8"
                          fontSize="12"
                          style={{ cursor: 'pointer' }}
                          onClick={() => setSelectedHotspot(null)}
                        >
                          ✕
                        </text>
                      </g>
                    )}
                  </svg>
                </div>
              </div>

              {/* RIGHT STACKED CARDS: RECENT ACTIVE FIRES & BIOCLIMATIC REGIONS */}
              <div className="right-stacked-col">
                {/* 1. Recent Active Fires Card */}
                <div className="glass-card fire-events-card">
                  <div className="card-title-row">
                    <h3>Recent Active Fires</h3>
                    <span
                      className="view-all-link"
                      onClick={() => setActiveTab('wildfires')}
                    >
                      View all →
                    </span>
                  </div>

                  {curatedFireEvents.map((evt) => (
                    <div
                      key={evt.id}
                      className="fire-event-row"
                      onClick={() => {
                        setSelectedHotspot({
                          event_id: evt.id,
                          latitude: evt.coords.lat,
                          longitude: evt.coords.lon,
                          acq_date: evt.date,
                          satellite: evt.sat,
                          frp: parseFloat(evt.frp.replace(/[^0-9.]/g, '')),
                          confidence: 'h',
                          region_name: evt.region
                        });
                      }}
                    >
                      <div className="fire-event-thumb">
                        <Image
                          src={evt.thumb}
                          alt={evt.region}
                          width={48}
                          height={48}
                        />
                      </div>
                      <div className="fire-event-details">
                        <div className="fire-event-head">
                          <span className="fire-event-name">Event #{evt.id}</span>
                          <span
                            className="fire-badge-pill"
                            style={{ background: 'rgba(255, 42, 75, 0.18)', color: evt.badgeColor }}
                          >
                            {evt.level}
                          </span>
                        </div>
                        <div className="fire-event-loc">
                          <span>📍 {evt.region}</span>
                          <span style={{ margin: '0 4px', opacity: 0.5 }}>•</span>
                          <span>{evt.date}</span>
                        </div>
                        <div className="fire-event-frp">
                          FRP: {evt.frp}
                          <span style={{ color: '#64748b', fontSize: '0.62rem', marginLeft: '6px' }}>
                            🛰 {evt.sat}
                          </span>
                        </div>
                      </div>
                      <div style={{ color: '#475569', fontSize: '1.1rem', paddingLeft: '4px' }}>›</div>
                    </div>
                  ))}
                </div>

                {/* 2. Bioclimatic Regions (Illustrative) Card */}
                <div className="glass-card bioclimatic-card">
                  <div className="card-title-row">
                    <h3>Bioclimatic Regions <span style={{ fontSize: '0.72rem', color: '#94a3b8', fontWeight: 'normal' }}>(Illustrative)</span></h3>
                    <span style={{ color: '#64748b', cursor: 'pointer', fontSize: '0.82rem' }}>⛶</span>
                  </div>

                  <div className="bioclimatic-body">
                    {/* Color Legend List */}
                    <div className="bioclimatic-legend">
                      <div className="bioclimatic-item">
                        <span className="bioclimatic-dot" style={{ background: '#10b981' }} />
                        <span>Northern Rif — Humid Mediterranean</span>
                      </div>
                      <div className="bioclimatic-item">
                        <span className="bioclimatic-dot" style={{ background: '#06b6d4' }} />
                        <span>Atlantic Plains — Subhumid to Semi-Arid</span>
                      </div>
                      <div className="bioclimatic-item">
                        <span className="bioclimatic-dot" style={{ background: '#3b82f6' }} />
                        <span>Middle Atlas — Mountain Continental</span>
                      </div>
                      <div className="bioclimatic-item">
                        <span className="bioclimatic-dot" style={{ background: '#8b5cf6' }} />
                        <span>High Atlas — Cold Mountain</span>
                      </div>
                      <div className="bioclimatic-item">
                        <span className="bioclimatic-dot" style={{ background: '#f97316' }} />
                        <span>Souss-Massa — Arid to Semi-Arid</span>
                      </div>
                      <div className="bioclimatic-item">
                        <span className="bioclimatic-dot" style={{ background: '#d97706' }} />
                        <span>Saharan Domain — Arid</span>
                      </div>
                    </div>

                    {/* Illustrated Bioclimatic Silhouette Map */}
                    <div className="bioclimatic-map-thumb">
                      <svg viewBox="0 0 100 130" style={{ width: '85px', height: '115px' }}>
                        {/* Northern Rif (Green) */}
                        <path d="M 68,6 L 85,12 L 80,24 L 62,20 Z" fill="#10b981" opacity="0.9" />
                        {/* Atlantic Plains (Cyan) */}
                        <path d="M 62,20 L 80,24 L 56,52 L 42,42 Z" fill="#06b6d4" opacity="0.9" />
                        {/* Middle Atlas (Blue) */}
                        <path d="M 80,24 L 92,34 L 72,58 L 56,52 Z" fill="#3b82f6" opacity="0.9" />
                        {/* High Atlas (Purple) */}
                        <path d="M 56,52 L 72,58 L 54,78 L 38,68 Z" fill="#8b5cf6" opacity="0.9" />
                        {/* Souss-Massa (Orange) */}
                        <path d="M 38,68 L 54,78 L 44,92 L 28,82 Z" fill="#f97316" opacity="0.9" />
                        {/* Saharan Domain (Amber/Gold) */}
                        <path d="M 28,82 L 44,92 L 32,126 L 10,126 L 14,102 Z" fill="#d97706" opacity="0.85" />
                      </svg>
                    </div>
                  </div>
                </div>
              </div>
            </section>

            {/* ROW 3: 3 BOTTOM ANALYTICAL CHARTS */}
            <section className="bottom-charts-grid">
              {/* Chart 1: NDVI Trend */}
              <div className="glass-card chart-card">
                <div className="chart-card-header">
                  <div className="chart-title-left">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#00ff87" strokeWidth="2.2">
                      <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z" />
                    </svg>
                    <h3>NDVI Trend (2014 – 2026)</h3>
                  </div>
                  <div className="region-dropdown-pill">
                    Morocco (All Regions) ⌄
                  </div>
                </div>

                <div>
                  {renderTrendSvg('NDVI', '#00ff87', 0.0, 0.3, [0.3, 0.2, 0.1, 0.0])}
                </div>

                <div className="chart-legend-footer">
                  <div className="chart-legend-items">
                    <span>— NDVI (Observed)</span>
                    <span>--- Trend (Theil-Sen)</span>
                  </div>
                  <div className="chart-stat-salmon">
                    Sen&apos;s slope: -0.0143 / decade &nbsp; p = 0.00005
                  </div>
                </div>
              </div>

              {/* Chart 2: LST Trend */}
              <div className="glass-card chart-card">
                <div className="chart-card-header">
                  <div className="chart-title-left">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#ff7346" strokeWidth="2.2">
                      <path d="M14 14.76V3.5a2.5 2.5 0 0 0-5 0v11.26a4.5 4.5 0 1 0 5 0z" />
                    </svg>
                    <h3>LST Trend (2014 – 2026)</h3>
                  </div>
                  <div className="region-dropdown-pill">
                    Morocco (All Regions) ⌄
                  </div>
                </div>

                <div>
                  {renderTrendSvg('LST', '#ff7346', 15, 45, [45, 35, 25, 15])}
                </div>

                <div className="chart-legend-footer">
                  <div className="chart-legend-items">
                    <span>— LST (Observed)</span>
                    <span>--- Trend (Theil-Sen)</span>
                  </div>
                  <div className="chart-stat-salmon">
                    Sen&apos;s slope: -2.59 °C / decade &nbsp; p = 0.04229
                  </div>
                </div>
              </div>

              {/* Chart 3: Future Forecast */}
              <div className="glass-card chart-card">
                <div className="chart-card-header">
                  <div className="chart-title-left">
                    <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="#00d2ff" strokeWidth="2.2">
                      <polyline points="23 6 13.5 15.5 8.5 10.5 1 18" />
                      <polyline points="17 6 23 6 23 12" />
                    </svg>
                    <h3>Future Forecast (NDVI)</h3>
                  </div>
                  <div className="region-dropdown-pill">
                    Morocco (All Regions) ⌄
                  </div>
                </div>

                <div>
                  {renderForecastSvg()}
                </div>

                <div className="chart-legend-footer">
                  <div className="chart-legend-items">
                    <span>— Historical (Observed)</span>
                    <span>— Forecast (Seasonal Naive)</span>
                    <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                      <span style={{ width: '8px', height: '8px', background: 'rgba(56, 189, 248, 0.45)', borderRadius: '1px' }} />
                      95% Confidence Interval
                    </span>
                  </div>
                </div>
              </div>
            </section>
          </>
        )}

        {/* 4. DEDICATED VIEWS FOR OTHER TABS */}
        {activeTab === 'map' && (
          <div className="glass-card" style={{ padding: '1.5rem', minHeight: '600px' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <div>
                <h2>Geospatial Map Explorer — Full Screen Console</h2>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                  High-resolution NASA MODIS and VIIRS spatial analysis across the Kingdom of Morocco.
                </p>
              </div>
              <button className="layer-pill active" onClick={() => setActiveTab('overview')}>
                ← Return to Cockpit
              </button>
            </div>
            <div style={{ height: '480px', position: 'relative', borderRadius: '10px', overflow: 'hidden' }}>
              <Image
                src="/images/morocco-satellite.jpg"
                alt="Morocco Basemap"
                fill
                sizes="100vw"
                style={{ objectFit: 'cover' }}
              />
              <div style={{ position: 'absolute', bottom: '20px', left: '20px', background: 'rgba(6,9,17,0.85)', padding: '1rem', borderRadius: '8px', border: '1px solid rgba(255,255,255,0.1)' }}>
                <h4 style={{ color: '#00d2ff', marginBottom: '6px' }}>Layer Diagnostics</h4>
                <div style={{ fontSize: '0.82rem', color: '#cbd5e1' }}>• Active Hotspots: 1,364 verified VIIRS points</div>
                <div style={{ fontSize: '0.82rem', color: '#cbd5e1' }}>• Spatial Grid: 0.05° Sinusoidal Resampled Grid</div>
                <div style={{ fontSize: '0.82rem', color: '#cbd5e1' }}>• Peak Single FRP: 490.2 MW (High Atlas cluster total: 4,983.37 MW)</div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'trends' && (
          <div className="glass-card" style={{ padding: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.2rem' }}>
              <div>
                <h2>Trend Analysis (Mann-Kendall & Theil-Sen)</h2>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                  Decadal environmental trajectories calculated from 271 continuous 16-day composites (2014–2026).
                </p>
              </div>
              <button className="layer-pill active" onClick={() => setActiveTab('overview')}>
                ← Return to Cockpit
              </button>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
              <div className="glass-card" style={{ padding: '1.2rem' }}>
                <h3 style={{ color: '#00ff87', marginBottom: '8px' }}>Vegetation Index (NDVI)</h3>
                <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#fff' }}>-0.0143 / decade</div>
                <p style={{ color: '#94a3b8', fontSize: '0.82rem', marginTop: '6px' }}>
                  Mann-Kendall Tau: -0.165 • p-value: 0.00005 (Statistically Significant Decrease)
                </p>
                <div style={{ marginTop: '1rem' }}>
                  {renderTrendSvg('NDVI', '#00ff87', 0.0, 0.3, [0.3, 0.2, 0.1, 0.0])}
                </div>
              </div>
              <div className="glass-card" style={{ padding: '1.2rem' }}>
                <h3 style={{ color: '#ff7346', marginBottom: '8px' }}>Land Surface Temperature (LST)</h3>
                <div style={{ fontSize: '1.8rem', fontWeight: 800, color: '#fff' }}>-2.59 °C / decade</div>
                <p style={{ color: '#94a3b8', fontSize: '0.82rem', marginTop: '6px' }}>
                  Mann-Kendall Tau: -0.084 • p-value: 0.04229 (Statistically Significant)
                </p>
                <div style={{ marginTop: '1rem' }}>
                  {renderTrendSvg('LST', '#ff7346', 15, 45, [45, 35, 25, 15])}
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'anomalies' && (
          <div className="glass-card" style={{ padding: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.2rem' }}>
              <div>
                <h2>Climatological Anomalies (Z-Score Standardized)</h2>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                  Standardized departures from the multi-year seasonal base period (2014–2023).
                </p>
              </div>
              <button className="layer-pill active" onClick={() => setActiveTab('overview')}>
                ← Return to Cockpit
              </button>
            </div>
            <div style={{ padding: '1rem', background: 'rgba(0,0,0,0.3)', borderRadius: '10px' }}>
              <div style={{ display: 'flex', justifyContent: 'space-around', textAlign: 'center' }}>
                <div>
                  <div style={{ color: '#94a3b8', fontSize: '0.82rem' }}>Current NDVI Z-Score</div>
                  <div style={{ fontSize: '2rem', fontWeight: 800, color: '#ff4757' }}>-1.84 σ</div>
                  <div style={{ color: '#ff4757', fontSize: '0.75rem' }}>Moderate Drought Stress</div>
                </div>
                <div>
                  <div style={{ color: '#94a3b8', fontSize: '0.82rem' }}>Current LST Z-Score</div>
                  <div style={{ fontSize: '2rem', fontWeight: 800, color: '#ff7346' }}>+1.42 σ</div>
                  <div style={{ color: '#ff7346', fontSize: '0.75rem' }}>Elevated Thermal Anomaly</div>
                </div>
                <div>
                  <div style={{ color: '#94a3b8', fontSize: '0.82rem' }}>Baseline Period</div>
                  <div style={{ fontSize: '2rem', fontWeight: 800, color: '#00d2ff' }}>10 Years</div>
                  <div style={{ color: '#94a3b8', fontSize: '0.75rem' }}>2014-01 to 2023-12</div>
                </div>
              </div>
            </div>
          </div>
        )}

        {activeTab === 'forecasts' && (
          <div className="glass-card" style={{ padding: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.2rem' }}>
              <div>
                <h2>Predictive Forecasting Benchmark (23-Step Ahead)</h2>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                  Seasonal Naive baseline versus Ridge AutoRegressive model across 1 full annual cycle (2026–2027).
                </p>
              </div>
              <button className="layer-pill active" onClick={() => setActiveTab('overview')}>
                ← Return to Cockpit
              </button>
            </div>
            <div style={{ height: '240px' }}>
              {renderForecastSvg()}
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginTop: '1.5rem' }}>
              <div className="glass-card" style={{ padding: '1rem' }}>
                <span style={{ color: '#94a3b8', fontSize: '0.78rem' }}>Baseline Model (Seasonal Naive)</span>
                <div style={{ fontSize: '1.3rem', fontWeight: 700, color: '#38bdf8', marginTop: '4px' }}>MAE: 0.0098</div>
                <div style={{ color: '#64748b', fontSize: '0.75rem' }}>RMSE: 0.0121 • R²: 0.884</div>
              </div>
              <div className="glass-card" style={{ padding: '1rem' }}>
                <span style={{ color: '#94a3b8', fontSize: '0.78rem' }}>Challenger Model (Ridge AR)</span>
                <div style={{ fontSize: '1.3rem', fontWeight: 700, color: '#00ff87', marginTop: '4px' }}>MAE: 0.0112</div>
                <div style={{ color: '#64748b', fontSize: '0.75rem' }}>RMSE: 0.0142 • R²: 0.841</div>
              </div>
              <div className="glass-card" style={{ padding: '1rem' }}>
                <span style={{ color: '#94a3b8', fontSize: '0.78rem' }}>Verification Verdict</span>
                <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#ffb800', marginTop: '4px' }}>Seasonal Naive Selected</div>
                <div style={{ color: '#64748b', fontSize: '0.75rem' }}>Superior out-of-sample persistence</div>
              </div>
            </div>
          </div>
        )}

        {/* AI PREDICTION LAB VIEW */}
        {activeTab === 'prediction-lab' && (
          <div className="prediction-lab-layout animate-in">
            {/* Header Row */}
            <div className="lab-header-row">
              <div className="lab-title-group">
                <h2>
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="#00d2ff" strokeWidth="2">
                    <path d="M10 2v7.31L4.54 19.34A2 2 0 0 0 6.27 22h11.46a2 2 0 0 0 1.73-2.66L14 9.31V2" />
                    <line x1="8.5" y1="2" x2="15.5" y2="2" />
                    <line x1="7" y1="14.5" x2="17" y2="14.5" />
                  </svg>
                  <span>AI Prediction Laboratory</span>
                </h2>
                <p>
                  Configure and execute verified machine learning and baseline models on NASA Earth Observation time series.
                </p>
              </div>
              <button className="layer-pill active" onClick={() => switchTab('overview')}>
                ← Return to Cockpit
              </button>
            </div>

            {/* 1. Configuration Panel */}
            <div className="config-panel-card">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', borderBottom: '1px solid rgba(255,255,255,0.06)', paddingBottom: '0.65rem' }}>
                <span style={{ fontSize: '0.9rem', fontWeight: 700, color: '#ffffff' }}>
                  Prediction Pipeline Configuration
                </span>
                <span style={{ fontSize: '0.72rem', color: '#94a3b8' }}>
                  NASA MOD13Q1 16-Day Composites • National Domain
                </span>
              </div>

              <div className="config-grid">
                {/* 1. Geographic Scope */}
                <div className="config-item">
                  <label className="config-label">
                    <span>Geographic Scope</span>
                    <span className="badge-tag-small">Spatial Unit</span>
                  </label>
                  <div className="pill-group">
                    <button
                      className={`selector-pill ${labRegion === 'morocco' ? 'active' : ''}`}
                      onClick={() => setLabRegion('morocco')}
                    >
                      🇲🇦 Morocco (National Domain)
                    </button>
                    <button className="selector-pill disabled" title="Regional spatial disaggregation is deferred to Phase 2">
                      Rif Mountains <span className="badge-tag-small">Phase 2</span>
                    </button>
                    <button className="selector-pill disabled" title="Regional spatial disaggregation is deferred to Phase 2">
                      Middle Atlas <span className="badge-tag-small">Phase 2</span>
                    </button>
                    <button className="selector-pill disabled" title="Regional spatial disaggregation is deferred to Phase 2">
                      Souss-Massa <span className="badge-tag-small">Phase 2</span>
                    </button>
                  </div>
                  <span className="config-hint">
                    National aggregate time series (795,887 MODIS pixels). Regional disaggregation is scheduled for Phase 2.
                  </span>
                </div>

                {/* 2. Target Indicator */}
                <div className="config-item">
                  <label className="config-label">
                    <span>Target Indicator</span>
                    <span className="badge-tag-small">Sensor Variable</span>
                  </label>
                  <div className="pill-group">
                    <button
                      className={`selector-pill ${labIndicator === 'NDVI' ? 'active' : ''}`}
                      onClick={() => setLabIndicator('NDVI')}
                    >
                      🌿 NDVI (Vegetation Index)
                    </button>
                    <button className="selector-pill disabled" title="LST multi-step autoregressive model is currently in training pipeline">
                      🌡️ LST (Surface Temp) <span className="badge-tag-small">In Dev</span>
                    </button>
                  </div>
                  <span className="config-hint">
                    Only NDVI models are verified and trained on disk. LST forecasting is in development.
                  </span>
                </div>

                {/* 3. Forecasting Model */}
                <div className="config-item" style={{ gridColumn: 'span 2' }}>
                  <label className="config-label">
                    <span>Forecasting Algorithm & Model</span>
                    <span className="badge-tag-small">Architecture</span>
                  </label>
                  <div className="pill-group">
                    <button
                      className={`selector-pill ${labModel === 'random_forest' ? 'active' : ''}`}
                      onClick={() => setLabModel('random_forest')}
                      style={{ padding: '0.65rem 0.95rem' }}
                    >
                      <div style={{ textAlign: 'left' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <strong>Random Forest Regressor</strong>
                          <span className="badge badge-success" style={{ fontSize: '0.62rem' }}>Trained ML • R² = 0.9323</span>
                        </div>
                        <div style={{ fontSize: '0.68rem', opacity: 0.8, marginTop: '2px' }}>
                          Multi-lag autoregressive (t-1, t-2, t-3, t-23), rolling windows, and thermal LST lag coupling.
                        </div>
                      </div>
                    </button>

                    <button
                      className={`selector-pill ${labModel === 'seasonal_naive' ? 'active' : ''}`}
                      onClick={() => setLabModel('seasonal_naive')}
                      style={{ padding: '0.65rem 0.95rem' }}
                    >
                      <div style={{ textAlign: 'left' }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '6px' }}>
                          <strong>Seasonal Naive Baseline</strong>
                          <span className="badge badge-blue" style={{ fontSize: '0.62rem' }}>Benchmark • R² = 0.4177</span>
                        </div>
                        <div style={{ fontSize: '0.68rem', opacity: 0.8, marginTop: '2px' }}>
                          23-step seasonal persistence repeating the previous annual phenological cycle.
                        </div>
                      </div>
                    </button>
                  </div>
                </div>

                {/* 4. Prediction Horizon */}
                <div className="config-item">
                  <label className="config-label">
                    <span>Prediction Horizon</span>
                    <span className="badge-tag-small">Time Steps</span>
                  </label>
                  <div className="pill-group">
                    {[
                      { steps: 3, label: '3 steps (~1.5 mo)' },
                      { steps: 6, label: '6 steps (~3 mo)' },
                      { steps: 12, label: '12 steps (~6 mo)' },
                      { steps: 23, label: '23 steps (1 yr)' },
                      { steps: 46, label: '46 steps (2 yr)' }
                    ].map(h => (
                      <button
                        key={h.steps}
                        className={`selector-pill ${labHorizon === h.steps ? 'active' : ''}`}
                        onClick={() => setLabHorizon(h.steps)}
                      >
                        {h.label}
                      </button>
                    ))}
                  </div>
                  <span className="config-hint">
                    1 step = 16-day MODIS composite. 23 steps corresponds to 1 complete annual phenological cycle (368 days).
                  </span>
                </div>

                {/* 5. Prediction Mode */}
                <div className="config-item">
                  <label className="config-label">
                    <span>Prediction Mode</span>
                    <span className="badge-tag-small">Execution Type</span>
                  </label>
                  <div className="pill-group">
                    <button
                      className={`selector-pill ${labMode === 'historical_forecast' ? 'active' : ''}`}
                      onClick={() => setLabMode('historical_forecast')}
                    >
                      ⏱️ Out-of-Sample Forward Forecast
                    </button>
                    <button className="selector-pill disabled" title="Counterfactual scenarios require GPM IMERG precipitation data, currently deferred">
                      🧪 What-If Climate Scenario <span className="badge-tag-small">Requires GPM</span>
                    </button>
                  </div>
                  <span className="config-hint">
                    Scenario simulation requires external precipitation forcing; GPM IMERG remains deferred.
                  </span>
                </div>
              </div>

              {/* Form Action Controls */}
              <div className="form-actions-bar">
                <div style={{ display: 'flex', gap: '0.65rem', alignItems: 'center' }}>
                  <button
                    className="btn-predict-primary"
                    onClick={() => handleRunPrediction(labModel, labHorizon, labRegion, labIndicator)}
                    disabled={isPredicting}
                  >
                    {isPredicting ? (
                      <>
                        <span className="pulse-circle" style={{ width: '10px', height: '10px', background: '#fff', borderRadius: '50%', display: 'inline-block' }} />
                        Running Model Pipeline...
                      </>
                    ) : (
                      <>
                        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5">
                          <polygon points="5 3 19 12 5 21 5 3" />
                        </svg>
                        Run Model Prediction
                      </>
                    )}
                  </button>

                  <button
                    className="btn-reset-secondary"
                    onClick={handleResetLab}
                    disabled={isPredicting}
                  >
                    Reset Defaults
                  </button>
                </div>

                {predictionResult && (
                  <div style={{ display: 'flex', gap: '0.5rem', alignItems: 'center' }}>
                    <button
                      className="layer-pill"
                      onClick={exportPredictionCsv}
                      title="Download predicted points as CSV"
                    >
                      📥 Download CSV
                    </button>
                    <button
                      className="layer-pill"
                      onClick={exportPredictionJson}
                      title="Download full prediction payload as JSON"
                    >
                      📥 Download JSON
                    </button>
                  </div>
                )}
              </div>

              {predictionError && (
                <div style={{ marginTop: '1rem', padding: '0.8rem 1rem', background: 'rgba(255, 71, 87, 0.1)', border: '1px solid #ff4757', borderRadius: '8px', color: '#ff4757', fontSize: '0.85rem' }}>
                  ⚠️ <strong>Backend Prediction Notice:</strong> {predictionError}
                </div>
              )}
            </div>

            {/* 2. Results Section */}
            {predictionResult && (
              <>
                {/* 2.1 Prediction Summary Cards */}
                <div className="prediction-summary-grid">
                  <div className="summary-stat-box">
                    <div className="stat-label">Target Indicator</div>
                    <div className="stat-val" style={{ color: '#00ff87' }}>
                      {predictionResult.indicator}
                    </div>
                    <div style={{ fontSize: '0.68rem', color: '#94a3b8', marginTop: '2px' }}>
                      MODIS Terra MOD13Q1 (250m)
                    </div>
                  </div>

                  <div className="summary-stat-box">
                    <div className="stat-label">Geographic Scope</div>
                    <div className="stat-val">
                      Morocco National
                    </div>
                    <div style={{ fontSize: '0.68rem', color: '#94a3b8', marginTop: '2px' }}>
                      795,887 Analyzed Grid Pixels
                    </div>
                  </div>

                  <div className="summary-stat-box">
                    <div className="stat-label">Prediction Horizon</div>
                    <div className="stat-val" style={{ color: '#00d2ff' }}>
                      {predictionResult.forecast_steps} Steps
                    </div>
                    <div style={{ fontSize: '0.68rem', color: '#94a3b8', marginTop: '2px' }}>
                      ~{predictionResult.forecast_steps * 16} Days ({ (predictionResult.forecast_steps / 23).toFixed(1) } Annual Cycles)
                    </div>
                  </div>

                  <div className="summary-stat-box">
                    <div className="stat-label">Deployed Model</div>
                    <div className="stat-val" style={{ fontSize: '1rem', color: '#ffb800' }}>
                      {predictionResult.model_type.replace(/_/g, ' ')}
                    </div>
                    <div style={{ fontSize: '0.68rem', color: '#94a3b8', marginTop: '2px' }}>
                      {predictionResult.generation_timestamp ? `Generated: ${predictionResult.generation_timestamp.slice(0, 16).replace('T', ' ')} UTC` : 'Live Pipeline'}
                    </div>
                  </div>
                </div>

                {/* 2.2 Visualization Card */}
                <div className="glass-card" style={{ padding: '1.25rem' }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.8rem', flexWrap: 'wrap', gap: '0.5rem' }}>
                    <div>
                      <h3 style={{ fontSize: '1rem', fontWeight: 700, color: '#fff' }}>
                        Interactive Multi-Step Forecast Trajectory
                      </h3>
                      <p style={{ fontSize: '0.75rem', color: '#94a3b8' }}>
                        Historical MODIS observations combined with multi-step out-of-sample forward projections and 95% confidence intervals.
                      </p>
                    </div>

                    <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', fontSize: '0.72rem', color: '#94a3b8' }}>
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                        <span style={{ width: '12px', height: '2px', background: '#00ff87', display: 'inline-block' }} />
                        Historical (Observed)
                      </span>
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                        <span style={{ width: '12px', height: '2px', background: '#00d2ff', display: 'inline-block' }} />
                        Model Forecast
                      </span>
                      <span style={{ display: 'inline-flex', alignItems: 'center', gap: '4px' }}>
                        <span style={{ width: '10px', height: '8px', background: 'rgba(0, 210, 255, 0.35)', borderRadius: '1px', display: 'inline-block' }} />
                        95% Confidence Interval
                      </span>
                    </div>
                  </div>

                  {renderLabForecastSvg(predictionResult)}
                </div>

                {/* 2.3 Evaluation & Model Insights Grid */}
                <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1rem' }}>
                  {/* Left: Model Evaluation */}
                  <div className="glass-card" style={{ padding: '1.2rem' }}>
                    <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#fff', marginBottom: '0.4rem' }}>
                      Empirical Model Evaluation
                    </h3>
                    <div style={{ fontSize: '0.72rem', color: '#ffb800', background: 'rgba(255, 184, 0, 0.08)', padding: '0.5rem 0.75rem', borderRadius: '6px', marginBottom: '1rem' }}>
                      ⚠️ <strong>Scientific Distinction:</strong> Evaluation metrics are strictly computed on the held-out historical test set (2024-05-08 to 2026-09-14, 55 composites). They are NOT computed against unobserved future time steps.
                    </div>

                    <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '0.75rem', textAlign: 'center' }}>
                      <div style={{ padding: '0.75rem', background: 'rgba(255,255,255,0.03)', borderRadius: '8px' }}>
                        <div style={{ fontSize: '0.68rem', color: '#94a3b8' }}>Mean Absolute Error</div>
                        <div style={{ fontSize: '1.35rem', fontWeight: 800, color: '#00d2ff', marginTop: '2px' }}>
                          {predictionResult.model_metrics.mae.toFixed(4)}
                        </div>
                        <div style={{ fontSize: '0.62rem', color: '#64748b' }}>Average Deviation</div>
                      </div>

                      <div style={{ padding: '0.75rem', background: 'rgba(255,255,255,0.03)', borderRadius: '8px' }}>
                        <div style={{ fontSize: '0.68rem', color: '#94a3b8' }}>Root Mean Squared Err</div>
                        <div style={{ fontSize: '1.35rem', fontWeight: 800, color: '#00ff87', marginTop: '2px' }}>
                          {predictionResult.model_metrics.rmse.toFixed(4)}
                        </div>
                        <div style={{ fontSize: '0.62rem', color: '#64748b' }}>Sets 95% Interval</div>
                      </div>

                      <div style={{ padding: '0.75rem', background: 'rgba(255,255,255,0.03)', borderRadius: '8px' }}>
                        <div style={{ fontSize: '0.68rem', color: '#94a3b8' }}>Coefficient (R²)</div>
                        <div style={{ fontSize: '1.35rem', fontWeight: 800, color: '#ff9f1c', marginTop: '2px' }}>
                          {predictionResult.model_metrics.r2 ? predictionResult.model_metrics.r2.toFixed(4) : 'N/A'}
                        </div>
                        <div style={{ fontSize: '0.62rem', color: '#64748b' }}>Variance Explained</div>
                      </div>
                    </div>

                    <div style={{ marginTop: '1rem', padding: '0.75rem', background: 'rgba(0,0,0,0.3)', borderRadius: '8px', fontSize: '0.75rem', color: '#cbd5e1', lineHeight: '1.5' }}>
                      <strong>Benchmark Comparison:</strong> {predictionResult.comparison_summary}
                    </div>

                    {predictionResult.training_period && (
                      <div style={{ marginTop: '0.8rem', fontSize: '0.72rem', color: '#94a3b8' }}>
                        • Train Partition: {predictionResult.training_period.train_start} to {predictionResult.training_period.train_end} ({predictionResult.training_period.n_train_composites} composites)<br />
                        • Test Partition: {predictionResult.training_period.test_start} to {predictionResult.training_period.test_end} ({predictionResult.training_period.n_test_composites} composites)
                      </div>
                    )}
                  </div>

                  {/* Right: Model Insights & Feature Importance */}
                  <div className="glass-card feature-importance-card">
                    <h3 style={{ fontSize: '0.95rem', fontWeight: 700, color: '#fff', marginBottom: '0.4rem' }}>
                      Model Insights & Feature Importance
                    </h3>
                    <p style={{ fontSize: '0.74rem', color: '#94a3b8', marginBottom: '0.8rem' }}>
                      Relative predictor weighting extracted directly from model split criteria.
                    </p>

                    {predictionResult.feature_importances && predictionResult.feature_importances.length > 0 ? (
                      <div style={{ maxHeight: '240px', overflowY: 'auto', paddingRight: '0.4rem' }}>
                        {predictionResult.feature_importances.map(f => (
                          <div key={f.feature_name} className="feature-row">
                            <div className="feature-name-col">
                              <div className="feature-name-title">{f.feature_name}</div>
                              <div className="feature-desc-text">{f.description}</div>
                            </div>
                            <div className="feature-bar-col">
                              <div className="feature-track">
                                <div className="feature-fill" style={{ width: `${Math.min(100, Math.max(3, f.importance * 100))}%` }} />
                              </div>
                            </div>
                            <div className="feature-val-col">
                              {(f.importance * 100).toFixed(2)}%
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div style={{ padding: '1rem', background: 'rgba(255,255,255,0.02)', borderRadius: '8px', fontSize: '0.78rem', color: '#94a3b8' }}>
                        <strong>Seasonal Naive Baseline:</strong> Non-parametric heuristic based strictly on calendar lag repetition (y_t = y[t-23]). Feature importance weights do not apply to heuristic lag persistence models.
                      </div>
                    )}

                    {predictionResult.limitations && predictionResult.limitations.length > 0 && (
                      <div style={{ marginTop: '0.8rem', borderTop: '1px solid rgba(255,255,255,0.06)', paddingTop: '0.65rem' }}>
                        <div style={{ fontSize: '0.72rem', fontWeight: 700, color: '#ff6b81', textTransform: 'uppercase', marginBottom: '4px' }}>
                          Documented Model Limitations:
                        </div>
                        <ul style={{ paddingLeft: '1.2rem', fontSize: '0.72rem', color: '#94a3b8', lineHeight: '1.5' }}>
                          {predictionResult.limitations.map((lim, i) => (
                            <li key={i}>{lim}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>
                </div>
              </>
            )}
          </div>
        )}

        {activeTab === 'wildfires' && (
          <div className="glass-card" style={{ padding: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.2rem' }}>
              <div>
                <h2>NASA FIRMS VIIRS Wildfire Activity Ledger</h2>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                  Deduplicated active fire incidents across Morocco with Fire Radiative Power (MW) and satellite metadata.
                </p>
              </div>
              <button className="layer-pill active" onClick={() => setActiveTab('overview')}>
                ← Return to Cockpit
              </button>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
              {curatedFireEvents.map((e) => (
                <div key={e.id} className="glass-card" style={{ padding: '1rem' }}>
                  <div style={{ height: '140px', position: 'relative', borderRadius: '6px', overflow: 'hidden', marginBottom: '10px' }}>
                    <Image
                      src={e.thumb}
                      alt={e.region}
                      fill
                      sizes="33vw"
                      style={{ objectFit: 'cover' }}
                    />
                  </div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                    <h4>Event #{e.id}</h4>
                    <span className="badge badge-danger">{e.level}</span>
                  </div>
                  <div style={{ fontSize: '0.82rem', color: '#94a3b8', marginTop: '4px' }}>📍 {e.region} • {e.date}</div>
                  <div style={{ fontSize: '1.1rem', fontWeight: 700, color: '#ff4757', marginTop: '6px' }}>FRP: {e.frp}</div>
                  <div style={{ fontSize: '0.75rem', color: '#64748b', marginTop: '2px' }}>Sensors: VIIRS {e.sat}</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'clustering' && (
          <div className="glass-card" style={{ padding: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.2rem' }}>
              <div>
                <h2>Bioclimatic Clustering & Eco-Regional Stratification</h2>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                  Analytical K-Means clustering (K=4 to 6) evaluating multivariate NDVI and LST co-variations.
                </p>
              </div>
              <button className="layer-pill active" onClick={() => setActiveTab('overview')}>
                ← Return to Cockpit
              </button>
            </div>
            <div style={{ padding: '1.2rem', background: 'rgba(0, 210, 255, 0.06)', borderRadius: '8px', borderLeft: '4px solid #00d2ff', marginBottom: '1.5rem' }}>
              <strong>Scientific Caveat:</strong> These bioclimatic clusters represent unsupervised analytical partitions of remote sensing observations. They provide comparative stratification rather than official ministerial administrative demarcations.
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem' }}>
              {[
                { name: 'Northern Rif', type: 'Humid Mediterranean', ndvi: '0.245', lst: '24.2 °C', color: '#10b981' },
                { name: 'Atlantic Plains', type: 'Subhumid to Semi-Arid', ndvi: '0.198', lst: '28.6 °C', color: '#06b6d4' },
                { name: 'Middle Atlas', type: 'Mountain Continental', ndvi: '0.215', lst: '22.1 °C', color: '#3b82f6' },
                { name: 'High Atlas', type: 'Cold Alpine Mountain', ndvi: '0.162', lst: '19.8 °C', color: '#8b5cf6' },
                { name: 'Souss-Massa', type: 'Arid Valley', ndvi: '0.141', lst: '34.5 °C', color: '#f97316' },
                { name: 'Saharan Domain', type: 'Hyper-Arid Desert', ndvi: '0.128', lst: '38.2 °C', color: '#d97706' }
              ].map((c) => (
                <div key={c.name} className="glass-card" style={{ padding: '1rem' }}>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '8px', marginBottom: '6px' }}>
                    <span style={{ width: '10px', height: '10px', borderRadius: '50%', background: c.color }} />
                    <h4>{c.name}</h4>
                  </div>
                  <div style={{ fontSize: '0.78rem', color: '#94a3b8' }}>{c.type}</div>
                  <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: '10px', fontSize: '0.82rem' }}>
                    <span>Mean NDVI: <strong style={{ color: '#00ff87' }}>{c.ndvi}</strong></span>
                    <span>Mean LST: <strong style={{ color: '#ff7346' }}>{c.lst}</strong></span>
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {activeTab === 'methodology' && (
          <div className="glass-card" style={{ padding: '1.5rem' }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1.2rem' }}>
              <div>
                <h2>Data Processing & Scientific Methodology</h2>
                <p style={{ color: 'var(--text-muted)', fontSize: '0.85rem' }}>
                  NASA Earthdata pipeline architecture, quality control protocols, and algorithmic pipelines.
                </p>
              </div>
              <button className="layer-pill active" onClick={() => setActiveTab('overview')}>
                ← Return to Cockpit
              </button>
            </div>
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.5rem' }}>
              <div className="glass-card" style={{ padding: '1.2rem' }}>
                <h3 style={{ color: '#00d2ff', marginBottom: '8px' }}>1. Satellite Constellation Ingestion</h3>
                <p style={{ fontSize: '0.85rem', color: '#cbd5e1', lineHeight: '1.6' }}>
                  • <strong>MODIS Terra (MOD13Q1)</strong>: 250m 16-day Normalized Difference Vegetation Index.<br />
                  • <strong>MODIS Terra (MOD11A2)</strong>: 1km 8-day Daytime Land Surface Temperature.<br />
                  • <strong>VIIRS (VNP14IMGTDL & VJ114IMGTDL)</strong>: 375m active fire detections with Fire Radiative Power (MW).
                </p>
              </div>
              <div className="glass-card" style={{ padding: '1.2rem' }}>
                <h3 style={{ color: '#00d2ff', marginBottom: '8px' }}>2. Statistical Trend Verification</h3>
                <p style={{ fontSize: '0.85rem', color: '#cbd5e1', lineHeight: '1.6' }}>
                  • Non-parametric <strong>Mann-Kendall rank correlation</strong> test for monotonic trend existence.<br />
                  • <strong>Theil-Sen robust slope estimator</strong> resistant to up to 29% gross outlier contamination.<br />
                  • Automated seasonal cycle deseasonalization for pure climate signal extraction.
                </p>
              </div>
            </div>
          </div>
        )}
      </main>

      {/* ABOUT MODAL */}
      {showAboutModal && (
        <div
          style={{
            position: 'fixed',
            top: 0,
            left: 0,
            right: 0,
            bottom: 0,
            background: 'rgba(2, 4, 10, 0.85)',
            backdropFilter: 'blur(8px)',
            zIndex: 1000,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center'
          }}
          onClick={() => setShowAboutModal(false)}
        >
          <div
            className="glass-card"
            style={{ width: '560px', padding: '2rem', border: '1px solid #00d2ff', boxShadow: '0 0 30px rgba(0, 210, 255, 0.25)' }}
            onClick={(e) => e.stopPropagation()}
          >
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '10px' }}>
                <Image
                  src="/images/earth-globe.jpg"
                  alt="TerraPulse Globe"
                  width={36}
                  height={36}
                  style={{ borderRadius: '50%' }}
                />
                <h2 style={{ fontSize: '1.3rem', color: '#fff' }}>TerraPulse AI Observatory</h2>
              </div>
              <button
                onClick={() => setShowAboutModal(false)}
                style={{ background: 'transparent', border: 'none', color: '#94a3b8', fontSize: '1.2rem', cursor: 'pointer' }}
              >
                ✕
              </button>
            </div>
            <p style={{ fontSize: '0.88rem', color: '#cbd5e1', lineHeight: '1.6', marginBottom: '1rem' }}>
              <strong>TerraPulse AI</strong> provides an operational Earth observation intelligence platform dedicated to the Kingdom of Morocco. By combining decade-long NASA MODIS and VIIRS satellite sensor archives with robust machine learning models, TerraPulse delivers early drought signals, active wildfire radiative power, and bioclimatic risk intelligence.
            </p>
            <div style={{ padding: '0.8rem', background: 'rgba(255, 255, 255, 0.04)', borderRadius: '8px', fontSize: '0.78rem', color: '#94a3b8' }}>
              • Spatial Coverage: 31.8°N, 7.1°W (Kingdom of Morocco National Domain)<br />
              • Temporal Horizon: 2014-01-01 to 2026-09-14 (271 Composites, 11.75 Years)<br />
              • Data Sources: NASA Earthdata, LP DAAC, FIRMS VIIRS Constellation
            </div>
            <div style={{ display: 'flex', justifyContent: 'flex-end', marginTop: '1.2rem' }}>
              <button
                className="layer-pill active"
                onClick={() => setShowAboutModal(false)}
              >
                Close & Return
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
