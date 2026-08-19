import React, { useState, useEffect } from 'react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  Legend, 
  ResponsiveContainer
} from 'recharts';
import { 
  RefreshCw,
  Zap,
  Info
} from 'lucide-react';
import { getApiUrl } from '../config';
import './Analytics.css';

interface PerformanceMetric {
  model: string;
  precision: number;
  recall: number;
  f1: number;
  pr_auc: number;
  precision_at_100: number;
  type: string;
}

interface SpecialtyData {
  provider_type: string;
  avg_risk: number;
  cnt: number;
}

interface StateRisk {
  state: number;
  avg_risk: number;
  cnt: number;
}

const Analytics: React.FC = () => {
  const [metrics, setMetrics] = useState<PerformanceMetric[]>([]);
  const [specialties, setSpecialties] = useState<SpecialtyData[]>([]);
  const [stateRisks, setStateRisks] = useState<StateRisk[]>([]);
  const [loading, setLoading] = useState(true);

  const getSignalLabel = (model: string) => {
    const normalized = model.toLowerCase();

    if (normalized.includes('combined') || normalized.includes('fusion')) return 'Composite Review Signal';
    if (normalized.includes('catboost') || normalized.includes('historical')) return 'Historical Pattern Review';
    if (normalized.includes('iforest') || normalized.includes('isolation')) return 'Outlier Review';
    if (normalized.includes('lof') || normalized.includes('deviation')) return 'Peer Deviation Review';
    if (normalized.includes('robust') || normalized.includes('statistical')) return 'Statistical Deviation Review';
    if (normalized.includes('leie') || normalized.includes('exclusion')) return 'Exclusion Match Review';
    if (normalized.includes('peer')) return 'Peer Comparison Review';

    return model || 'Review Signal';
  };

  useEffect(() => {
    fetchAnalyticsData();
  }, []);

  const fetchAnalyticsData = async () => {
    try {
      setLoading(true);
      // Fetch model performance
      const metricsRes = await fetch(`${getApiUrl()}/api/v1/model-performance`);
      if (metricsRes.ok) {
        const metricsJson = await metricsRes.json();
        setMetrics(metricsJson);
      }

      // Fetch dashboard summaries for specialty and state averages
      const summaryRes = await fetch(`${getApiUrl()}/api/v1/dashboard`);
      if (summaryRes.ok) {
        const summaryJson = await summaryRes.json();
        setSpecialties(summaryJson.risk_by_type);
        setStateRisks(summaryJson.risk_by_state);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="analytics-loading">
        <div className="pulse-loader"></div>
        <p>Generating Analytical Reports...</p>
      </div>
    );
  }

  // Prep chart data
  const specChartData = specialties.map(s => ({
    name: s.provider_type,
    'Average Risk': Math.round(s.avg_risk * 10) / 10,
    'Count': s.cnt
  }));

  const stateChartData = stateRisks.map(sr => ({
    name: `State ${sr.state}`,
    'Avg Risk Score': Math.round(sr.avg_risk * 10) / 10
  }));

  return (
    <div className="analytics-page animate-fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">Auditor Insights</h1>
          <p className="page-subtitle">Provider risk patterns, geographic concentration, and investigation priorities across the active case load.</p>
        </div>
        <button className="btn btn-secondary" onClick={fetchAnalyticsData}>
          <RefreshCw size={16} />
          Reload Analytics
        </button>
      </div>

      {/* Model Performance Comparison Grid */}
      <div className="card performance-card">
        <div className="card-header-flex">
          <div>
            <h3 className="section-title">
              <Zap size={18} className="zap-icon" />
              Investigation Signal Review
            </h3>
            <p className="section-desc">Operational risk indicators reviewed across the active case population to support prioritization and human oversight.</p>
          </div>
        </div>

        <div className="table-container">
          <table>
            <thead>
              <tr>
                <th>Investigation Signal</th>
                <th>Review Type</th>
                <th>Precision</th>
                <th>Recall</th>
                <th>F1-Score</th>
                <th>PR-AUC</th>
                <th>Precision @ Top 100</th>
              </tr>
            </thead>
            <tbody>
              {metrics.map((metric, i) => (
                <tr key={i} className={metric.model.toLowerCase().includes('combined') || metric.model.toLowerCase().includes('fusion') ? 'highlight-row' : ''}>
                  <td>
                    <strong>{getSignalLabel(metric.model)}</strong>
                  </td>
                  <td>
                    <span className={`type-tag type-${metric.type.replace(/\s+/g, '').toLowerCase()}`}>
                      {metric.type || 'Review Signal'}
                    </span>
                  </td>
                  <td>{(metric.precision * 100).toFixed(1)}%</td>
                  <td>{(metric.recall * 100).toFixed(1)}%</td>
                  <td>{(metric.f1 * 100).toFixed(1)}%</td>
                  <td>{metric.pr_auc.toFixed(3)}</td>
                  <td>
                    <span className="percent-100-badge">
                      {(metric.precision_at_100 * 100).toFixed(0)}%
                    </span>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="evaluation-note">
          <Info size={14} />
          <p>
            <strong>Key Insight:</strong> The investigation engine combines claim patterns, peer comparisons, provider history, and rule-based review triggers to surface the most actionable cases for investigators. This keeps priority review focused on providers with the clearest evidence of inappropriate payment behavior.
          </p>
        </div>
      </div>

      {/* Charts Grid */}
      <div className="analytics-charts-grid">
        {/* Geographic Risk density */}
        <div className="card chart-card">
          <h3 className="chart-title">Top 10 State Average Risk Indexes</h3>
          <p className="chart-desc">Average provider risk level aggregated by operational state headquarters.</p>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={stateChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                <XAxis dataKey="name" stroke="#94a3b8" fontSize={11} />
                <YAxis stroke="#94a3b8" fontSize={11} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#11182c', borderColor: 'rgba(255,255,255,0.08)', borderRadius: '8px' }}
                  labelStyle={{ color: '#94a3b8', fontWeight: 600 }}
                  itemStyle={{ color: '#fff' }}
                />
                <Bar dataKey="Avg Risk Score" fill="#14b8a6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Specialty distribution */}
        <div className="card chart-card">
          <h3 className="chart-title">Billing Segment Comparison</h3>
          <p className="chart-desc">Average risk indices and provider volume segmented by billing type.</p>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={specChartData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                <XAxis dataKey="name" stroke="#94a3b8" fontSize={11} />
                <YAxis stroke="#94a3b8" fontSize={11} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#11182c', borderColor: 'rgba(255,255,255,0.08)', borderRadius: '8px' }}
                  labelStyle={{ color: '#94a3b8', fontWeight: 600 }}
                  itemStyle={{ color: '#fff' }}
                />
                <Legend wrapperStyle={{ paddingTop: '10px' }} />
                <Bar dataKey="Average Risk" fill="#8b5cf6" radius={[4, 4, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Analytics;
