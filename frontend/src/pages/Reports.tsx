import React, { useEffect, useState } from 'react';
import { Download, FileText, RefreshCw, Save, ShieldAlert, Table2 } from 'lucide-react';
import { Bar, BarChart, CartesianGrid, Cell, Pie, PieChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts';
import { getApiUrl } from '../config';
import './Reports.css';

type ReportTab = 'provider' | 'claims' | 'monthly' | 'distribution';
interface ReportData { generated_at: string; provider: any; model_scores: any; benchmark_rows: any[]; financial_exposure: any; claims: any[]; reasons: string[]; narrative: string; recommendations: string[]; }
const selectedKey = 'mediclaim_selected_provider';
const money = (value: number) => `$${Number(value || 0).toLocaleString(undefined, { maximumFractionDigits: 0 })}`;

const Reports: React.FC = () => {
  const [providerId, setProviderId] = useState(localStorage.getItem(selectedKey) || '');
  const [report, setReport] = useState<ReportData | null>(null);
  const [dashboard, setDashboard] = useState<any>(null);
  const [tab, setTab] = useState<ReportTab>('provider');
  const [notes, setNotes] = useState('');
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');

  const load = async () => {
    const id = localStorage.getItem(selectedKey) || providerId;
    setProviderId(id);
    if (!id) return;
    setBusy(true);
    try {
      const [reportResponse, dashboardResponse] = await Promise.all([
        fetch(`${getApiUrl()}/api/v1/reports/provider/${id}`),
        fetch(`${getApiUrl()}/api/v1/dashboard`)
      ]);
      if (reportResponse.ok) {
        const data = await reportResponse.json(); setReport(data); setNotes(data.provider.investigation_notes || '');
      }
      if (dashboardResponse.ok) setDashboard(await dashboardResponse.json());
    } finally { setBusy(false); }
  };

  useEffect(() => { load(); const handler = () => load(); window.addEventListener('mediclaim-provider-selected', handler); return () => window.removeEventListener('mediclaim-provider-selected', handler); }, []);

  const exportFile = async (kind: 'claims.csv' | 'export.xlsx') => {
    if (!providerId) return;
    setBusy(true); setMessage('Preparing export...');
    try {
      const response = await fetch(`${getApiUrl()}/api/v1/reports/provider/${providerId}/${kind}`);
      if (!response.ok) throw new Error((await response.json()).detail || 'Export failed');
      const blob = await response.blob(); const url = URL.createObjectURL(blob); const anchor = document.createElement('a');
      anchor.href = url; anchor.download = `${providerId}-${kind}`; anchor.click(); URL.revokeObjectURL(url); setMessage('Export ready');
    } catch (error) { setMessage(error instanceof Error ? error.message : 'Export failed'); } finally { setBusy(false); }
  };

  const saveNotes = async () => {
    if (!providerId) return; setBusy(true);
    await fetch(`${getApiUrl()}/api/v1/investigations/${providerId}/notes`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ notes }) });
    setBusy(false); setMessage('Investigator notes saved');
  };

  if (!providerId) return <div className="reports-empty"><ShieldAlert size={42} /><h2>Select a provider to generate reports</h2><p>Open a case from the Investigation Queue to load its audit package.</p></div>;
  if (busy && !report) return <div className="reports-loading"><RefreshCw className="spinning" /><p>Building evidence package from provider records...</p></div>;
  if (!report) return <div className="reports-empty"><ShieldAlert size={42} /><h2>Report unavailable</h2><button className="btn btn-secondary" onClick={load}>Retry</button></div>;

  const { provider, model_scores: scores, financial_exposure: exposure } = report;
  const modelData = [{ name: 'Isolation Forest', value: scores.isolation_score }, { name: 'LOF', value: scores.lof_score }, { name: 'Robust Z', value: scores.statistical_score }, { name: 'Peer', value: scores.peer_score }, { name: 'LEIE', value: scores.leie_score }];
  const riskData = dashboard ? Object.entries(dashboard.risk_distribution).map(([name, value]) => ({ name, value })) : [];
  const stateData = dashboard?.risk_by_state?.map((item: any) => ({ name: `State ${item.state}`, risk: Number(item.avg_risk).toFixed(1) })) || [];

  return <div className="reports-page animate-fade-in">
    <div className="page-header reports-header"><div><p className="eyebrow">Payment Integrity / Audit Package</p><h1 className="page-title">Investigation Reports</h1><p className="page-subtitle">Evidence generated for Provider {provider.provider_id} from the current case record.</p></div><div className="report-actions"><span className={`badge badge-${String(provider.risk_level).toLowerCase()}`}>{provider.risk_level} Risk</span><button className="btn btn-secondary" onClick={() => window.print()}><FileText size={16} /> PDF / Print</button><button className="btn btn-secondary" onClick={() => exportFile('export.xlsx')}><Download size={16} /> Excel</button><button className="btn btn-secondary" onClick={() => exportFile('claims.csv')}><Download size={16} /> Claims CSV</button></div></div>
    <div className="report-meta"><span>Generated {new Date(report.generated_at).toLocaleString()}</span><span>Case status: {provider.investigation_status || 'New'}</span><span>Assigned: {provider.assigned_investigator || 'Unassigned'}</span>{message && <strong>{message}</strong>}</div>
    <nav className="report-tabs">{[['provider', 'Provider Investigation'], ['claims', 'Claim Investigation'], ['monthly', 'Monthly Leakage'], ['distribution', 'Risk Distribution']].map(([value, label]) => <button className={tab === value ? 'active' : ''} onClick={() => setTab(value as ReportTab)} key={value}><Table2 size={15} />{label}</button>)}</nav>
    {tab === 'provider' && <>
      <section className="executive-summary card"><div><p className="eyebrow">Executive Summary</p><h2>Provider {provider.provider_id}</h2><p>{report.narrative}</p></div><div className="risk-score"><span>Composite risk</span><strong>{Number(provider.risk_score).toFixed(1)}</strong><small>/ 100</small></div></section>
      <div className="report-grid"><section className="card"><h3>Provider Overview</h3><dl className="facts"><dt>Provider type</dt><dd>{provider.provider_type}</dd><dt>Operating state</dt><dd>State {provider.primary_state}</dd><dt>Total claims</dt><dd>{Number(provider.total_claims).toLocaleString()}</dd><dt>Reimbursement</dt><dd>{money(provider.total_reimbursement)}</dd></dl></section><section className="card"><h3>Financial Exposure</h3><div className="metric-grid"><div><span>Total claims</span><strong>{Number(provider.total_claims).toLocaleString()}</strong></div><div><span>Average claim</span><strong>{money(exposure.average_claim_value)}</strong></div><div><span>Potential leakage</span><strong className="danger-text">{money(exposure.potential_leakage)}</strong></div></div></section></div>
      <section className="card"><h3>Model Findings</h3><div className="chart-wrap"><ResponsiveContainer width="100%" height={250}><BarChart data={modelData}><CartesianGrid strokeDasharray="3 3" /><XAxis dataKey="name" /><YAxis domain={[0, 100]} /><Tooltip /><Bar dataKey="value" fill="#147d92" /></BarChart></ResponsiveContainer></div></section>
      <section className="card"><h3>Peer Benchmark Analysis</h3><div className="table-container"><table><thead><tr><th>Metric</th><th>Provider value</th><th>Peer median</th><th>Difference</th><th>Percentile</th></tr></thead><tbody>{report.benchmark_rows.map(row => <tr key={row.metric}><td>{row.metric}</td><td>{row.metric.includes('Reimbursement') ? money(row.provider_value) : row.provider_value.toLocaleString()}</td><td>{row.metric.includes('Reimbursement') ? money(row.peer_median) : row.peer_median.toLocaleString()}</td><td>{row.difference_percent.toFixed(1)}%</td><td>{row.national_percentile.toFixed(1)}th</td></tr>)}</tbody></table></div></section>
      <div className="report-grid"><section className="card"><h3>Explainability</h3><p className="narrative">{report.narrative}</p><ul>{report.reasons.map(reason => <li key={reason}>{reason}</li>)}</ul></section><section className="card"><h3>Recommended Actions</h3><ul className="action-list">{report.recommendations.map(action => <li key={action}>{action}</li>)}</ul><h3 className="notes-heading">Investigator Notes</h3><textarea value={notes} onChange={event => setNotes(event.target.value)} placeholder="Record review findings..." /><button className="btn btn-primary" onClick={saveNotes}><Save size={15} /> Save Notes</button></section></div>
    </>}
    {tab === 'claims' && <section className="card"><div className="section-heading"><div><h2>Flagged Claims</h2><p>Claims ranked by recorded anomaly and fraud scores.</p></div><button className="btn btn-primary" onClick={() => exportFile('claims.csv')}><Download size={15} /> Export suspicious claims</button></div><div className="table-container"><table><thead><tr><th>Claim ID</th><th>Beneficiary</th><th>Amount</th><th>Fraud probability</th><th>Anomaly score</th><th>Explanation</th></tr></thead><tbody>{report.claims.map(claim => <tr key={claim.claim_id}><td>{claim.claim_id}</td><td>{claim.bene_id}</td><td>{money(claim.claim_amount)}</td><td>{Number(claim.fraud_flag || 0) ? 'Flagged' : `${Number(claim.risk_score || 0).toFixed(1)}%`}</td><td>{Number(claim.risk_score || 0).toFixed(1)}</td><td>{claim.explanation || claim.suspicious_codes || 'Recorded risk signal'}</td></tr>)}</tbody></table></div></section>}
    {tab === 'monthly' && <><section className="report-grid"><div className="card"><h3>Executive Leakage Summary</h3><div className="metric-grid"><div><span>Providers reviewed</span><strong>{dashboard?.total_providers?.toLocaleString()}</strong></div><div><span>Flagged providers</span><strong>{dashboard?.flagged_providers?.toLocaleString()}</strong></div><div><span>Potential leakage</span><strong>{money(dashboard?.potential_leakage)}</strong></div><div><span>Average risk</span><strong>{Number(dashboard?.average_risk || 0).toFixed(1)}</strong></div></div></div><div className="card"><h3>Top States by Risk</h3><ResponsiveContainer width="100%" height={220}><BarChart data={stateData}><XAxis dataKey="name" /><YAxis /><Tooltip /><Bar dataKey="risk" fill="#e07a45" /></BarChart></ResponsiveContainer></div></section></>}
    {tab === 'distribution' && <section className="report-grid"><div className="card"><h3>Risk Distribution</h3><ResponsiveContainer width="100%" height={280}><PieChart><Pie data={riskData} dataKey="value" nameKey="name" outerRadius={95} label>{riskData.map((entry: any, index: number) => <Cell key={entry.name} fill={['#2c9c78', '#e0a13b', '#d86645', '#8d3f52'][index]} />)}</Pie><Tooltip /></PieChart></ResponsiveContainer></div><div className="card"><h3>Highest Risk Providers</h3><div className="rank-list">{dashboard?.top_suspicious?.slice(0, 20).map((item: any, index: number) => <div key={item.provider_id}><span>{index + 1}. {item.provider_id}</span><strong>{Number(item.risk_score).toFixed(1)} / {item.risk_level}</strong></div>)}</div></div></section>}
  </div>;
};
export default Reports;