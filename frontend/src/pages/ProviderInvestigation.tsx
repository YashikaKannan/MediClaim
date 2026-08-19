import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import { 
  ArrowLeft, 
  MapPin, 
  FileSpreadsheet, 
  Award,
  Sparkles,
  ClipboardList,
  Save,
  MessageSquareCode,
  AlertTriangle,
  HeartPulse
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  Cell,
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  LineChart,
  Line,
  Legend
} from 'recharts';
import { getApiUrl } from '../config';
import './ProviderInvestigation.css';

interface ProviderDetails {
  profile: {
    provider_id: string;
    total_claims: number;
    inpatient_claims: number;
    outpatient_claims: number;
    inpatient_ratio: number;
    total_beneficiaries: number;
    total_reimbursement: number;
    mean_reimbursement: number;
    risk_score: number;
    risk_level: string;
    primary_state: number;
    provider_type: string;
    PotentialFraud: number;
    investigation_status: string | null;
    investigation_notes: string | null;
    status_updated_at: string | null;
    assigned_investigator: string | null;
  };
  model_scores: {
    isolation_score: number;
    autoencoder_score: number;
    lof_score: number;
    ocsvm_score: number;
    catboost_score: number;
    ml_score: number;
    statistical_score: number;
    peer_score: number;
    leie_score?: number;
  };
  peer_benchmarks: {
    reimbursement_ratio: number;
    claims_ratio: number;
    beneficiary_ratio: number;
    reimbursement_percentile: number;
    claims_percentile: number;
    beneficiary_percentile: number;
  };
  reasons: string[];
  explanation?: {
    risk_score: number;
    risk_category: string;
    priority: string;
    why_flagged: string[];
    why_suspicious: string[];
    peer_comparison: string[];
    billing_behaviour_summary: string[];
    temporal_drift_findings: string[];
    financial_impact: string[];
    recommended_action: string;
    ai_summary: string;
  };
  drift?: {
    drift_score: number;
    drift_level: string;
    claims_spike_ratio: number;
    reimbursement_spike_ratio: number;
    coding_shift_index: number;
    historical_monthly_data: Array<{
      month: string;
      claims: number;
      reimbursement: number;
    }>;
  };
}

const ProviderInvestigation: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [data, setData] = useState<ProviderDetails | null>(null);
  const [loading, setLoading] = useState(true);
  
  // Form states
  const [status, setStatus] = useState('New');
  const [notes, setNotes] = useState('');
  const [assignedInvestigator, setAssignedInvestigator] = useState('Unassigned');
  const [savingStatus, setSavingStatus] = useState(false);
  const [savingNotes, setSavingNotes] = useState(false);
  const [savingAssign, setSavingAssign] = useState(false);
  const [statusMsg, setStatusMsg] = useState('');
  const [notesMsg, setNotesMsg] = useState('');
  const [assignMsg, setAssignMsg] = useState('');

  useEffect(() => {
    if (id) {
      fetchDetails();
    }
  }, [id]);

  const fetchDetails = async () => {
    try {
      setLoading(true);
      if (id) localStorage.setItem('mediclaim_selected_provider', id);
      const res = await fetch(`${getApiUrl()}/api/v1/providers/${id}`);
      if (res.ok) {
        const json = await res.json();
        setData(json);
        setStatus(json.profile.investigation_status || 'New');
        setNotes(json.profile.investigation_notes || '');
        setAssignedInvestigator(json.profile.assigned_investigator || 'Unassigned');
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const handleStatusUpdate = async () => {
    try {
      setSavingStatus(true);
      setStatusMsg('');
      const res = await fetch(`${getApiUrl()}/api/v1/investigations/${id}/status`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status })
      });
      if (res.ok) {
        setStatusMsg('Status updated successfully!');
        setTimeout(() => setStatusMsg(''), 3000);
      }
    } catch (e) {
      console.error(e);
      setStatusMsg('Failed to update status.');
    } finally {
      setSavingStatus(false);
    }
  };

  const handleNotesUpdate = async () => {
    try {
      setSavingNotes(true);
      setNotesMsg('');
      const res = await fetch(`${getApiUrl()}/api/v1/investigations/${id}/notes`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ notes })
      });
      if (res.ok) {
        setNotesMsg('Notes saved successfully!');
        setTimeout(() => setNotesMsg(''), 3000);
      }
    } catch (e) {
      console.error(e);
      setNotesMsg('Failed to save notes.');
    } finally {
      setSavingNotes(false);
    }
  };

  const handleAssignUpdate = async () => {
    try {
      setSavingAssign(true);
      setAssignMsg('');
      const res = await fetch(`${getApiUrl()}/api/v1/investigations/${id}/assign`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ assigned_investigator: assignedInvestigator })
      });
      if (res.ok) {
        setAssignMsg('Investigator assigned successfully!');
        setTimeout(() => setAssignMsg(''), 3000);
      }
    } catch (e) {
      console.error(e);
      setAssignMsg('Failed to assign investigator.');
    } finally {
      setSavingAssign(false);
    }
  };

  if (loading) {
    return (
      <div className="provider-loading">
        <div className="pulse-loader"></div>
        <p>Loading Provider Evidence Profile...</p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="provider-error">
        <AlertTriangle size={48} className="error-icon" />
        <h3>Provider Not Found</h3>
        <p>The provider ID **{id}** does not exist in the database.</p>
        <button className="btn btn-secondary" onClick={() => navigate('/queue')}>
          Back to Queue
        </button>
      </div>
    );
  }

  const { profile, model_scores, peer_benchmarks, reasons, drift, explanation } = data;

  // Radar chart data for investigation signals
  const radarData = [
    { subject: 'Historical Pattern', score: model_scores.catboost_score || 0 },
    { subject: 'Outlier Review', score: model_scores.isolation_score || 0 },
    { subject: 'Peer Deviation', score: model_scores.lof_score || 0 },
    { subject: 'Statistical Deviation', score: model_scores.statistical_score || 0 },
    { subject: 'Peer Comparison', score: model_scores.peer_score || 0 },
    { subject: 'Exclusion Match', score: model_scores.leie_score || 0 }
  ];

  // Bar chart data for peer ratios
  const peerRatioData = [
    { name: 'Reimbursement', ratio: peer_benchmarks.reimbursement_ratio, fill: '#3b82f6' },
    { name: 'Claims Volume', ratio: peer_benchmarks.claims_ratio, fill: '#8b5cf6' },
    { name: 'Beneficiaries', ratio: peer_benchmarks.beneficiary_ratio, fill: '#14b8a6' }
  ];

  return (
    <div className="provider-page animate-fade-in">
      {/* Back Header */}
      <div className="back-navigation">
        <button className="back-link-btn" onClick={() => navigate('/queue')}>
          <ArrowLeft size={16} />
          Back to Investigation Queue
        </button>
      </div>

      {/* Main Profile Header Card */}
      <div className="card profile-header-card">
        <div className="profile-identity">
          <div className="avatar-shield">
            <HeartPulse size={30} />
          </div>
          <div>
            <div className="profile-title-row">
              <h2>Provider {profile.provider_id}</h2>
              <span className={`badge badge-${profile.risk_level.toLowerCase()}`}>
                {profile.risk_score.toFixed(1)} Risk Score / {profile.risk_level}
              </span>
              {profile.PotentialFraud === 1 && (
                <span className="badge badge-critical fraud-label">
                  Ground Truth Fraud
                </span>
              )}
            </div>
            <div className="profile-meta-row">
              <span className="meta-item">
                <Award size={14} />
                {profile.provider_type}
              </span>
              <span className="meta-item">
                <MapPin size={14} />
                State {profile.primary_state}
              </span>
              <span className="meta-item">
                <FileSpreadsheet size={14} />
                {profile.total_claims} Total Claims
              </span>
            </div>
          </div>
        </div>

        <button 
          className="btn btn-primary chat-assistant-btn"
          onClick={() => navigate('/assistant', { state: { providerId: profile.provider_id } })}
        >
          <MessageSquareCode size={16} />
          Consult AI Investigation Copilot
        </button>
      </div>

      {/* Primary 2-Column Content Layout */}
      <div className="provider-grid">
        {/* Left Side: Score breakdowns, Peer Comparison */}
        <div className="provider-left-col">
          {/* Risk Model Breakdown */}
          <div className="card breakdown-card">
            <h3 className="section-title">Investigation Signal Overview</h3>
            <p className="section-desc">Risk signals reviewed across the provider’s historical and peer comparison patterns.</p>
            
            <div className="breakdown-content">
              {/* Radar chart */}
              <div className="radar-container">
                <ResponsiveContainer width="100%" height={260}>
                  <RadarChart cx="50%" cy="50%" outerRadius="80%" data={radarData}>
                    <PolarGrid stroke="rgba(255,255,255,0.05)" />
                    <PolarAngleAxis dataKey="subject" stroke="#94a3b8" fontSize={10} />
                    <PolarRadiusAxis angle={30} domain={[0, 100]} stroke="#64748b" fontSize={9} />
                    <Radar
                      name="Score"
                      dataKey="score"
                      stroke="#8b5cf6"
                      fill="#8b5cf6"
                      fillOpacity={0.3}
                    />
                  </RadarChart>
                </ResponsiveContainer>
              </div>

              {/* Progress bars */}
              <div className="signals-metrics-list">
                <div className="signal-progress-row">
                  <div className="progress-lbl-container">
                    <span>Historical Pattern Score</span>
                    <span className="bold">{(model_scores.catboost_score || 0).toFixed(1)}%</span>
                  </div>
                  <div className="progress-bg"><div className="progress-fg purple" style={{width: `${model_scores.catboost_score || 0}%`}}></div></div>
                </div>
                <div className="signal-progress-row">
                  <div className="progress-lbl-container">
                    <span>Outlier Review Score</span>
                    <span className="bold">{(model_scores.isolation_score || 0).toFixed(1)}%</span>
                  </div>
                  <div className="progress-bg"><div className="progress-fg blue" style={{width: `${model_scores.isolation_score || 0}%`}}></div></div>
                </div>
                <div className="signal-progress-row">
                  <div className="progress-lbl-container">
                    <span>Peer Deviation Score</span>
                    <span className="bold">{(model_scores.lof_score || 0).toFixed(1)}%</span>
                  </div>
                  <div className="progress-bg"><div className="progress-fg blue" style={{width: `${model_scores.lof_score || 0}%`}}></div></div>
                </div>
                <div className="signal-progress-row">
                  <div className="progress-lbl-container">
                    <span>Statistical Deviation Score</span>
                    <span className="bold">{(model_scores.statistical_score || 0).toFixed(1)}%</span>
                  </div>
                  <div className="progress-bg"><div className="progress-fg orange" style={{width: `${model_scores.statistical_score || 0}%`}}></div></div>
                </div>
                <div className="signal-progress-row">
                  <div className="progress-lbl-container">
                    <span>Peer Comparison Score</span>
                    <span className="bold">{(model_scores.peer_score || 0).toFixed(1)}%</span>
                  </div>
                  <div className="progress-bg"><div className="progress-fg teal" style={{width: `${model_scores.peer_score || 0}%`}}></div></div>
                </div>
                <div className="signal-progress-row">
                  <div className="progress-lbl-container">
                    <span>Exclusion Match Score</span>
                    <span className="bold">{(model_scores.leie_score || 0).toFixed(1)}%</span>
                  </div>
                  <div className="progress-bg"><div className="progress-fg red" style={{width: `${model_scores.leie_score || 0}%`}}></div></div>
                </div>
              </div>
            </div>
          </div>

          {/* Peer Ratios */}
          <div className="card peer-card">
            <h3 className="section-title">Peer Comparison</h3>
            <p className="section-desc">Comparison against similar providers in the same operational and service profile.</p>
            
            <div className="peer-content">
              <div className="peer-chart-container">
                <ResponsiveContainer width="100%" height={220}>
                  <BarChart data={peerRatioData} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.03)" />
                    <XAxis type="number" stroke="#94a3b8" />
                    <YAxis dataKey="name" type="category" stroke="#94a3b8" width={110} fontSize={11} />
                    <Tooltip 
                      contentStyle={{ backgroundColor: '#11182c', borderColor: 'rgba(255,255,255,0.08)', borderRadius: '8px' }}
                      itemStyle={{ color: '#fff' }}
                      formatter={(value) => [`${Number(value).toFixed(2)}x Median`, 'Ratio']}
                    />
                    <Bar dataKey="ratio" radius={[0, 4, 4, 0]}>
                      {peerRatioData.map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.fill} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>
              </div>

              {/* Ratios Metrics Detail */}
              <div className="ratios-detailed-grid">
                <div className="ratio-tile">
                  <p className="tile-label">Reimbursement Ratio</p>
                  <p className="tile-value">{peer_benchmarks.reimbursement_ratio.toFixed(2)}x</p>
                  <p className="tile-sub">{peer_benchmarks.reimbursement_percentile.toFixed(1)}th Percentile</p>
                </div>
                <div className="ratio-tile">
                  <p className="tile-label">Claims Ratio</p>
                  <p className="tile-value">{peer_benchmarks.claims_ratio.toFixed(2)}x</p>
                  <p className="tile-sub">{peer_benchmarks.claims_percentile.toFixed(1)}th Percentile</p>
                </div>
                <div className="ratio-tile">
                  <p className="tile-label">Patient Service Ratio</p>
                  <p className="tile-value">{peer_benchmarks.beneficiary_ratio.toFixed(2)}x</p>
                  <p className="tile-sub">{peer_benchmarks.beneficiary_percentile.toFixed(1)}th Percentile</p>
                </div>
              </div>
            </div>
            
            {/* Temporal Drift Card */}
            {drift && drift.historical_monthly_data && drift.historical_monthly_data.length > 0 && (
              <div className="card drift-card" style={{ marginTop: '1.5rem' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.5rem' }}>
                  <h3 className="section-title">Temporal Behavioral Drift</h3>
                  <span className={`badge badge-${drift.drift_level.toLowerCase()}`}>
                    Drift Level: {drift.drift_level} ({drift.drift_score.toFixed(1)})
                  </span>
                </div>
                <p className="section-desc">Month-over-month trend analysis of claim frequency and reimbursement value showing behavior deviation.</p>
                
                <div className="drift-chart-container" style={{ marginTop: '1rem', background: 'rgba(255, 255, 255, 0.02)', padding: '1rem 0.5rem 0.5rem 0.5rem', borderRadius: '8px', border: '1px solid var(--border-color)' }}>
                  <ResponsiveContainer width="100%" height={220}>
                    <LineChart data={drift.historical_monthly_data} margin={{ top: 5, right: 20, left: 10, bottom: 5 }}>
                      <CartesianGrid strokeDasharray="3 3" stroke="rgba(0,0,0,0.05)" />
                      <XAxis dataKey="month" stroke="#94a3b8" fontSize={10} />
                      <YAxis yAxisId="left" orientation="left" stroke="#3b82f6" fontSize={10} />
                      <YAxis yAxisId="right" orientation="right" stroke="#10b981" fontSize={10} />
                      <Tooltip 
                        contentStyle={{ backgroundColor: '#11182c', borderColor: 'rgba(255,255,255,0.08)', borderRadius: '8px' }}
                        itemStyle={{ color: '#fff' }}
                      />
                      <Legend wrapperStyle={{ fontSize: '11px', marginTop: '10px' }} />
                      <Line yAxisId="left" type="monotone" dataKey="claims" stroke="#3b82f6" name="Claims count" strokeWidth={2.5} activeDot={{ r: 8 }} />
                      <Line yAxisId="right" type="monotone" dataKey="reimbursement" stroke="#10b981" name="Reimbursements ($)" strokeWidth={2.5} />
                    </LineChart>
                  </ResponsiveContainer>
                </div>

                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '1rem', marginTop: '1rem' }}>
                  <div className="ratio-tile" style={{ textAlign: 'center' }}>
                    <p className="tile-label" style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Claims Spike Ratio</p>
                    <p className="tile-value" style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--accent-blue)', margin: '0.2rem 0' }}>{drift.claims_spike_ratio.toFixed(2)}x</p>
                    <p className="tile-sub" style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>vs. prior months median</p>
                  </div>
                  <div className="ratio-tile" style={{ textAlign: 'center' }}>
                    <p className="tile-label" style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Reimbursement Spike</p>
                    <p className="tile-value" style={{ fontSize: '1.2rem', fontWeight: 700, color: '#10b981', margin: '0.2rem 0' }}>{drift.reimbursement_spike_ratio.toFixed(2)}x</p>
                    <p className="tile-sub" style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>vs. prior months median</p>
                  </div>
                  <div className="ratio-tile" style={{ textAlign: 'center' }}>
                    <p className="tile-label" style={{ fontSize: '0.75rem', color: 'var(--text-secondary)' }}>Coding Shift Index</p>
                    <p className="tile-value" style={{ fontSize: '1.2rem', fontWeight: 700, color: 'var(--accent-purple)', margin: '0.2rem 0' }}>{drift.coding_shift_index.toFixed(2)}</p>
                    <p className="tile-sub" style={{ fontSize: '0.7rem', color: 'var(--text-muted)' }}>Procedure billing shifts</p>
                  </div>
                </div>
              </div>
            )}
          </div>
        </div>

        {/* Right Side: Explanations list, Status updater & Notes */}
        <div className="provider-right-col">
          {/* Automated Explanations & Reasons */}
          <div className="card reasons-card">
            <h3 className="section-title">
              <Sparkles className="spark-icon" />
              Evidence Summary
            </h3>
            <p className="section-desc">Key indicators and supporting review findings behind the current case risk.</p>
            
            <div className="reasons-list">
              {reasons.length === 0 ? (
                <p className="no-reasons">No major anomalies detected. General billing conforms to peer limits.</p>
              ) : (
                reasons.map((reason, index) => (
                  <div className="reason-item" key={index}>
                    <div className="reason-bullet-container">
                      <span className="reason-dot"></span>
                    </div>
                    <p className="reason-text">{reason}</p>
                  </div>
                ))
              )}
            </div>
          </div>

          {explanation && (
            <div className="card reasons-card business-explanation-card">
              <h3 className="section-title">Investigation Summary</h3>
              <div className="profile-meta-row" style={{ marginBottom: '0.75rem' }}>
                <span className={`badge badge-${explanation.risk_category.toLowerCase()}`}>{explanation.risk_category} Risk</span>
                <span className="badge status-review">Priority {explanation.priority}</span>
              </div>
              <div className="profile-meta-row" style={{ display: 'grid', gap: '0.25rem', alignItems: 'start', marginBottom: '0.75rem' }}>
                <span><strong>Risk Score:</strong> {explanation.risk_score}</span>
                <span><strong>Risk Category:</strong> {explanation.risk_category}</span>
                <span><strong>Investigation Priority:</strong> {explanation.priority}</span>
              </div>
              <p className="section-desc">{explanation.ai_summary}</p>
              <h4>Why Flagged</h4>
              <ul>{explanation.why_flagged.map((item) => <li key={item}>{item}</li>)}</ul>
              <h4>Why Suspicious</h4>
              <ul>{explanation.why_suspicious.map((item) => <li key={item}>{item}</li>)}</ul>
              <h4>Peer Comparison</h4>
              <ul>{explanation.peer_comparison.map((item) => <li key={item}>{item}</li>)}</ul>
              <h4>Financial Impact</h4>
              <ul>{explanation.financial_impact.map((item) => <li key={item}>{item}</li>)}</ul>
              <h4>Recommended Action</h4>
              <p className="reason-text">{explanation.recommended_action}</p>
            </div>
          )}

          {/* Clinical Audit Status & Notes */}
          <div className="card audit-card">
            <h3 className="section-title">
              <ClipboardList className="clip-icon" />
              Audit Decision Manager
            </h3>
            <p className="section-desc">Manage investigation records and document compliance files.</p>
            
            <div className="audit-controls">
              {/* Status Updater */}
              <div className="audit-field">
                <label className="field-label">Investigation Status</label>
                <div className="status-flex">
                  <select 
                    value={status} 
                    onChange={(e) => setStatus(e.target.value)}
                    className="status-selector"
                  >
                    <option value="New">New / Unassigned</option>
                    <option value="Under Review">Under Review</option>
                    <option value="Reviewed">Reviewed / Cleared</option>
                    <option value="Flagged for Audit">Flagged for Audit</option>
                  </select>
                  <button 
                    className="btn btn-secondary save-status-btn"
                    onClick={handleStatusUpdate}
                    disabled={savingStatus}
                  >
                    <Save size={14} />
                    {savingStatus ? 'Saving...' : 'Save'}
                  </button>
                </div>
                {statusMsg && <p className="success-msg">{statusMsg}</p>}
              </div>

              {/* Assigned Investigator */}
              <div className="audit-field">
                <label className="field-label">Assigned Investigator</label>
                <div className="status-flex">
                  <input 
                    type="text" 
                    value={assignedInvestigator} 
                    onChange={(e) => setAssignedInvestigator(e.target.value)}
                    className="investigator-input"
                    placeholder="Enter investigator name..."
                    style={{
                      flex: 1,
                      padding: '0.5rem 0.75rem',
                      background: 'rgba(255, 255, 255, 0.03)',
                      border: '1px solid var(--border-color)',
                      borderRadius: '8px',
                      color: 'var(--text-primary)',
                      fontSize: '0.85rem'
                    }}
                  />
                  <button 
                    className="btn btn-secondary save-status-btn"
                    onClick={handleAssignUpdate}
                    disabled={savingAssign}
                  >
                    <Save size={14} />
                    {savingAssign ? 'Saving...' : 'Assign'}
                  </button>
                </div>
                {assignMsg && <p className="success-msg">{assignMsg}</p>}
              </div>

              {/* Investigator Notes */}
              <div className="audit-field">
                <label className="field-label">Audit Notes & Documentation</label>
                <textarea 
                  rows={6}
                  placeholder="Record provider explanations, contact logs, medical record requests, or clinical findings here..."
                  value={notes}
                  onChange={(e) => setNotes(e.target.value)}
                ></textarea>
                <div className="textarea-footer">
                  {notesMsg && <span className="success-msg">{notesMsg}</span>}
                  <button 
                    className="btn btn-primary save-notes-btn"
                    onClick={handleNotesUpdate}
                    disabled={savingNotes}
                  >
                    <Save size={14} />
                    {savingNotes ? 'Saving Notes...' : 'Save Notes'}
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default ProviderInvestigation;
