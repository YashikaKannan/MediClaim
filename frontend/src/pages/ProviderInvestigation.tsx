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
  Radar
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
}

const ProviderInvestigation: React.FC = () => {
  const { id } = useParams<{ id: string }>();
  const navigate = useNavigate();
  const [data, setData] = useState<ProviderDetails | null>(null);
  const [loading, setLoading] = useState(true);
  
  // Form states
  const [status, setStatus] = useState('New');
  const [notes, setNotes] = useState('');
  const [savingStatus, setSavingStatus] = useState(false);
  const [savingNotes, setSavingNotes] = useState(false);
  const [statusMsg, setStatusMsg] = useState('');
  const [notesMsg, setNotesMsg] = useState('');

  useEffect(() => {
    if (id) {
      fetchDetails();
    }
  }, [id]);

  const fetchDetails = async () => {
    try {
      setLoading(true);
      const res = await fetch(`${getApiUrl()}/api/v1/providers/${id}`);
      if (res.ok) {
        const json = await res.json();
        setData(json);
        setStatus(json.profile.investigation_status || 'New');
        setNotes(json.profile.investigation_notes || '');
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

  const { profile, model_scores, peer_benchmarks, reasons } = data;

  // Radar chart data for individual models
  const radarData = [
    { subject: 'Isolation Forest', score: model_scores.isolation_score },
    { subject: 'Autoencoder', score: model_scores.autoencoder_score },
    { subject: 'LOF', score: model_scores.lof_score },
    { subject: 'One-Class SVM', score: model_scores.ocsvm_score },
    { subject: 'CatBoost (Supervised)', score: model_scores.catboost_score }
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
                {profile.risk_score.toFixed(1)} / {profile.risk_level} Risk
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
          Consult AI Assistant
        </button>
      </div>

      {/* Primary 2-Column Content Layout */}
      <div className="provider-grid">
        {/* Left Side: Score breakdowns, Peer Comparison */}
        <div className="provider-left-col">
          {/* Risk Model Breakdown */}
          <div className="card breakdown-card">
            <h3 className="section-title">Algorithmic Risk Signature</h3>
            <p className="section-desc">Multi-layered anomaly scoring profiles comparing unsupervised and supervised algorithms.</p>
            
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
                    <span>Supervised Score (CatBoost)</span>
                    <span className="bold">{model_scores.catboost_score.toFixed(1)}%</span>
                  </div>
                  <div className="progress-bg"><div className="progress-fg purple" style={{width: `${model_scores.catboost_score}%`}}></div></div>
                </div>
                <div className="signal-progress-row">
                  <div className="progress-lbl-container">
                    <span>Unsupervised Score (Ensemble)</span>
                    <span className="bold">{model_scores.ml_score.toFixed(1)}%</span>
                  </div>
                  <div className="progress-bg"><div className="progress-fg blue" style={{width: `${model_scores.ml_score}%`}}></div></div>
                </div>
                <div className="signal-progress-row">
                  <div className="progress-lbl-container">
                    <span>Statistical Deviation (Z-Score)</span>
                    <span className="bold">{model_scores.statistical_score.toFixed(1)}%</span>
                  </div>
                  <div className="progress-bg"><div className="progress-fg orange" style={{width: `${model_scores.statistical_score}%`}}></div></div>
                </div>
                <div className="signal-progress-row">
                  <div className="progress-lbl-container">
                    <span>Peer Benchmark Variance</span>
                    <span className="bold">{model_scores.peer_score.toFixed(1)}%</span>
                  </div>
                  <div className="progress-bg"><div className="progress-fg teal" style={{width: `${model_scores.peer_score}%`}}></div></div>
                </div>
              </div>
            </div>
          </div>

          {/* Peer Ratios */}
          <div className="card peer-card">
            <h3 className="section-title">Peer Group Benchmarking</h3>
            <p className="section-desc">Comparison against provider peer medians. A value of **1.0x** is equal to peer median billing.</p>
            
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
          </div>
        </div>

        {/* Right Side: Explanations list, Status updater & Notes */}
        <div className="provider-right-col">
          {/* Automated Explanations & Reasons */}
          <div className="card reasons-card">
            <h3 className="section-title">
              <Sparkles className="spark-icon" />
              Automated Fraud Evidence
            </h3>
            <p className="section-desc">Heuristics-grounded and model-based evidence justifying the flagged category.</p>
            
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
