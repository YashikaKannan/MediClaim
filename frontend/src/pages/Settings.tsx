import React, { useState } from 'react';
import { 
  Sliders, 
  Save, 
  Database,
  SlidersHorizontal,
  Info
} from 'lucide-react';
import { getApiUrl, setApiUrl } from '../config';
import './Settings.css';

const Settings: React.FC = () => {
  const [apiUrl, setApiUrlState] = useState(getApiUrl());
  const [catboostWeight, setCatboostWeight] = useState(25);
  const [iforestWeight, setIforestWeight] = useState(20);
  const [lofWeight, setLofWeight] = useState(15);
  const [robustZWeight, setRobustZWeight] = useState(15);
  const [peerBenchmarkWeight, setPeerBenchmarkWeight] = useState(15);
  const [leieWeight, setLeieWeight] = useState(10);
  const [zCutoff, setZCutoff] = useState(3.0);
  const [highRiskLimit, setHighRiskLimit] = useState(65);
  const [critRiskLimit, setCritRiskLimit] = useState(85);
  const [savedMsg, setSavedMsg] = useState('');
  const [loading, setLoading] = useState(true);

  React.useEffect(() => {
    fetchSettings();
  }, []);

  const fetchSettings = async () => {
    try {
      setLoading(true);
      const res = await fetch(`${getApiUrl()}/api/v1/settings`);
      if (res.ok) {
        const data = await res.json();
        setApiUrlState(data.api_url || getApiUrl());
        setCatboostWeight(Math.round((data.catboost_weight || 0.25) * 100));
        setIforestWeight(Math.round((data.iforest_weight || 0.20) * 100));
        setLofWeight(Math.round((data.lof_weight || 0.15) * 100));
        setRobustZWeight(Math.round((data.robust_z_weight || 0.15) * 100));
        setPeerBenchmarkWeight(Math.round((data.peer_benchmark_weight || 0.15) * 100));
        setLeieWeight(Math.round((data.leie_weight || 0.10) * 100));
        setZCutoff(data.z_cutoff);
        setHighRiskLimit(data.high_risk_limit);
        setCritRiskLimit(data.crit_risk_limit);
      }
    } catch (e) {
      console.error("Error loading settings from DB", e);
    } finally {
      setLoading(false);
    }
  };

  const handleSaveSettings = async (e: React.FormEvent) => {
    e.preventDefault();
    setApiUrl(apiUrl);
    
    try {
      const res = await fetch(`${getApiUrl()}/api/v1/settings`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          api_url: apiUrl,
          catboost_weight: catboostWeight / 100,
          iforest_weight: iforestWeight / 100,
          lof_weight: lofWeight / 100,
          robust_z_weight: robustZWeight / 100,
          peer_benchmark_weight: peerBenchmarkWeight / 100,
          leie_weight: leieWeight / 100,
          z_cutoff: zCutoff,
          high_risk_limit: highRiskLimit,
          crit_risk_limit: critRiskLimit
        })
      });
      if (res.ok) {
        setSavedMsg('Configuration successfully updated and persisted in database.');
      } else {
        setSavedMsg('Failed to persist configuration to database.');
      }
    } catch (err) {
      setSavedMsg('Network error while saving configurations.');
    }
    setTimeout(() => setSavedMsg(''), 4000);
  };

  const resetToDefault = () => {
    setApiUrlState('http://localhost:8000');
    setCatboostWeight(25);
    setIforestWeight(20);
    setLofWeight(15);
    setRobustZWeight(15);
    setPeerBenchmarkWeight(15);
    setLeieWeight(10);
    setZCutoff(3.0);
    setHighRiskLimit(65);
    setCritRiskLimit(85);
  };

  const totalWeights = catboostWeight + iforestWeight + lofWeight + robustZWeight + peerBenchmarkWeight + leieWeight;

  if (loading) {
    return (
      <div className="provider-loading">
        <div className="pulse-loader"></div>
        <p>Loading System Configurations...</p>
      </div>
    );
  }

  return (
    <div className="settings-page animate-fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">System Settings</h1>
          <p className="page-subtitle">Configure backend connections, risk weighting factors, and severity thresholds.</p>
        </div>
      </div>

      <form onSubmit={handleSaveSettings} className="settings-form">
        {/* API Server Section */}
        <div className="card settings-card">
          <h3 className="section-title">
            <Database size={18} className="settings-icon-blue" />
            Backend Connection Profile
          </h3>
          <p className="section-desc">Specify the hostname and endpoint configuration for the FastAPI service.</p>
          
          <div className="settings-field">
            <label className="field-label">FastAPI API Server URL</label>
            <input 
              type="url" 
              placeholder="e.g. http://localhost:8000"
              value={apiUrl}
              onChange={(e) => setApiUrlState(e.target.value)}
              required
            />
            <p className="field-help">Default is http://localhost:8000. Changes apply immediately upon saving.</p>
          </div>
        </div>

        {/* Weights Section */}
        <div className="card settings-card">
          <h3 className="section-title">
            <Sliders size={18} className="settings-icon-purple" />
            Risk score Calibration weights
          </h3>
          <p className="section-desc">Tune the coefficients for the multi-signal risk scoring fusion engine (must equal 100%).</p>
          
          <div className="weights-sliders-container">
            {/* CatBoost */}
            <div className="slider-group">
              <div className="slider-header">
                <span>CatBoost Classifier Weight</span>
                <span className="slider-val">{catboostWeight}%</span>
              </div>
              <input 
                type="range" 
                min="0" 
                max="100" 
                value={catboostWeight} 
                onChange={(e) => setCatboostWeight(Number(e.target.value))}
              />
            </div>

            {/* Isolation Forest */}
            <div className="slider-group">
              <div className="slider-header">
                <span>Isolation Forest Weight</span>
                <span className="slider-val">{iforestWeight}%</span>
              </div>
              <input 
                type="range" 
                min="0" 
                max="100" 
                value={iforestWeight} 
                onChange={(e) => setIforestWeight(Number(e.target.value))}
              />
            </div>

            {/* LOF */}
            <div className="slider-group">
              <div className="slider-header">
                <span>Local Outlier Factor (LOF) Weight</span>
                <span className="slider-val">{lofWeight}%</span>
              </div>
              <input 
                type="range" 
                min="0" 
                max="100" 
                value={lofWeight} 
                onChange={(e) => setLofWeight(Number(e.target.value))}
              />
            </div>

            {/* Robust Z-Score */}
            <div className="slider-group">
              <div className="slider-header">
                <span>Robust Z-Score Weight</span>
                <span className="slider-val">{robustZWeight}%</span>
              </div>
              <input 
                type="range" 
                min="0" 
                max="100" 
                value={robustZWeight} 
                onChange={(e) => setRobustZWeight(Number(e.target.value))}
              />
            </div>

            {/* Peer Benchmarking */}
            <div className="slider-group">
              <div className="slider-header">
                <span>Peer Benchmarking Weight</span>
                <span className="slider-val">{peerBenchmarkWeight}%</span>
              </div>
              <input 
                type="range" 
                min="0" 
                max="100" 
                value={peerBenchmarkWeight} 
                onChange={(e) => setPeerBenchmarkWeight(Number(e.target.value))}
              />
            </div>

            {/* LEIE Exclusion */}
            <div className="slider-group">
              <div className="slider-header">
                <span>LEIE Exclusion Screening Weight</span>
                <span className="slider-val">{leieWeight}%</span>
              </div>
              <input 
                type="range" 
                min="0" 
                max="100" 
                value={leieWeight} 
                onChange={(e) => setLeieWeight(Number(e.target.value))}
              />
            </div>

            <div className={`weights-total-indicator ${totalWeights === 100 ? 'valid' : 'invalid'}`}>
              <Info size={16} />
              <span>Total Combined Weights: <strong>{totalWeights}%</strong> (Must sum to exactly 100%)</span>
            </div>
          </div>
        </div>

        {/* Decision Limits */}
        <div className="card settings-card">
          <h3 className="section-title">
            <SlidersHorizontal size={18} className="settings-icon-teal" />
            Outlier Classification Thresholds
          </h3>
          <p className="section-desc">Manage standard deviations and classification boundaries used to trigger warnings.</p>
          
          <div className="thresholds-grid">
            <div className="settings-field">
              <label className="field-label">Robust Z-Score Cutoff Limit</label>
              <input 
                type="number" 
                step="0.1" 
                min="1.0" 
                max="5.0"
                value={zCutoff}
                onChange={(e) => setZCutoff(Number(e.target.value))}
              />
              <p className="field-help">Standard deviations above peer medians using robust median absolute deviations (MAD).</p>
            </div>

            <div className="settings-field">
              <label className="field-label">High-Risk Limit</label>
              <input 
                type="number" 
                min="30" 
                max="100"
                value={highRiskLimit}
                onChange={(e) => setHighRiskLimit(Number(e.target.value))}
              />
              <p className="field-help">Composite score (0-100) above which a provider is labeled as High Risk.</p>
            </div>

            <div className="settings-field">
              <label className="field-label">Critical-Risk Limit</label>
              <input 
                type="number" 
                min="50" 
                max="100"
                value={critRiskLimit}
                onChange={(e) => setCritRiskLimit(Number(e.target.value))}
              />
              <p className="field-help">Composite score (0-100) above which a provider is labeled as Critical Risk.</p>
            </div>
          </div>
        </div>

        {/* Action Panel */}
        <div className="settings-actions-bar">
          {savedMsg && <span className="save-indicator-msg">{savedMsg}</span>}
          <button 
            type="button" 
            className="btn btn-secondary"
            onClick={resetToDefault}
          >
            Restore Defaults
          </button>
          <button 
            type="submit" 
            className="btn btn-primary"
            disabled={totalWeights !== 100}
          >
            <Save size={16} />
            Save Configurations
          </button>
        </div>
      </form>
    </div>
  );
};

export default Settings;
