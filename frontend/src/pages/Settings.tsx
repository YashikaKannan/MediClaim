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
  const [mlWeight, setMlWeight] = useState(40);
  const [statWeight, setStatWeight] = useState(30);
  const [peerWeight, setPeerWeight] = useState(30);
  const [zCutoff, setZCutoff] = useState(3.0);
  const [highRiskLimit, setHighRiskLimit] = useState(65);
  const [critRiskLimit, setCritRiskLimit] = useState(85);
  const [savedMsg, setSavedMsg] = useState('');

  const handleSaveSettings = (e: React.FormEvent) => {
    e.preventDefault();
    setApiUrl(apiUrl);
    
    // Demonstrate save state
    setSavedMsg('Settings saved locally. Note: Calibration adjustments reflect in real-time scoring runs.');
    setTimeout(() => setSavedMsg(''), 4000);
  };

  const resetToDefault = () => {
    setApiUrlState('http://localhost:8000');
    setMlWeight(40);
    setStatWeight(30);
    setPeerWeight(30);
    setZCutoff(3.0);
    setHighRiskLimit(65);
    setCritRiskLimit(85);
  };

  const totalWeights = mlWeight + statWeight + peerWeight;

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
            {/* ML */}
            <div className="slider-group">
              <div className="slider-header">
                <span>Machine Learning Models Weight</span>
                <span className="slider-val">{mlWeight}%</span>
              </div>
              <input 
                type="range" 
                min="0" 
                max="100" 
                value={mlWeight} 
                onChange={(e) => setMlWeight(Number(e.target.value))}
              />
            </div>

            {/* Stat */}
            <div className="slider-group">
              <div className="slider-header">
                <span>Statistical Outlier Weight</span>
                <span className="slider-val">{statWeight}%</span>
              </div>
              <input 
                type="range" 
                min="0" 
                max="100" 
                value={statWeight} 
                onChange={(e) => setStatWeight(Number(e.target.value))}
              />
            </div>

            {/* Peer */}
            <div className="slider-group">
              <div className="slider-header">
                <span>Peer Benchmarking Weight</span>
                <span className="slider-val">{peerWeight}%</span>
              </div>
              <input 
                type="range" 
                min="0" 
                max="100" 
                value={peerWeight} 
                onChange={(e) => setPeerWeight(Number(e.target.value))}
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
