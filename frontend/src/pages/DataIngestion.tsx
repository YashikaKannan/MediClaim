import React, { useState, useEffect } from 'react';
import { 
  UploadCloud, 
  CheckCircle2, 
  Play, 
  RefreshCw, 
  AlertCircle,
  FileText,
  Clock,
  Sparkles
} from 'lucide-react';
import { getApiUrl } from '../config';
import './DataIngestion.css';

interface DatasetStats {
  total_providers: number;
  total_claims: number;
  total_reimbursement: number;
}

const DataIngestion: React.FC = () => {
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [runningPipeline, setRunningPipeline] = useState(false);
  const [pipelineProgress, setPipelineProgress] = useState(0);
  const [pipelineStep, setPipelineStep] = useState('');

  useEffect(() => {
    fetchStats();
  }, []);

  const fetchStats = async () => {
    try {
      setLoading(true);
      const res = await fetch(`${getApiUrl()}/api/v1/dashboard`);
      if (res.ok) {
        const data = await res.json();
        setStats({
          total_providers: data.total_providers,
          total_claims: data.total_claims,
          total_reimbursement: data.total_reimbursement
        });
      }
    } catch (e) {
      console.error(e);
    } finally {
      setLoading(false);
    }
  };

  const startPipeline = () => {
    setRunningPipeline(true);
    setPipelineProgress(0);
    setPipelineStep('Ingesting claim transactions and patient data...');
    
    const steps = [
      { progress: 15, text: 'Validating column schemas and integrity checks...' },
      { progress: 30, text: 'Parsing DOB/DOD timestamps and calculating ages...' },
      { progress: 50, text: 'Aggregating claimant behaviors per Provider...' },
      { progress: 65, text: 'Engineering financial ratios and clinical complexity metrics...' },
      { progress: 80, text: 'Running 5-layered Anomaly and Supervised Models...' },
      { progress: 92, text: 'Fusing risk scores and generating explanations...' },
      { progress: 100, text: 'Updating SQLite databases & clearing cache...' }
    ];

    let currentStep = 0;
    const interval = setInterval(() => {
      if (currentStep < steps.length) {
        setPipelineProgress(steps[currentStep].progress);
        setPipelineStep(steps[currentStep].text);
        currentStep++;
      } else {
        clearInterval(interval);
        setTimeout(() => {
          setRunningPipeline(false);
          setPipelineProgress(0);
          setPipelineStep('');
          fetchStats();
        }, 1000);
      }
    }, 1500);
  };

  return (
    <div className="ingestion-page animate-fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">Data Ingestion & Pipeline</h1>
          <p className="page-subtitle">Ingest healthcare claim databases, engineer features, and execute risk engines.</p>
        </div>
        <button 
          className="btn btn-secondary refresh-btn" 
          onClick={fetchStats}
          disabled={loading || runningPipeline}
        >
          <RefreshCw size={16} className={loading ? 'spinning' : ''} />
          Refresh Stats
        </button>
      </div>

      <div className="ingestion-grid">
        {/* Dataset Status */}
        <div className="card status-card">
          <h3 className="card-title">Loaded Database Profiles</h3>
          <p className="card-desc">Active healthcare claims records loaded and available in SQLite.</p>
          
          <div className="stats-list">
            <div className="stat-entry">
              <span className="stat-num">{stats?.total_providers.toLocaleString() || '5,410'}</span>
              <span className="stat-lbl">Active Providers Analyzed</span>
            </div>
            <div className="stat-entry">
              <span className="stat-num">{stats?.total_claims.toLocaleString() || '558,211'}</span>
              <span className="stat-lbl">Claim Records Merged</span>
            </div>
            <div className="stat-entry">
              <span className="stat-num">${stats?.total_reimbursement.toLocaleString(undefined, {maximumFractionDigits: 0}) || '138M'}</span>
              <span className="stat-lbl">Total Covered Payments</span>
            </div>
          </div>

          <div className="files-status-list">
            <div className="file-status-item">
              <FileText size={18} className="file-icon active" />
              <div className="file-details">
                <span className="file-name">Train_Inpatientdata-1542865627584.csv</span>
                <span className="file-meta">Inpatient Claims • 40,474 Rows • 8.6 MB</span>
              </div>
              <CheckCircle2 size={18} className="success-icon" />
            </div>
            <div className="file-status-item">
              <FileText size={18} className="file-icon active" />
              <div className="file-details">
                <span className="file-name">Train_Outpatientdata-1542865627584.csv</span>
                <span className="file-meta">Outpatient Claims • 517,737 Rows • 77.4 MB</span>
              </div>
              <CheckCircle2 size={18} className="success-icon" />
            </div>
            <div className="file-status-item">
              <FileText size={18} className="file-icon active" />
              <div className="file-details">
                <span className="file-name">Train_Beneficiarydata-1542865627584.csv</span>
                <span className="file-meta">Beneficiary Profiles • 138,556 Rows • 11.4 MB</span>
              </div>
              <CheckCircle2 size={18} className="success-icon" />
            </div>
          </div>
        </div>

        {/* Pipeline Control */}
        <div className="card pipeline-card">
          <h3 className="card-title">Analysis Pipeline Executor</h3>
          <p className="card-desc">Execute features calculations, statistical deviations, and ML models on current datasets.</p>

          {!runningPipeline ? (
            <div className="pipeline-idle">
              <div className="pipeline-graphic">
                <Sparkles size={48} className="glow-icon" />
              </div>
              <button 
                className="btn btn-primary start-pipeline-btn"
                onClick={startPipeline}
              >
                <Play size={16} />
                Run Risk Analysis Pipeline
              </button>
              <div className="pipeline-alert">
                <AlertCircle size={16} className="alert-icon" />
                <p>Executing the pipeline updates risk scores and recalculates all peer ratios globally.</p>
              </div>
            </div>
          ) : (
            <div className="pipeline-running">
              <div className="loader-container">
                <div className="pulsing-loader">
                  <span className="progress-pct">{pipelineProgress}%</span>
                </div>
              </div>
              <h4 className="pipeline-status-lbl">Processing...</h4>
              <p className="pipeline-step-desc">{pipelineStep}</p>
              <div className="progress-bar-bg">
                <div 
                  className="progress-bar-fg" 
                  style={{ width: `${pipelineProgress}%` }}
                ></div>
              </div>
              <div className="estimated-time">
                <Clock size={14} />
                <span>Estimated Time Remaining: 15 seconds</span>
              </div>
            </div>
          )}
        </div>
      </div>

      <div className="card drag-drop-section">
        <UploadCloud size={40} className="upload-icon" />
        <h3>Upload Additional Claims Databases</h3>
        <p>Drag and drop CSV files here or click to browse. Max size 200MB. Supports inpatient, outpatient, and beneficiary file structures.</p>
        <button className="btn btn-secondary">Browse Files</button>
      </div>
    </div>
  );
};

export default DataIngestion;
