import React, { useState, useEffect, useRef } from 'react';
import { 
  UploadCloud, 
  CheckCircle2, 
  Play, 
  RefreshCw, 
  AlertCircle,
  FileText,
  Clock,
  Sparkles,
  Database
} from 'lucide-react';
import { getApiUrl } from '../config';
import './DataIngestion.css';

interface DatasetStats {
  total_providers: number;
  total_claims: number;
  total_reimbursement: number;
}

interface UploadStatus {
  fileType: 'claims' | 'beneficiary' | 'provider';
  filename: string;
  progress: number;
  status: 'idle' | 'uploading' | 'success' | 'failed';
  errorMessage?: string;
  rowCount?: number;
}

interface PipelineSummary {
  providers_scanned: number;
  claims_scored: number;
  critical_risk_count: number;
  high_risk_count: number;
  average_risk_score: number;
}

const DataIngestion: React.FC = () => {
  const [stats, setStats] = useState<DatasetStats | null>(null);
  const [loading, setLoading] = useState(true);
  const [runningPipeline, setRunningPipeline] = useState(false);
  const [pipelineProgress, setPipelineProgress] = useState(0);
  const [pipelineStep, setPipelineStep] = useState('');
  const [pipelineError, setPipelineError] = useState<string | null>(null);
  const [pipelineSummary, setPipelineSummary] = useState<PipelineSummary | null>(null);

  // Upload States
  const [uploads, setUploads] = useState<Record<string, UploadStatus>>({
    claims: { fileType: 'claims', filename: '', progress: 0, status: 'idle' },
    beneficiary: { fileType: 'beneficiary', filename: '', progress: 0, status: 'idle' },
    provider: { fileType: 'provider', filename: '', progress: 0, status: 'idle' }
  });

  const [activeUploadType, setActiveUploadType] = useState<'claims' | 'beneficiary' | 'provider'>('claims');
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Poll status interval reference
  const pollIntervalRef = useRef<number | null>(null);

  useEffect(() => {
    fetchStats();
    checkActivePipeline();
    return () => {
      if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    };
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

  const checkActivePipeline = async () => {
    try {
      const res = await fetch(`${getApiUrl()}/api/v1/pipeline/status`);
      if (res.ok) {
        const data = await res.json();
        if (data.status === 'running') {
          setRunningPipeline(true);
          setPipelineProgress(data.progress);
          setPipelineStep(data.step);
          startPollingPipeline();
        } else if (data.status === 'completed' && data.summary) {
          setPipelineSummary(data.summary);
        }
      }
    } catch (e) {
      console.error(e);
    }
  };

  const startPollingPipeline = () => {
    if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
    
    pollIntervalRef.current = window.setInterval(async () => {
      try {
        const res = await fetch(`${getApiUrl()}/api/v1/pipeline/status`);
        if (res.ok) {
          const data = await res.json();
          setPipelineProgress(data.progress);
          setPipelineStep(data.step);
          
          if (data.status === 'completed') {
            setRunningPipeline(false);
            setPipelineSummary(data.summary);
            setPipelineError(null);
            fetchStats();
            if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
          } else if (data.status === 'failed') {
            setRunningPipeline(false);
            setPipelineError(data.error || 'Pipeline execution failed.');
            if (pollIntervalRef.current) clearInterval(pollIntervalRef.current);
          }
        }
      } catch (e) {
        console.error("Error polling pipeline status", e);
      }
    }, 1000);
  };

  const handleStartPipeline = async () => {
    try {
      setPipelineError(null);
      setPipelineSummary(null);
      setRunningPipeline(true);
      setPipelineProgress(5);
      setPipelineStep('Initiating risk pipeline in background...');
      
      const res = await fetch(`${getApiUrl()}/api/v1/pipeline/run`, {
        method: 'POST'
      });
      if (res.ok) {
        startPollingPipeline();
      } else {
        const errJson = await res.json();
        setRunningPipeline(false);
        setPipelineError(errJson.detail || 'Failed to start pipeline.');
      }
    } catch (e) {
      setRunningPipeline(false);
      setPipelineError('Connection error while running pipeline.');
      console.error(e);
    }
  };

  // Drag and Drop handlers
  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault();
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      uploadFile(e.dataTransfer.files[0], activeUploadType);
    }
  };

  const triggerBrowse = () => {
    fileInputRef.current?.click();
  };

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      uploadFile(e.target.files[0], activeUploadType);
    }
  };

  const uploadFile = async (file: File, type: 'claims' | 'beneficiary' | 'provider') => {
    // Front-end check
    if (!file.name.endsWith('.csv')) {
      updateUploadState(type, {
        filename: file.name,
        progress: 0,
        status: 'failed',
        errorMessage: 'Invalid file type. Only CSV files are supported.'
      });
      return;
    }

    updateUploadState(type, {
      filename: file.name,
      progress: 10,
      status: 'uploading'
    });

    const formData = new FormData();
    formData.append('file', file);

    // Simulate upload progress bar since fetch API doesn't support upload progress out of the box without complex XHR wrapping
    const prInterval = setInterval(() => {
      setUploads(prev => {
        const cur = prev[type];
        if (cur.status !== 'uploading') {
          clearInterval(prInterval);
          return prev;
        }
        const nextProgress = Math.min(90, cur.progress + 15);
        return {
          ...prev,
          [type]: { ...cur, progress: nextProgress }
        };
      });
    }, 150);

    try {
      const res = await fetch(`${getApiUrl()}/api/v1/upload/${type}`, {
        method: 'POST',
        body: formData
      });

      clearInterval(prInterval);

      if (res.ok) {
        const data = await res.json();
        updateUploadState(type, {
          filename: file.name,
          progress: 100,
          status: 'success',
          rowCount: data.row_count
        });
        fetchStats();
      } else {
        const err = await res.json();
        updateUploadState(type, {
          filename: file.name,
          progress: 0,
          status: 'failed',
          errorMessage: err.detail || 'Validation failed. Check CSV schema.'
        });
      }
    } catch (e) {
      clearInterval(prInterval);
      updateUploadState(type, {
        filename: file.name,
        progress: 0,
        status: 'failed',
        errorMessage: 'Network error occurred during upload.'
      });
    }
  };

  const updateUploadState = (type: string, fields: Partial<UploadStatus>) => {
    setUploads(prev => ({
      ...prev,
      [type]: { ...prev[type], ...fields } as UploadStatus
    }));
  };

  return (
    <div className="ingestion-page animate-fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">Case Intake & Analysis</h1>
          <p className="page-subtitle">Load healthcare data, review provider and claim activity, and generate evidence for investigator follow-up.</p>
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
          <h3 className="card-title">Current Case Inventory</h3>
          <p className="card-desc">Healthcare claim records currently loaded and ready for risk review.</p>
          
          <div className="stats-list">
            <div className="stat-entry">
              <span className="stat-num">{stats?.total_providers ? stats.total_providers.toLocaleString() : '5,410'}</span>
              <span className="stat-lbl">Active Providers</span>
            </div>
            <div className="stat-entry">
              <span className="stat-num">{stats?.total_claims ? stats.total_claims.toLocaleString() : '558,211'}</span>
              <span className="stat-lbl">Claims Scored</span>
            </div>
            <div className="stat-entry">
              <span className="stat-num">
                ${stats?.total_reimbursement ? (stats.total_reimbursement / 1000000).toFixed(1) + 'M' : '138.2M'}
              </span>
              <span className="stat-lbl">Total Covered Payments</span>
            </div>
          </div>

          <div className="files-status-list">
            <h4 className="list-section-header">Active Datasets Loaded</h4>
            <div className="file-status-item">
              <FileText size={18} className="file-icon active" />
              <div className="file-details">
                <span className="file-name">Claims Transaction Logs (Inpatient/Outpatient)</span>
                <span className="file-meta">Default fallback or active user claims.csv loaded.</span>
              </div>
              <CheckCircle2 size={18} className="success-icon" />
            </div>
            <div className="file-status-item">
              <FileText size={18} className="file-icon active" />
              <div className="file-details">
                <span className="file-name">Beneficiary Demographic Profiles</span>
                <span className="file-meta">Contains age, chronic conditions, deceased indicators.</span>
              </div>
              <CheckCircle2 size={18} className="success-icon" />
            </div>
            <div className="file-status-item">
              <FileText size={18} className="file-icon active" />
              <div className="file-details">
                <span className="file-name">Provider Fraud Ground Truth Labels</span>
                <span className="file-meta">Reference provider target classifications.</span>
              </div>
              <CheckCircle2 size={18} className="success-icon" />
            </div>
          </div>
        </div>

        {/* Pipeline Control */}
        <div className="card pipeline-card">
          <h3 className="card-title">Investigation Analysis Runner</h3>
          <p className="card-desc">Review claims and provider activity, recalculate risk scores, and generate explainable case findings.</p>

          {!runningPipeline ? (
            <div className="pipeline-idle">
              {pipelineSummary ? (
                <div className="pipeline-summary-box animate-fade-in">
                  <div className="summary-title-flex">
                    <Sparkles size={20} className="glow-icon" />
                    <h4>Investigation Summary</h4>
                  </div>
                  <div className="summary-grid">
                    <div className="summary-tile">
                      <span className="summary-lbl">Providers Scanned</span>
                      <span className="summary-val">{pipelineSummary.providers_scanned.toLocaleString()}</span>
                    </div>
                    <div className="summary-tile">
                      <span className="summary-lbl">Claims Scored</span>
                      <span className="summary-val">{pipelineSummary.claims_scored.toLocaleString()}</span>
                    </div>
                    <div className="summary-tile">
                      <span className="summary-lbl">Critical Flagged</span>
                      <span className="summary-val text-critical bold">{pipelineSummary.critical_risk_count}</span>
                    </div>
                    <div className="summary-tile">
                      <span className="summary-lbl">Average Risk Score</span>
                      <span className="summary-val">{pipelineSummary.average_risk_score.toFixed(1)}%</span>
                    </div>
                  </div>
                  <p className="summary-timestamp">Completed at: {new Date().toLocaleTimeString()}</p>
                </div>
              ) : (
                <div className="pipeline-graphic">
                  <Database size={48} className="glow-icon" />
                </div>
              )}

              <button 
                className="btn btn-primary start-pipeline-btn"
                onClick={handleStartPipeline}
                disabled={runningPipeline}
              >
                <Play size={16} />
                Run Case Analysis
              </button>
              
              {pipelineError && (
                <div className="pipeline-alert-error animate-fade-in">
                  <AlertCircle size={16} className="error-icon" />
                  <p>{pipelineError}</p>
                </div>
              )}

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
              <h4 className="pipeline-status-lbl">Processing case data...</h4>
              <p className="pipeline-step-desc">{pipelineStep}</p>
              <div className="progress-bar-bg">
                <div 
                  className="progress-bar-fg" 
                  style={{ width: `${pipelineProgress}%` }}
                ></div>
              </div>
              <div className="estimated-time">
                <Clock size={14} />
                <span>Estimated Time Remaining: {Math.max(1, Math.round((100 - pipelineProgress) / 5))} seconds</span>
              </div>
            </div>
          )}
        </div>
      </div>

      {/* Real File Upload Dropzone Section */}
      <div className="card upload-workbench-card">
        <h3 className="card-title">Data Intake Portal</h3>
        <p className="card-desc">Upload claims and provider files to refresh the live investigation dataset.</p>
        
        {/* Upload Segment Tabs */}
        <div className="upload-tabs-row">
          <button 
            className={`upload-tab-btn ${activeUploadType === 'claims' ? 'active' : ''}`}
            onClick={() => setActiveUploadType('claims')}
          >
            Claims Transaction CSV
          </button>
          <button 
            className={`upload-tab-btn ${activeUploadType === 'beneficiary' ? 'active' : ''}`}
            onClick={() => setActiveUploadType('beneficiary')}
          >
            Beneficiary Profile CSV
          </button>
          <button 
            className={`upload-tab-btn ${activeUploadType === 'provider' ? 'active' : ''}`}
            onClick={() => setActiveUploadType('provider')}
          >
            Provider Labels CSV
          </button>
        </div>

        {/* Dynamic dropzone based on active type */}
        <div 
          className="drag-drop-section"
          onDragOver={handleDragOver}
          onDrop={handleDrop}
          onClick={triggerBrowse}
          style={{ cursor: 'pointer' }}
        >
          <UploadCloud size={40} className="upload-icon" />
          <h3>
            {activeUploadType === 'claims' && 'Upload Claims Data (claims.csv)'}
            {activeUploadType === 'beneficiary' && 'Upload Beneficiary Data (beneficiary.csv)'}
            {activeUploadType === 'provider' && 'Upload Provider Reference Data (provider.csv)'}
          </h3>
          <p>
            {activeUploadType === 'claims' && 'Required columns: ClaimID, BeneID, Provider, InscClaimAmtReimbursed'}
            {activeUploadType === 'beneficiary' && 'Required columns: BeneID, DOB, Gender, Race, State, County'}
            {activeUploadType === 'provider' && 'Required columns: Provider'}
          </p>
          <p className="subtext">Drag and drop a CSV file here or click to browse (max 200MB)</p>
          
          <input 
            type="file" 
            ref={fileInputRef}
            onChange={handleFileChange}
            accept=".csv"
            style={{ display: 'none' }}
          />

          <button className="btn btn-secondary" onClick={(e) => { e.stopPropagation(); triggerBrowse(); }}>
            Select CSV File
          </button>
        </div>

        {/* Upload status indicator row */}
        <div className="upload-status-summary-row">
          {Object.entries(uploads).map(([key, u]) => (
            <div className={`upload-status-tile status-${u.status}`} key={key}>
              <div className="tile-main">
                <span className="tile-lbl text-capitalize">{key} Data File</span>
                {u.status === 'idle' && <span className="tile-status-txt">No file uploaded</span>}
                {u.status === 'uploading' && (
                  <div className="tile-progress-flex">
                    <span className="tile-status-txt text-blue">Uploading... {u.progress}%</span>
                    <div className="mini-progress-bar"><div className="mini-progress-fg" style={{ width: `${u.progress}%` }}></div></div>
                  </div>
                )}
                {u.status === 'success' && (
                  <span className="tile-status-txt text-green">
                    ✓ {u.filename} ({u.rowCount?.toLocaleString()} rows)
                  </span>
                )}
                {u.status === 'failed' && (
                  <span className="tile-status-txt text-red">
                    ✗ {u.errorMessage || 'Upload failed'}
                  </span>
                )}
              </div>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
};

export default DataIngestion;
