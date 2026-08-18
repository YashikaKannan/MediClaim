import React from 'react';
import { 
  Network, 
  Cpu, 
  Database,
  ArrowRight,
  Sparkles,
  FileText
} from 'lucide-react';
import './HowItWorks.css';

const HowItWorks: React.FC = () => {
  return (
    <div className="how-page animate-fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">How It Works</h1>
          <p className="page-subtitle">Inside the MediClaim Multi-Layered Fraud Detection Pipeline & Risk Score Fusion</p>
        </div>
      </div>

      {/* Visual Pipeline flow chart */}
      <div className="card pipeline-flow-card">
        <h3 className="section-title">End-to-End Analytical Processing</h3>
        <p className="section-desc">From raw claim databases to compiled risk indicators and evidence generation.</p>
        
        <div className="flow-steps-container">
          <div className="flow-step">
            <div className="step-icon-box">
              <Database size={24} />
            </div>
            <h4>1. Claim Data Aggregation</h4>
            <p>Aggregates inpatient, outpatient, and beneficiary databases by Provider ID.</p>
          </div>
          <ArrowRight className="flow-arrow" />
          <div className="flow-step">
            <div className="step-icon-box purple">
              <Cpu size={24} />
            </div>
            <h4>2. Feature Engineering</h4>
            <p>Calculates reimbursement densities, physician networks, age variations, and ratios.</p>
          </div>
          <ArrowRight className="flow-arrow" />
          <div className="flow-step">
            <div className="step-icon-box blue">
              <Network size={24} />
            </div>
            <h4>3. Multi-Layer Scoring</h4>
            <p>Executes CatBoost, PyTorch Autoencoder, IF, LOF, and Z-score outlier engines.</p>
          </div>
          <ArrowRight className="flow-arrow" />
          <div className="flow-step">
            <div className="step-icon-box green">
              <Sparkles size={24} />
            </div>
            <h4>4. Risk Score Fusion</h4>
            <p>Applies a weighted formula to compile a 0-100 score and automated justifications.</p>
          </div>
        </div>
      </div>

      {/* Three layers breakdown */}
      <div className="layers-grid">
        {/* Layer 1: ML Model Layer */}
        <div className="card layer-card">
          <div className="layer-header">
            <span className="layer-num">40% Weight</span>
            <h3 className="layer-title">ML Detection Layer</h3>
          </div>
          <p className="layer-desc">Combines supervised classification and unsupervised clustering models to uncover patterns.</p>
          
          <ul className="layer-bullets">
            <li>
              <strong>CatBoost Classifier:</strong> Supervised model trained on past potential fraud targets. Capture non-linear billing networks.
            </li>
            <li>
              <strong>PyTorch Autoencoder:</strong> Reconstructs claims features to flag providers with high reconstruction errors (structural anomalies).
            </li>
            <li>
              <strong>Isolation Forest & One-Class SVM:</strong> Isolates provider outliers in high-dimensional space.
            </li>
          </ul>
        </div>

        {/* Layer 2: Statistical Layer */}
        <div className="card layer-card">
          <div className="layer-header">
            <span className="layer-num">30% Weight</span>
            <h3 className="layer-title">Statistical Outlier Layer</h3>
          </div>
          <p className="layer-desc">Uses robust statistical engines to isolate providers whose billing averages deviate significantly from normal benchmarks.</p>
          
          <ul className="layer-bullets">
            <li>
              <strong>Robust Z-Score Engine:</strong> Employs Median and Median Absolute Deviation (MAD) rather than mean/std, making it highly resilient to extreme values.
            </li>
            <li>
              <strong>Threshold Cutoffs:</strong> Identifies providers with Z-scores &gt; 3.0 (extreme outliers) in mean payouts or volume.
            </li>
          </ul>
        </div>

        {/* Layer 3: Peer Benchmarking Layer */}
        <div className="card layer-card">
          <div className="layer-header">
            <span className="layer-num">30% Weight</span>
            <h3 className="layer-title">Peer Benchmarking Layer</h3>
          </div>
          <p className="layer-desc">Aggregates medians within specific peer groups (e.g. Outpatient-heavy providers in State 39) to measure relative variance.</p>
          
          <ul className="layer-bullets">
            <li>
              <strong>Reimbursement Ratios:</strong> Provider total billing divided by the peer median. Identifies providers billing 10x or 20x the median of their direct neighbors.
            </li>
            <li>
              <strong>State/Specialty Normalization:</strong> Controls for geographical and service variances (e.g. inpatient complexity vs outpatient routines).
            </li>
          </ul>
        </div>
      </div>

      {/* Explanations Logic */}
      <div className="card reasons-logic-card">
        <h3 className="section-title">
          <FileText size={18} className="text-icon" />
          Automated Reasoning Engine
        </h3>
        <p className="section-desc">We convert complex mathematical outcomes into human-readable evidence using dynamic rules based on calculated ratios:</p>
        <div className="reason-rules-grid">
          <div className="rule-box">
            <span className="rule-condition">Ratio &gt; 5x Peer</span>
            <p className="rule-explanation">Generates flag for disproportionate billing: "Total reimbursement is X times the median of peers in State Y."</p>
          </div>
          <div className="rule-box">
            <span className="rule-condition">CatBoost &gt; 80%</span>
            <p className="rule-explanation">Generates supervised flag: "Supervised models indicate high correlation with historical fraud patterns."</p>
          </div>
          <div className="rule-box">
            <span className="rule-condition">Z-Score &gt; 3.0</span>
            <p className="rule-explanation">Generates volume outlier flag: "Billed claim count is statistical outlier with robust Z-score of Z."</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HowItWorks;
