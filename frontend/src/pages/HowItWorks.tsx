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
          <p className="page-subtitle">Inside the MediClaim fraud investigation workflow and audit intelligence platform</p>
        </div>
      </div>

      {/* Visual Pipeline flow chart */}
      <div className="card pipeline-flow-card">
        <h3 className="section-title">End-to-End Investigation Workflow</h3>
        <p className="section-desc">From intake to case review, each workflow stage builds evidence for human investigators.</p>
        
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
            <h4>3. Risk Review</h4>
            <p>Assesses billing patterns, provider comparisons, and unusual claim behavior to prioritize investigation.</p>
          </div>
          <ArrowRight className="flow-arrow" />
          <div className="flow-step">
            <div className="step-icon-box green">
              <Sparkles size={24} />
            </div>
            <h4>4. Case Prioritization</h4>
            <p>Builds a risk score, evidence summary, and recommended next action for investigators.</p>
          </div>
        </div>
      </div>

      {/* Three layers breakdown */}
      <div className="layers-grid">
        {/* Layer 1: ML Model Layer */}
        <div className="card layer-card">
          <div className="layer-header">
            <span className="layer-num">40% Weight</span>
            <h3 className="layer-title">Pattern Detection Layer</h3>
          </div>
          <p className="layer-desc">Combines historical risk signals, billing irregularities, and claim anomalies to identify priority cases.</p>
          
          <ul className="layer-bullets">
            <li>
              <strong>Historical Risk Patterning:</strong> Prioritizes providers whose claims behavior resembles previously reviewed suspect patterns.
            </li>
            <li>
              <strong>Claims Structure Review:</strong> Identifies unusually shaped claim patterns or elevated service concentration.
            </li>
            <li>
              <strong>Provider Outlier Screening:</strong> Isolates providers whose billing behavior differs materially from peers and historical norms.
            </li>
          </ul>
        </div>

        {/* Layer 2: Statistical Layer */}
        <div className="card layer-card">
          <div className="layer-header">
            <span className="layer-num">30% Weight</span>
            <h3 className="layer-title">Deviation Review Layer</h3>
          </div>
          <p className="layer-desc">Highlights providers whose billing averages or claim volumes deviate significantly from expected ranges.</p>
          
          <ul className="layer-bullets">
            <li>
              <strong>Statistical Variance Review:</strong> Uses robust comparisons to spot unusual provider behavior without overreacting to a single extreme value.
            </li>
            <li>
              <strong>Threshold Alerts:</strong> Identifies substantial claim or reimbursement deviations that warrant investigative review.
            </li>
          </ul>
        </div>

        {/* Layer 3: Peer Benchmarking Layer */}
        <div className="card layer-card">
          <div className="layer-header">
            <span className="layer-num">30% Weight</span>
            <h3 className="layer-title">Peer Comparison Layer</h3>
          </div>
          <p className="layer-desc">Compares providers within similar service and geography cohorts to identify materially unusual activity.</p>
          
          <ul className="layer-bullets">
            <li>
              <strong>Reimbursement Ratios:</strong> Flags providers whose billing substantially exceeds the norm for their peer group.
            </li>
            <li>
              <strong>State and Specialty Normalization:</strong> Adjusts review for geography and service mix so investigators compare like with like.
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
        <p className="section-desc">The platform turns risk signals into plain-language evidence so investigators can quickly understand why a case matters:</p>
        <div className="reason-rules-grid">
          <div className="rule-box">
            <span className="rule-condition">Payment Volume &gt; Peers</span>
            <p className="rule-explanation">Highlights disproportionate billing versus the provider’s peer group.</p>
          </div>
          <div className="rule-box">
            <span className="rule-condition">Historical Pattern Match</span>
            <p className="rule-explanation">Flags providers whose billing profile resembles prior reviewed fraud or abuse patterns.</p>
          </div>
          <div className="rule-box">
            <span className="rule-condition">Behavioral Shift</span>
            <p className="rule-explanation">Identifies sudden changes in claim volume, reimbursement, or billing mix that need review.</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default HowItWorks;
