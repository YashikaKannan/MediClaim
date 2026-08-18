import React, { useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Users, 
  AlertTriangle, 
  ShieldAlert, 
  Layers,
  Activity,
  ArrowUpRight
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer,
  PieChart,
  Pie,
  Cell,
  Legend
} from 'recharts';
import { getApiUrl } from '../config';
import './Dashboard.css';

interface DashboardData {
  total_providers: number;
  total_claims: number;
  total_reimbursement: number;
  investigation_queue_count: number;
  risk_distribution: {
    Low: number;
    Medium: number;
    High: number;
    Critical: number;
  };
  top_suspicious: Array<{
    provider_id: string;
    risk_score: number;
    risk_level: string;
    total_reimbursement: number;
    total_claims: number;
    provider_type: string;
  }>;
  risk_by_type: Array<{
    provider_type: string;
    avg_risk: number;
    cnt: number;
  }>;
  risk_by_state: Array<{
    state: number;
    avg_risk: number;
    cnt: number;
  }>;
  recent_activity: Array<{
    provider_id: string;
    status: string;
    updated_at: string;
    risk_score: number;
    risk_level: string;
  }>;
}

const Dashboard: React.FC = () => {
  const [data, setData] = useState<DashboardData | null>(null);
  const [loading, setLoading] = useState(true);
  const navigate = useNavigate();

  useEffect(() => {
    const fetchDashboardData = async () => {
      try {
        setLoading(true);
        const res = await fetch(`${getApiUrl()}/api/v1/dashboard`);
        if (res.ok) {
          const json = await res.json();
          setData(json);
        }
      } catch (err) {
        console.error('Error fetching dashboard data:', err);
      } finally {
        setLoading(false);
      }
    };
    fetchDashboardData();
  }, []);

  if (loading) {
    return (
      <div className="dashboard-loading">
        <div className="pulse-loader"></div>
        <p>Loading Executive Dashboard...</p>
      </div>
    );
  }

  if (!data) {
    return (
      <div className="dashboard-error">
        <AlertTriangle size={48} className="error-icon" />
        <h3>Failed to Load Data</h3>
        <p>Could not connect to FastAPI backend. Make sure the server is running.</p>
      </div>
    );
  }

  // Prep Chart Data
  const pieData = [
    { name: 'Low Risk', value: data.risk_distribution.Low, color: '#10b981' },
    { name: 'Medium Risk', value: data.risk_distribution.Medium, color: '#f59e0b' },
    { name: 'High Risk', value: data.risk_distribution.High, color: '#ef4444' },
    { name: 'Critical Risk', value: data.risk_distribution.Critical, color: '#d946ef' }
  ];

  const barData = data.risk_by_type.map(item => ({
    name: item.provider_type,
    'Average Risk Score': Math.round(item.avg_risk * 10) / 10,
    'Provider Count': item.cnt
  }));



  return (
    <div className="dashboard-page animate-fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">Executive Dashboard</h1>
          <p className="page-subtitle">MediClaim Claims Risk Intelligence and Fraud Prevention Overview</p>
        </div>
        <div className="time-badge">
          <Activity size={14} />
          <span>Live Session Active</span>
        </div>
      </div>

      {/* KPI Cards */}
      <div className="kpi-grid">
        <div className="card kpi-card">
          <div className="kpi-icon-container blue">
            <Users size={22} />
          </div>
          <div className="kpi-content">
            <p className="kpi-lbl">Total Providers</p>
            <h2 className="kpi-val">{data.total_providers.toLocaleString()}</h2>
            <span className="kpi-subtext">Across 52 States</span>
          </div>
        </div>

        <div className="card kpi-card">
          <div className="kpi-icon-container warning">
            <AlertTriangle size={22} />
          </div>
          <div className="kpi-content">
            <p className="kpi-lbl">High Risk Providers</p>
            <h2 className="kpi-val">{data.risk_distribution.High.toLocaleString()}</h2>
            <span className="kpi-subtext">Requires regular audit</span>
          </div>
        </div>

        <div className="card kpi-card">
          <div className="kpi-icon-container critical animate-pulse">
            <ShieldAlert size={22} />
          </div>
          <div className="kpi-content">
            <p className="kpi-lbl">Critical Providers</p>
            <h2 className="kpi-val">{data.risk_distribution.Critical.toLocaleString()}</h2>
            <span className="kpi-subtext">Immediate audit recommended</span>
          </div>
        </div>

        <div className="card kpi-card">
          <div className="kpi-icon-container purple">
            <Layers size={22} />
          </div>
          <div className="kpi-content">
            <p className="kpi-lbl">Pending Review</p>
            <h2 className="kpi-val">{data.investigation_queue_count.toLocaleString()}</h2>
            <span className="kpi-subtext">Active investigation queue</span>
          </div>
        </div>
      </div>

      {/* Charts Section */}
      <div className="dashboard-charts-grid">
        {/* Risk Distribution Pie Chart */}
        <div className="card chart-card">
          <h3 className="chart-title">Risk Severity Distribution</h3>
          <p className="chart-desc">Relative proportions of providers categorized by composite risk level.</p>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={260}>
              <PieChart>
                <Pie
                  data={pieData}
                  cx="50%"
                  cy="50%"
                  innerRadius={65}
                  outerRadius={90}
                  paddingAngle={5}
                  dataKey="value"
                >
                  {pieData.map((entry) => (
                    <Cell key={entry.name} fill={entry.color} />
                  ))}
                </Pie>
                <Tooltip 
                  contentStyle={{ backgroundColor: '#11182c', borderColor: 'rgba(255,255,255,0.08)', borderRadius: '8px' }}
                  itemStyle={{ color: '#fff' }}
                />
                <Legend iconType="circle" wrapperStyle={{ paddingTop: '10px' }} />
              </PieChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Risk by Provider Type */}
        <div className="card chart-card">
          <h3 className="chart-title">Average Risk by Facility Type</h3>
          <p className="chart-desc">Comparison of Inpatient-heavy vs Outpatient-heavy provider risk scores.</p>
          <div className="chart-container">
            <ResponsiveContainer width="100%" height={260}>
              <BarChart data={barData}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" />
                <XAxis dataKey="name" stroke="#94a3b8" fontSize={11} />
                <YAxis stroke="#94a3b8" fontSize={11} />
                <Tooltip
                  contentStyle={{ backgroundColor: '#11182c', borderColor: 'rgba(255,255,255,0.08)', borderRadius: '8px' }}
                  labelStyle={{ color: '#94a3b8', fontWeight: 600 }}
                  itemStyle={{ color: '#fff' }}
                />
                <Bar dataKey="Average Risk Score" fill="url(#blue-purple-gradient)" radius={[6, 6, 0, 0]}>
                  {barData.map((_, index) => (
                    <Cell key={`cell-${index}`} />
                  ))}
                </Bar>
                <defs>
                  <linearGradient id="blue-purple-gradient" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="0%" stopColor="#3b82f6" />
                    <stop offset="100%" stopColor="#8b5cf6" />
                  </linearGradient>
                </defs>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </div>

      <div className="dashboard-double-panel">
        {/* Top Suspicious Providers */}
        <div className="card list-card">
          <div className="panel-header">
            <h3 className="chart-title">Critical Providers Pending Action</h3>
            <button className="btn btn-secondary btn-sm" onClick={() => navigate('/queue')}>
              View Queue
            </button>
          </div>
          
          <div className="provider-list-table">
            <table>
              <thead>
                <tr>
                  <th>Provider ID</th>
                  <th>Type</th>
                  <th>Claims</th>
                  <th>Reimbursement</th>
                  <th>Risk Score</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {data.top_suspicious.slice(0, 5).map((prov) => (
                  <tr key={prov.provider_id}>
                    <td>
                      <span className="provider-code">{prov.provider_id}</span>
                    </td>
                    <td><span className="type-lbl">{prov.provider_type}</span></td>
                    <td>{prov.total_claims}</td>
                    <td>${prov.total_reimbursement.toLocaleString(undefined, {maximumFractionDigits: 0})}</td>
                    <td>
                      <span className={`badge badge-${prov.risk_level.toLowerCase()}`}>
                        {prov.risk_score.toFixed(1)} / {prov.risk_level}
                      </span>
                    </td>
                    <td>
                      <button 
                        className="action-link-btn"
                        onClick={() => navigate(`/provider/${prov.provider_id}`)}
                      >
                        Investigate <ArrowUpRight size={14} />
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>

        {/* Recent Activity / Active Audits */}
        <div className="card list-card">
          <h3 className="chart-title">Recent Investigation Activity</h3>
          <p className="chart-desc">Real-time status updates from clinical audit teams.</p>
          
          <div className="activity-timeline">
            {data.recent_activity.length === 0 ? (
              <div className="no-activity">
                <p>No active investigations started yet. Go to the queue to assign cases.</p>
              </div>
            ) : (
              data.recent_activity.map((act, index) => (
                <div className="timeline-item" key={index}>
                  <div className="timeline-badge-container">
                    <span className={`timeline-dot status-${act.status.toLowerCase().replace(' ', '-')}`}></span>
                  </div>
                  <div className="timeline-content">
                    <div className="timeline-header">
                      <h4 
                        className="timeline-title" 
                        onClick={() => navigate(`/provider/${act.provider_id}`)}
                      >
                        Provider {act.provider_id}
                      </h4>
                      <span className="timeline-time">
                        {new Date(act.updated_at).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                      </span>
                    </div>
                    <p className="timeline-desc">
                      Marked as <span className={`badge-text status-${act.status.toLowerCase().replace(' ', '-')}`}>{act.status}</span>.
                      Risk score was **{act.risk_score.toFixed(1)}** ({act.risk_level}).
                    </p>
                  </div>
                </div>
              ))
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
