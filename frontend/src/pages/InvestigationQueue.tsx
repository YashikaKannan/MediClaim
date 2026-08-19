import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { 
  Search, 
  Filter, 
  ArrowUpDown, 
  ChevronLeft, 
  ChevronRight,
  Eye,
  RefreshCw
} from 'lucide-react';
import { getApiUrl } from '../config';
import './InvestigationQueue.css';

interface ProviderItem {
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
  investigation_status: string;
  assigned_investigator?: string;
  priority_score?: number;
}

interface ClaimItem {
  claim_id: string;
  provider_id: string;
  bene_id: string;
  risk_score: number;
  risk_category: string;
  is_anomaly: number;
  explanation_1: string;
  explanation_2: string;
  explanation_3: string;
  business_interpretation: string;
  claim_amount: number;
  claim_type: string;
  claim_date: string;
}

const InvestigationQueue: React.FC = () => {
  const navigate = useNavigate();
  const [activeTab, setActiveTab] = useState<'providers' | 'claims'>('providers');
  
  const [providers, setProviders] = useState<ProviderItem[]>([]);
  const [claims, setClaims] = useState<ClaimItem[]>([]);
  const [loading, setLoading] = useState(true);
  
  // Search & Filters
  const [search, setSearch] = useState('');
  const [riskLevel, setRiskLevel] = useState('');
  const [providerType, setProviderType] = useState('');
  const [stateFilter, setStateFilter] = useState<string>('');
  
  // Sorting
  const [sortBy, setSortBy] = useState('priority_score');
  const [sortOrder, setSortOrder] = useState('desc');
  
  // Pagination
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalRecords, setTotalRecords] = useState(0);
  const pageSize = 15;

  // Reset page and search when active tab changes
  useEffect(() => {
    setPage(1);
    setSearch('');
    setRiskLevel('');
    if (activeTab === 'claims') {
      setSortBy('risk_score');
    } else {
      setSortBy('priority_score');
    }
  }, [activeTab]);

  useEffect(() => {
    fetchQueue();
  }, [page, riskLevel, providerType, stateFilter, sortBy, sortOrder, activeTab]);

  const fetchQueue = async () => {
    try {
      setLoading(true);
      if (activeTab === 'providers') {
        let queryParams = `page=${page}&page_size=${pageSize}&sort_by=${sortBy}&sort_order=${sortOrder}`;
        if (search) queryParams += `&search=${search}`;
        if (riskLevel) queryParams += `&risk_level=${riskLevel}`;
        if (providerType) queryParams += `&provider_type=${providerType}`;
        if (stateFilter) queryParams += `&state=${stateFilter}`;
        
        const res = await fetch(`${getApiUrl()}/api/v1/providers?${queryParams}`);
        if (res.ok) {
          const json = await res.json();
          setProviders(json.data);
          setTotalPages(json.total_pages);
          setTotalRecords(json.total_records);
        }
      } else {
        let queryParams = `page=${page}&page_size=${pageSize}&sort_by=${sortBy === 'priority_score' ? 'risk_score' : sortBy}&sort_order=${sortOrder}`;
        if (search) queryParams += `&search=${search}`;
        if (riskLevel) queryParams += `&risk_level=${riskLevel}`;
        
        const res = await fetch(`${getApiUrl()}/api/v1/claims?${queryParams}`);
        if (res.ok) {
          const json = await res.json();
          setClaims(json.data);
          setTotalPages(json.total_pages);
          setTotalRecords(json.total_records);
        }
      }
    } catch (e) {
      console.error('Error fetching queue:', e);
    } finally {
      setLoading(false);
    }
  };

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    setPage(1);
    fetchQueue();
  };

  const toggleSort = (field: string) => {
    if (sortBy === field) {
      setSortOrder(sortOrder === 'desc' ? 'asc' : 'desc');
    } else {
      setSortBy(field);
      setSortOrder('desc');
    }
    setPage(1);
  };

  return (
    <div className="queue-page animate-fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">Investigation Queue</h1>
          <p className="page-subtitle">Prioritized backlog of healthcare providers and claim-level anomalies flagged for potential fraud, waste, or abuse.</p>
        </div>
        <div className="queue-stats">
          <span className="queue-badge-lbl">Total Flagged:</span>
          <span className="queue-badge-val">
            {activeTab === 'providers' ? `${totalRecords.toLocaleString()} Providers` : `${totalRecords.toLocaleString()} Claims`}
          </span>
        </div>
      </div>

      {/* Queue View Switcher */}
      <div className="card" style={{ padding: '0.75rem 1.25rem', display: 'flex', gap: '0.75rem', alignItems: 'center', border: '1px solid var(--border-color)', background: 'rgba(255,255,255,0.7)' }}>
        <span style={{ fontSize: '0.85rem', fontWeight: 700, color: 'var(--text-secondary)' }}>View Mode:</span>
        <button 
          className={`btn ${activeTab === 'providers' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setActiveTab('providers')}
          style={{ padding: '0.45rem 1.15rem', fontSize: '0.8rem', borderRadius: '8px' }}
        >
          Providers Backlog
        </button>
        <button 
          className={`btn ${activeTab === 'claims' ? 'btn-primary' : 'btn-secondary'}`}
          onClick={() => setActiveTab('claims')}
          style={{ padding: '0.45rem 1.15rem', fontSize: '0.8rem', borderRadius: '8px' }}
        >
          Claims Triage List
        </button>
      </div>

      {/* Filter Bar */}
      <div className="card filter-card">
        <form onSubmit={handleSearchSubmit} className="search-form">
          <div className="search-input-container">
            <Search size={18} className="search-icon" />
            <input 
              type="text" 
              placeholder={activeTab === 'providers' ? "Search by Provider ID (e.g. PRV51003)..." : "Search by Claim ID or Provider ID..."}
              value={search}
              onChange={(e) => setSearch(e.target.value)}
            />
          </div>
          <button type="submit" className="btn btn-primary search-btn">
            Search
          </button>
        </form>

        <div className="filters-container">
          <div className="filter-group">
            <Filter size={14} className="filter-group-icon" />
            <span className="filter-label">Risk Severity</span>
            <select value={riskLevel} onChange={(e) => { setRiskLevel(e.target.value); setPage(1); }}>
              <option value="">All Severities</option>
              <option value="Critical">Critical</option>
              <option value="High">High</option>
              <option value="Medium">Medium</option>
              <option value="Low">Low</option>
            </select>
          </div>

          {activeTab === 'providers' && (
            <>
              <div className="filter-group">
                <Filter size={14} className="filter-group-icon" />
                <span className="filter-label">Facility Type</span>
                <select value={providerType} onChange={(e) => { setProviderType(e.target.value); setPage(1); }}>
                  <option value="">All Types</option>
                  <option value="Inpatient-heavy">Inpatient-Heavy</option>
                  <option value="Outpatient-heavy">Outpatient-Heavy</option>
                </select>
              </div>

              <div className="filter-group">
                <Filter size={14} className="filter-group-icon" />
                <span className="filter-label">Operating State</span>
                <input 
                  type="number" 
                  placeholder="e.g. 39" 
                  className="state-filter-input"
                  value={stateFilter} 
                  onChange={(e) => { setStateFilter(e.target.value); setPage(1); }} 
                />
              </div>
            </>
          )}
        </div>
      </div>

      {/* Table Section */}
      <div className="card table-card">
        {loading ? (
          <div className="table-loader">
            <RefreshCw size={24} className="spinning" />
            <p>Fetching queue records...</p>
          </div>
        ) : (activeTab === 'providers' ? providers.length : claims.length) === 0 ? (
          <div className="empty-table">
            <p>No records match the selected search criteria or filters.</p>
          </div>
        ) : activeTab === 'providers' ? (
          <>
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>Provider ID</th>
                    <th onClick={() => toggleSort('total_claims')} className="sortable-th">
                      Claims <ArrowUpDown size={12} />
                    </th>
                    <th onClick={() => toggleSort('total_reimbursement')} className="sortable-th">
                      Reimbursements <ArrowUpDown size={12} />
                    </th>
                    <th>Beneficiaries</th>
                    <th>Inpatient Ratio</th>
                    <th onClick={() => toggleSort('priority_score')} className="sortable-th">
                      Priority Score <ArrowUpDown size={12} />
                    </th>
                    <th onClick={() => toggleSort('risk_score')} className="sortable-th">
                      Risk Score <ArrowUpDown size={12} />
                    </th>
                    <th>Audit Status</th>
                    <th>Action</th>
                  </tr>
                </thead>
                <tbody>
                  {providers.map((p) => (
                    <tr key={p.provider_id}>
                      <td>
                        <span className="provider-code">{p.provider_id}</span>
                        <span className="state-sub-lbl">State {p.primary_state}</span>
                      </td>
                      <td>{p.total_claims}</td>
                      <td>${p.total_reimbursement.toLocaleString(undefined, {maximumFractionDigits: 0})}</td>
                      <td>{p.total_beneficiaries}</td>
                      <td>{(p.inpatient_ratio * 100).toFixed(1)}%</td>
                      <td>
                        <span className="priority-val-lbl">
                          {p.priority_score ? p.priority_score.toLocaleString(undefined, {maximumFractionDigits: 0}) : '0'}
                        </span>
                      </td>
                      <td>
                        <span className={`badge badge-${p.risk_level.toLowerCase()}`}>
                          {p.risk_score.toFixed(1)}
                        </span>
                      </td>
                      <td>
                        <div style={{ display: 'flex', flexDirection: 'column', gap: '4px' }}>
                          <span className={`badge status-${(p.investigation_status || 'New').toLowerCase().replace(' ', '-')}`}>
                            {p.investigation_status || 'New'}
                          </span>
                          {p.assigned_investigator && p.assigned_investigator !== 'Unassigned' && (
                            <span style={{ fontSize: '0.7rem', color: 'var(--text-muted)', fontWeight: 600 }}>
                              Assigned: {p.assigned_investigator}
                            </span>
                          )}
                        </div>
                      </td>
                      <td>
                        <button 
                          className="btn btn-secondary btn-sm table-action-btn"
                          onClick={() => {
                            localStorage.setItem('mediclaim_selected_provider', p.provider_id);
                            window.dispatchEvent(new Event('mediclaim-provider-selected'));
                            navigate(`/provider/${p.provider_id}`);
                          }}
                        >
                          <Eye size={14} />
                          Review
                        </button>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </>
        ) : (
          <>
            <div className="table-container">
              <table>
                <thead>
                  <tr>
                    <th>Claim ID</th>
                    <th>Beneficiary ID</th>
                    <th>Provider ID</th>
                    <th onClick={() => toggleSort('claim_amount')} className="sortable-th">
                      Claim Amount <ArrowUpDown size={12} />
                    </th>
                    <th>Type</th>
                    <th>Claim Date</th>
                    <th onClick={() => toggleSort('risk_score')} className="sortable-th">
                      Risk Score <ArrowUpDown size={12} />
                    </th>
                    <th>Status</th>
                    <th>Business Interpretation (RAG Explanations)</th>
                  </tr>
                </thead>
                <tbody>
                  {claims.map((c) => {
                    let rLevel = 'low';
                    if (c.risk_score >= 85) rLevel = 'critical';
                    else if (c.risk_score >= 65) rLevel = 'high';
                    else if (c.risk_score >= 35) rLevel = 'medium';

                    return (
                      <tr key={c.claim_id}>
                        <td>
                          <span className="provider-code" style={{ color: 'var(--accent-blue)', fontWeight: 700 }}>
                            {c.claim_id}
                          </span>
                        </td>
                        <td>{c.bene_id}</td>
                        <td>
                          <button 
                            className="btn btn-secondary btn-sm"
                            style={{ padding: '0.2rem 0.5rem', fontSize: '0.75rem', fontWeight: 600 }}
                            onClick={() => {
                              localStorage.setItem('mediclaim_selected_provider', c.provider_id);
                              window.dispatchEvent(new Event('mediclaim-provider-selected'));
                              navigate(`/provider/${c.provider_id}`);
                            }}
                          >
                            {c.provider_id}
                          </button>
                        </td>
                        <td>
                          {new Intl.NumberFormat('en-US', { style: 'currency', currency: 'USD', maximumFractionDigits: 0 }).format(c.claim_amount || 0)}
                        </td>
                        <td>
                          <span className={`badge ${c.claim_type === 'Inpatient' ? 'status-new' : 'status-resolved'}`} style={{ fontSize: '0.7rem' }}>
                            {c.claim_type || 'Outpatient'}
                          </span>
                        </td>
                        <td>{c.claim_date || 'N/A'}</td>
                        <td>
                          <span className={`badge badge-${rLevel}`}>
                            {Number(c.risk_score || 0).toFixed(1)}
                          </span>
                        </td>
                        <td>
                          <span className={`badge ${c.is_anomaly ? 'btn-danger' : 'status-resolved'}`} style={{ color: c.is_anomaly ? '#fff' : '', fontSize: '0.7rem' }}>
                            {c.is_anomaly ? 'Anomaly' : 'Normal'}
                          </span>
                        </td>
                        <td style={{ maxWidth: '350px', fontSize: '0.8rem', color: 'var(--text-secondary)', lineHeight: '1.4' }}>
                          <div style={{ maxHeight: '60px', overflowY: 'auto', paddingRight: '4px' }}>
                            {c.business_interpretation || 'No interpretation generated.'}
                          </div>
                        </td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </>
        )}

        {/* Pagination Controls */}
        <div className="pagination-controls">
          <span className="pagination-info">
            Showing page <strong>{page}</strong> of <strong>{totalPages}</strong> ({totalRecords.toLocaleString()} total records)
          </span>
          
          <div className="pagination-buttons">
            <button 
              className="btn btn-secondary btn-sm"
              onClick={() => setPage(p => Math.max(1, p - 1))}
              disabled={page === 1}
            >
              <ChevronLeft size={16} />
              Prev
            </button>
            <button 
              className="btn btn-secondary btn-sm"
              onClick={() => setPage(p => Math.min(totalPages, p + 1))}
              disabled={page === totalPages}
            >
              Next
              <ChevronRight size={16} />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};

export default InvestigationQueue;
