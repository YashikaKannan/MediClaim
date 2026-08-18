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
}

const InvestigationQueue: React.FC = () => {
  const navigate = useNavigate();
  const [providers, setProviders] = useState<ProviderItem[]>([]);
  const [loading, setLoading] = useState(true);
  
  // Search & Filters
  const [search, setSearch] = useState('');
  const [riskLevel, setRiskLevel] = useState('');
  const [providerType, setProviderType] = useState('');
  const [stateFilter, setStateFilter] = useState<string>('');
  
  // Sorting
  const [sortBy, setSortBy] = useState('risk_score');
  const [sortOrder, setSortOrder] = useState('desc');
  
  // Pagination
  const [page, setPage] = useState(1);
  const [totalPages, setTotalPages] = useState(1);
  const [totalRecords, setTotalRecords] = useState(0);
  const pageSize = 15;

  useEffect(() => {
    fetchQueue();
  }, [page, riskLevel, providerType, stateFilter, sortBy, sortOrder]);

  const fetchQueue = async () => {
    try {
      setLoading(true);
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
          <p className="page-subtitle">Prioritized backlog of healthcare providers flagged for potential fraud, waste, or abuse.</p>
        </div>
        <div className="queue-stats">
          <span className="queue-badge-lbl">Total Flagged:</span>
          <span className="queue-badge-val">{totalRecords.toLocaleString()} Providers</span>
        </div>
      </div>

      {/* Filter Bar */}
      <div className="card filter-card">
        <form onSubmit={handleSearchSubmit} className="search-form">
          <div className="search-input-container">
            <Search size={18} className="search-icon" />
            <input 
              type="text" 
              placeholder="Search by Provider ID (e.g. PRV51003)..."
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
        </div>
      </div>

      {/* Table Section */}
      <div className="card table-card">
        {loading ? (
          <div className="table-loader">
            <RefreshCw size={24} className="spinning" />
            <p>Fetching queue records...</p>
          </div>
        ) : providers.length === 0 ? (
          <div className="empty-table">
            <p>No providers match the selected search criteria or filters.</p>
          </div>
        ) : (
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
                        <span className={`badge badge-${p.risk_level.toLowerCase()}`}>
                          {p.risk_score.toFixed(1)}
                        </span>
                      </td>
                      <td>
                        <span className={`badge status-${(p.investigation_status || 'New').toLowerCase().replace(' ', '-')}`}>
                          {p.investigation_status || 'New'}
                        </span>
                      </td>
                      <td>
                        <button 
                          className="btn btn-secondary btn-sm table-action-btn"
                          onClick={() => navigate(`/provider/${p.provider_id}`)}
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

            {/* Pagination Controls */}
            <div className="pagination-controls">
              <span className="pagination-info">
                Showing page <strong>{page}</strong> of <strong>{totalPages}</strong> ({totalRecords.toLocaleString()} total providers)
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
          </>
        )}
      </div>
    </div>
  );
};

export default InvestigationQueue;
