import { useState, useMemo } from 'react'
import { BrowserRouter, NavLink, Route, Routes, useNavigate, useParams } from 'react-router-dom'
import { PieChart, Pie, Cell, ResponsiveContainer, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip } from 'recharts'
import { Bell, Download, Search } from 'lucide-react'
import './App.css'

import { SearchFilterToolbar, type FilterConfig, type SortOption } from './components/SearchFilterToolbar'
import { ActiveFilterChips, type ActiveFilterItem } from './components/ActiveFilterChips'
import { EmptyFilterState } from './components/EmptyFilterState'

import dashboardData from './data/mockDashboard.json'
import providersData from './data/mockProviders.json'
import queueData from './data/mockQueue.json'
import claimsData from './data/mockClaims.json'
import reportsData from './data/mockReports.json'

const riskTierDefinitions = [
  { tier: 'Low', range: '0-30', color: 'Green' },
  { tier: 'Medium', range: '31-60', color: 'Amber' },
  { tier: 'High', range: '61-80', color: 'Orange' },
  { tier: 'Critical', range: '81-100', color: 'Red' },
]

const modelInventory = [
  {
    model: 'Peer Benchmark Analysis',
    purpose: 'Provider comparison against specialty peers',
    weight: '35%',
    status: 'Active',
  },
  {
    model: 'Isolation Forest',
    purpose: 'Provider anomaly detection',
    weight: '25%',
    status: 'Active',
  },
  {
    model: 'Local Outlier Factor (LOF)',
    purpose: 'Local density anomaly detection',
    weight: '20%',
    status: 'Active',
  },
  {
    model: 'Rule Engine',
    purpose: 'Duplicate billing and payment integrity rules',
    weight: '20%',
    status: 'Active',
  },
  {
    model: 'One-Class SVM',
    purpose: 'Advanced anomaly detection',
    weight: '-',
    status: 'Planned',
  },
  {
    model: 'Behavioral Drift Detection',
    purpose: 'Billing trend analysis',
    weight: '-',
    status: 'Planned',
  },
]

const globalModelWeights = [
  { label: 'Peer Analysis', value: 35, color: '#0F172A' },
  { label: 'Isolation Forest', value: 25, color: '#F59E0B' },
  { label: 'LOF', value: 20, color: '#64748B' },
  { label: 'Rule Engine', value: 20, color: '#CBD5E1' },
]

const governanceDataSources = [
  { name: 'Carrier Claims', records: 1360028, refreshDate: '2026-08-15', status: 'Connected' },
  { name: 'Outpatient Claims', records: 482411, refreshDate: '2026-08-15', status: 'Connected' },
  { name: 'Inpatient Claims', records: 224730, refreshDate: '2026-08-15', status: 'Connected' },
  { name: 'Provider Master Dataset', records: 4287, refreshDate: '2026-08-14', status: 'Connected' },
  { name: 'Beneficiary Dataset', records: 912553, refreshDate: '2026-08-15', status: 'Connected' },
  { name: 'LEIE Exclusion List', records: 79264, refreshDate: '2026-08-13', status: 'Connected' },
]

const dataQualityMetrics = {
  totalRecordsProcessed: 2143273,
  missingValues: 19425,
  duplicateRecords: 3812,
  failedValidations: 1024,
  dataQualityScore: 98.4,
}

const businessValidationMetrics = {
  flaggedProviders: 143,
  leieMatches: 12,
  leieMatchRate: 8.4,
  potentialExcessAmountDetected: 18400000,
  claimsReviewed: 1360028,
}

const systemHealth = [
  { component: 'Provider Scoring Pipeline', status: 'Healthy' },
  { component: 'Claim Scoring Pipeline', status: 'Healthy' },
  { component: 'Risk Fusion Engine', status: 'Healthy' },
  { component: 'Report Generation', status: 'Healthy' },
  { component: 'AI Assistant', status: 'Healthy' },
]

const auditLogRows = [
  { timestamp: '08:45 AM', event: 'Provider scoring completed', user: 'System', status: 'Success' },
  { timestamp: '09:12 AM', event: 'Monthly leakage report generated', user: 'Hannah Smith', status: 'Success' },
]

type RiskBand = 'Low Risk' | 'Medium Risk' | 'High Risk' | 'Critical Risk'

type DashboardRow = {
  providersAnalyzed: number
  claimsAnalyzed: number
  flaggedProviders: number
  potentialFinancialLeakage: number
  riskDistribution: { label: string; value: number; color: string }[]
  leakageBySpecialty: { label: string; value: number }[]
  leakageByState: { label: string; value: number }[]
  leakageByProviderType: { label: string; value: number }[]
  topPriorityCases: {
    providerName: string
    npi: string
    riskScore: number
    potentialLeakage: number
    priorityLevel: string
    status: string
    providerId: string
  }[]
  systemModelWeights: { label: string; value: number }[]
}

type Provider = {
  id: string
  name: string
  npi: string
  specialty: string
  location: string
  totalClaims: number
  totalPaid: number
  riskScore: number
  riskLabel: string
  reasons: {
    reason: string
    source: string
    status: string
  }[]
  peerMetrics: {
    metric: string
    provider: number
    peerMedian: number
    providerDisplay: string
    peerDisplay: string
  }[]
  peerGroup: {
    specialty: string
    region: string
    peerCount: number
    threshold: string
    confidence: string
  }
  potentialExcessPayment: number
  scoreBreakdown: { label: string; value: number }[]
  associatedClaims: string[]
}

type QueueRow = {
  id: number
  priorityRank: number
  providerName: string
  npi: string
  specialty: string
  riskScore: number
  potentialLeakage: number
  status: string
  assignedInvestigator: string
  providerId: string
}

type Claim = {
  id: string
  providerId: string
  providerName: string
  beneficiaryId: string
  serviceDate: string
  claimAmount: number
  paidAmount: number
  riskScore: number
  riskLabel: string
  reasons: {
    reason: string
    source: string
    status: string
  }[]
  procedureBreakdown: {
    hcpcs: string
    description: string
    units: number
    paidAmount: number
    flag: string
  }[]
  potentialExcessPayment: number
}

type Report = {
  id: string
  name: string
  description: string
  exportTypes: string[]
}

const navItems = [
  { path: '/', label: 'Dashboard' },
  { path: '/queue', label: 'Investigation Queue' },
  { path: '/providers/provider-chen', label: 'Provider Investigations' },
  { path: '/claims/CLM-824791', label: 'Claim Investigations' },
  { path: '/reports', label: 'Reports' },
  { path: '/assistant', label: 'AI Investigation Assistant' },
  { path: '/governance', label: 'Model Governance' },
]

const dashboardDataTyped = dashboardData as DashboardRow
const providers = providersData as Provider[]
const queueRows = queueData as QueueRow[]
const claims = claimsData as Claim[]
const reportTemplates = reportsData as Report[]

const formatCurrency = (value: number) =>
  new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    maximumFractionDigits: 0,
  }).format(value)

const formatCompactCurrency = (value: number) =>
  new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    notation: 'compact',
    maximumFractionDigits: 2,
  }).format(value)

const formatNumber = (value: number) => new Intl.NumberFormat('en-US').format(value)

const getRiskBand = (score: number): RiskBand => {
  if (score >= 81) return 'Critical Risk'
  if (score >= 61) return 'High Risk'
  if (score >= 31) return 'Medium Risk'
  return 'Low Risk'
}

const getRiskClass = (score: number) => {
  const band = getRiskBand(score)
  if (band === 'Critical Risk') return 'critical'
  if (band === 'High Risk') return 'high'
  if (band === 'Medium Risk') return 'medium'
  return 'low'
}

const getStateFromLocation = (loc: string) => {
  if (!loc) return ''
  const parts = loc.split(',')
  return parts.length > 1 ? parts[1].trim() : loc.trim()
}

const matchLeakageRange = (amount: number, range: string) => {
  if (range === 'all') return true
  if (range === 'under50k') return amount < 50000
  if (range === '50k-100k') return amount >= 50000 && amount <= 100000
  if (range === '100k-250k') return amount > 100000 && amount <= 250000
  if (range === 'over250k') return amount > 250000
  return true
}

const matchClaimAmountRange = (amount: number, range: string) => {
  if (range === 'all') return true
  if (range === 'under1k') return amount < 1000
  if (range === '1k-5k') return amount >= 1000 && amount <= 5000
  if (range === '5k-10k') return amount > 5000 && amount <= 10000
  if (range === 'over10k') return amount > 10000
  return true
}

const matchRiskLevel = (score: number, level: string) => {
  if (level === 'all') return true
  const band = getRiskBand(score)
  if (level === 'critical') return band === 'Critical Risk'
  if (level === 'high') return band === 'High Risk'
  if (level === 'medium') return band === 'Medium Risk'
  if (level === 'low') return band === 'Low Risk'
  return true
}

const matchDateRange = (dateStr: string, range: string) => {
  if (range === 'all') return true
  if (!dateStr) return false
  const claimTime = new Date(dateStr).getTime()
  // Mock dataset reference date anchor: 2026-08-15
  const baseTime = new Date('2026-08-15').getTime()
  const diffDays = Math.max(0, Math.floor((baseTime - claimTime) / (1000 * 60 * 60 * 24)))

  if (range === 'last7') return diffDays <= 7
  if (range === 'last30') return diffDays <= 30
  if (range === 'last90') return diffDays <= 90
  return true
}

function RiskBadge({ score }: { score: number }) {
  const label = getRiskBand(score)
  const className = `risk-badge risk-₹{getRiskClass(score)}`

  return (
    <span className={className}>
      {label}
      <strong>{score}</strong>
    </span>
  )
}

function StatusBadge({ status }: { status: string }) {
  const isLive = status === 'LIVE' || status === 'Investigating' || status === 'Assigned'
  return <span className={isLive ? 'status-badge live' : 'status-badge planned'}>{status}</span>
}

function GovernanceStatusBadge({ status }: { status: string }) {
  return <span className={status === 'Active' ? 'gov-badge active' : 'gov-badge planned'}>{status}</span>
}

function AppShell() {
  const navigate = useNavigate()
  const [globalSearch, setGlobalSearch] = useState('')

  const handleGlobalSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault()
    if (!globalSearch.trim()) return
    navigate(`/queue`)
  }

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark">M</div>
          <div>
            <p className="eyebrow subtle">Payment Integrity</p>
            <h1>MEDICLAIM</h1>
          </div>
        </div>

        <form className="searchbar" onSubmit={handleGlobalSearchSubmit} aria-label="Search providers and claims">
          <Search size={18} />
          <input
            type="text"
            value={globalSearch}
            onChange={(e) => setGlobalSearch(e.target.value)}
            placeholder="Search Provider Name, NPI, Claim ID, Beneficiary ID, Specialty"
          />
        </form>

        <div className="header-actions">
          <button type="button" className="icon-button" aria-label="Notifications">
            <Bell size={16} />
          </button>
          <button type="button" className="export-button">
            <Download size={16} />
            Export
          </button>
          <div className="user-card">
            <div className="avatar">HS</div>
            <div>
              <strong>Hannah Smith</strong>
              <small>SIU Lead</small>
            </div>
          </div>
        </div>
      </header>

      <div className="layout-shell">
        <aside className="sidebar">
          <nav>
            {navItems.map(({ path, label }) => (
              <NavLink
                key={path}
                to={path}
                className={({ isActive }) => `sidebar-link ₹{isActive ? 'active' : ''}`}
              >
                {label}
              </NavLink>
            ))}
          </nav>
        </aside>

        <main className="main-content">
          <Routes>
            <Route path="/" element={<DashboardPage />} />
            <Route path="/queue" element={<QueuePage />} />
            <Route path="/providers" element={<ProviderPage />} />
            <Route path="/providers/:providerId" element={<ProviderPage />} />
            <Route path="/claims" element={<ClaimPage />} />
            <Route path="/claims/:claimId" element={<ClaimPage />} />
            <Route path="/reports" element={<ReportsPage />} />
            <Route path="/assistant" element={<AssistantPage />} />
            <Route path="/governance" element={<ModelGovernancePage />} />
          </Routes>
        </main>
      </div>
    </div>
  )
}

function DashboardPage() {
  const navigate = useNavigate()

  return (
    <div className="page-grid">
      <section className="kpi-grid">
        <article className="panel stat-card">
          <div className="stat-header">
            <span>Providers Analyzed</span>
            <span className="delta">+8.2%</span>
          </div>
          <div className="stat-value">{formatNumber(dashboardDataTyped.providersAnalyzed)}</div>
        </article>
        <article className="panel stat-card">
          <div className="stat-header">
            <span>Claims Analyzed</span>
            <span className="delta">+12.7%</span>
          </div>
          <div className="stat-value">{formatNumber(dashboardDataTyped.claimsAnalyzed)}</div>
        </article>
        <article className="panel stat-card">
          <div className="stat-header">
            <span>Flagged Providers</span>
            <span className="delta">31 critical</span>
          </div>
          <div className="stat-value">{formatNumber(dashboardDataTyped.flaggedProviders)}</div>
        </article>
        <article className="panel stat-card">
          <div className="stat-header">
            <span>Potential Financial Leakage</span>
            <span className="delta">+₹2.1M</span>
          </div>
          <div className="stat-value">{formatCompactCurrency(dashboardDataTyped.potentialFinancialLeakage)}</div>
        </article>
      </section>

      <section className="panel feature-panel panel-8">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Risk Distribution</p>
            <h3>Risk Distribution</h3>
          </div>
        </div>
        <div className="donut-wrap">
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={dashboardDataTyped.riskDistribution} dataKey="value" nameKey="label" innerRadius={55} outerRadius={82} paddingAngle={3}>
                {dashboardDataTyped.riskDistribution.map((entry) => (
                  <Cell key={entry.label} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip formatter={(value) => [`₹{Number(value ?? 0)}%`, 'Share']} />
            </PieChart>
          </ResponsiveContainer>
          <div className="legend-list">
            {dashboardDataTyped.riskDistribution.map((item) => (
              <div key={item.label} className="legend-item">
                <span className="color-dot" style={{ background: item.color }} />
                <span>{item.label}</span>
                <strong>{item.value}%</strong>
              </div>
            ))}
          </div>
        </div>
      </section>

      <aside className="panel panel-4 model-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Governance</p>
            <h3>System Model Weights</h3>
          </div>
        </div>
        <div className="model-list">
          {dashboardDataTyped.systemModelWeights.map((item) => (
            <div key={item.label} className="model-row">
              <span>{item.label}</span>
              <strong>{item.value}%</strong>
            </div>
          ))}
        </div>
      </aside>

      <section className="panel panel-12 chart-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Financial Leakage Analysis</p>
            <h3>Leakage by Specialty, State, and Provider Type</h3>
          </div>
        </div>
        <div className="bar-grid">
          <div className="bar-card">
            <h4>Leakage by Specialty</h4>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={dashboardDataTyped.leakageBySpecialty} margin={{ top: 8, right: 10, left: 0, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="label" tickLine={false} axisLine={false} />
                <YAxis tickLine={false} axisLine={false} />
                <Tooltip formatter={(value) => [`${Number(value ?? 0)}%`, 'Leakage']} />
                <Bar dataKey="value" fill="#0F172A" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="bar-card">
            <h4>Leakage by State</h4>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={dashboardDataTyped.leakageByState} margin={{ top: 8, right: 10, left: 0, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="label" tickLine={false} axisLine={false} />
                <YAxis tickLine={false} axisLine={false} />
                <Tooltip formatter={(value) => [`${Number(value ?? 0)}%`, 'Leakage']} />
                <Bar dataKey="value" fill="#F59E0B" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
          <div className="bar-card">
            <h4>Leakage by Provider Type</h4>
            <ResponsiveContainer width="100%" height={180}>
              <BarChart data={dashboardDataTyped.leakageByProviderType} margin={{ top: 8, right: 10, left: 0, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} />
                <XAxis dataKey="label" tickLine={false} axisLine={false} />
                <YAxis tickLine={false} axisLine={false} />
                <Tooltip formatter={(value) => [`${Number(value ?? 0)}%`, 'Leakage']} />
                <Bar dataKey="value" fill="#64748B" radius={[8, 8, 0, 0]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>
      </section>

      <section className="panel panel-8 priority-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Top Priority Cases</p>
            <h3>Investigation Priorities</h3>
          </div>
        </div>
        <table className="table data-table">
          <thead>
            <tr>
              <th>Provider Name</th>
              <th>NPI</th>
              <th>Risk Score</th>
              <th>Potential Leakage</th>
              <th>Priority Level</th>
              <th>Status</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {dashboardDataTyped.topPriorityCases.map((row) => (
              <tr key={row.providerName}>
                <td>{row.providerName}</td>
                <td>{row.npi}</td>
                <td><RiskBadge score={row.riskScore} /></td>
                <td>{formatCurrency(row.potentialLeakage)}</td>
                <td>{row.priorityLevel}</td>
                <td><StatusBadge status={row.status} /></td>
                <td>
                  <button className="secondary-action" type="button" onClick={() => navigate(`/providers/${row.providerId}`)}>
                    Investigate
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  )
}

function QueuePage() {
  const navigate = useNavigate()
  const [searchTerm, setSearchTerm] = useState('')
  const [riskFilter, setRiskFilter] = useState('all')
  const [specialtyFilter, setSpecialtyFilter] = useState('all')
  const [stateFilter, setStateFilter] = useState('all')
  const [statusFilter, setStatusFilter] = useState('all')
  const [leakageFilter, setLeakageFilter] = useState('all')
  const [fwaFilter, setFwaFilter] = useState('all')
  const [sortBy, setSortBy] = useState('priority-desc')

  // Dynamic filter options from existing data
  const specialtyOptions = useMemo(() => {
    const list = Array.from(new Set(queueRows.map((r) => r.specialty))).filter(Boolean)
    return list.map((s) => ({ value: s, label: s }))
  }, [])

  const stateOptions = useMemo(() => {
    const states = Array.from(
      new Set(
        queueRows
          .map((r) => {
            const p = providers.find((prov) => prov.id === r.providerId)
            return p ? getStateFromLocation(p.location) : ''
          })
          .filter(Boolean)
      )
    )
    return states.map((st) => ({ value: st, label: st }))
  }, [])

  const statusOptions = useMemo(() => {
    const list = Array.from(new Set(queueRows.map((r) => r.status))).filter(Boolean)
    return list.map((s) => ({ value: s, label: s }))
  }, [])

  const fwaOptions = useMemo(() => {
    const sources = Array.from(
      new Set(
        providers.flatMap((p) => p.reasons.map((r) => r.source)).filter(Boolean)
      )
    )
    return sources.map((s) => ({ value: s, label: s }))
  }, [])

  // Filter queue rows using AND logic
  const filteredQueue = useMemo(() => {
    return queueRows.filter((row) => {
      // Search: Provider Name, NPI, Specialty
      if (searchTerm.trim()) {
        const query = searchTerm.toLowerCase().trim()
        const matchName = row.providerName.toLowerCase().includes(query)
        const matchNpi = row.npi.includes(query)
        const matchSpec = row.specialty.toLowerCase().includes(query)
        if (!matchName && !matchNpi && !matchSpec) return false
      }

      // Risk Level
      if (!matchRiskLevel(row.riskScore, riskFilter)) return false

      // Specialty
      if (specialtyFilter !== 'all' && row.specialty !== specialtyFilter) return false

      // State
      if (stateFilter !== 'all') {
        const p = providers.find((prov) => prov.id === row.providerId)
        if (!p || getStateFromLocation(p.location) !== stateFilter) return false
      }

      // Status
      if (statusFilter !== 'all' && row.status !== statusFilter) return false

      // Potential Leakage
      if (!matchLeakageRange(row.potentialLeakage, leakageFilter)) return false

      // FWA Type
      if (fwaFilter !== 'all') {
        const p = providers.find((prov) => prov.id === row.providerId)
        if (!p || !p.reasons.some((r) => r.source === fwaFilter || r.reason.toLowerCase().includes(fwaFilter.toLowerCase()))) {
          return false
        }
      }

      return true
    })
  }, [searchTerm, riskFilter, specialtyFilter, stateFilter, statusFilter, leakageFilter, fwaFilter])

  // Sorting
  const sortedQueue = useMemo(() => {
    const items = [...filteredQueue]
    if (sortBy === 'priority-desc') {
      // Default: Priority Score descending
      return items.sort((a, b) => b.priorityRank - a.priorityRank)
    }
    if (sortBy === 'risk-desc') {
      return items.sort((a, b) => b.riskScore - a.riskScore)
    }
    if (sortBy === 'leakage-desc') {
      return items.sort((a, b) => b.potentialLeakage - a.potentialLeakage)
    }
    if (sortBy === 'recent') {
      return items.sort((a, b) => a.id - b.id)
    }
    return items
  }, [filteredQueue, sortBy])

  // Active filter chips
  const activeChips: ActiveFilterItem[] = useMemo(() => {
    const chips: ActiveFilterItem[] = []
    if (searchTerm.trim()) {
      chips.push({
        id: 'search',
        label: 'Search',
        value: searchTerm,
        displayLabel: `Search: "${searchTerm}"`,
        onRemove: () => setSearchTerm(''),
      })
    }
    if (riskFilter !== 'all') {
      chips.push({
        id: 'risk',
        label: 'Risk',
        value: riskFilter,
        displayLabel: `Risk: ${riskFilter.charAt(0).toUpperCase() + riskFilter.slice(1)}`,
        onRemove: () => setRiskFilter('all'),
      })
    }
    if (specialtyFilter !== 'all') {
      chips.push({
        id: 'specialty',
        label: 'Specialty',
        value: specialtyFilter,
        displayLabel: `Specialty: ${specialtyFilter}`,
        onRemove: () => setSpecialtyFilter('all'),
      })
    }
    if (stateFilter !== 'all') {
      chips.push({
        id: 'state',
        label: 'State',
        value: stateFilter,
        displayLabel: `State: ${stateFilter}`,
        onRemove: () => setStateFilter('all'),
      })
    }
    if (statusFilter !== 'all') {
      chips.push({
        id: 'status',
        label: 'Status',
        value: statusFilter,
        displayLabel: `Status: ${statusFilter}`,
        onRemove: () => setStatusFilter('all'),
      })
    }
    if (leakageFilter !== 'all') {
      const labelMap: Record<string, string> = {
        under50k: '< $50K',
        '50k-100k': '$50K - $100K',
        '100k-250k': '$100K - $250K',
        over250k: '> $250K',
      }
      chips.push({
        id: 'leakage',
        label: 'Leakage',
        value: leakageFilter,
        displayLabel: `Leakage: ${labelMap[leakageFilter] || leakageFilter}`,
        onRemove: () => setLeakageFilter('all'),
      })
    }
    if (fwaFilter !== 'all') {
      chips.push({
        id: 'fwa',
        label: 'FWA',
        value: fwaFilter,
        displayLabel: `FWA: ${fwaFilter}`,
        onRemove: () => setFwaFilter('all'),
      })
    }
    return chips
  }, [searchTerm, riskFilter, specialtyFilter, stateFilter, statusFilter, leakageFilter, fwaFilter])

  const clearAllFilters = () => {
    setSearchTerm('')
    setRiskFilter('all')
    setSpecialtyFilter('all')
    setStateFilter('all')
    setStatusFilter('all')
    setLeakageFilter('all')
    setFwaFilter('all')
    setSortBy('priority-desc')
  }

  const filterConfigs: FilterConfig[] = [
    {
      id: 'risk',
      label: 'Risk Level',
      value: riskFilter,
      options: [
        { value: 'critical', label: 'Critical' },
        { value: 'high', label: 'High' },
        { value: 'medium', label: 'Medium' },
        { value: 'low', label: 'Low' },
      ],
      onChange: setRiskFilter,
    },
    {
      id: 'fwa',
      label: 'FWA Type',
      value: fwaFilter,
      options: fwaOptions,
      onChange: setFwaFilter,
    },
    {
      id: 'specialty',
      label: 'Specialty',
      value: specialtyFilter,
      options: specialtyOptions,
      onChange: setSpecialtyFilter,
    },
    {
      id: 'state',
      label: 'State',
      value: stateFilter,
      options: stateOptions,
      onChange: setStateFilter,
    },
    {
      id: 'status',
      label: 'Status',
      value: statusFilter,
      options: statusOptions,
      onChange: setStatusFilter,
    },
    {
      id: 'leakage',
      label: 'Potential Leakage',
      value: leakageFilter,
      options: [
        { value: 'under50k', label: '< ₹50K' },
        { value: '50k-100k', label: '₹50K - $₹100K' },
        { value: '100k-250k', label: '₹100K - ₹250K' },
        { value: 'over250k', label: '> ₹250K' },
      ],
      onChange: setLeakageFilter,
    },
  ]

  const sortConfigs: SortOption[] = [
    { value: 'priority-desc', label: 'Priority Score descending' },
    { value: 'risk-desc', label: 'Risk Score descending' },
    { value: 'leakage-desc', label: 'Potential Leakage descending' },
    { value: 'recent', label: 'Most Recent' },
  ]

  return (
    <section className="panel panel-12">
      <div className="panel-header stack-header">
        <div>
          <p className="eyebrow">Operational Workbench</p>
          <h3>Investigation Queue</h3>
        </div>
      </div>

      <SearchFilterToolbar
        searchPlaceholder="Search provider name, NPI, specialty..."
        searchValue={searchTerm}
        onSearchChange={setSearchTerm}
        filters={filterConfigs}
        sortOptions={sortConfigs}
        sortValue={sortBy}
        onSortChange={setSortBy}
      />

      <ActiveFilterChips chips={activeChips} onClearAll={clearAllFilters} />

      <div className="result-count-bar">
        <span className="result-count-text">
          {sortedQueue.length} {sortedQueue.length === 1 ? 'investigation case' : 'investigation cases'} found
        </span>
      </div>

      <div className="priority-summary">
        <strong>Priority Score = Risk Score × Potential Leakage</strong>
      </div>

      {sortedQueue.length === 0 ? (
        <EmptyFilterState
          title="No matching records found"
          message="No investigation cases matched your current search and filter settings. Try adjusting your query or resetting filters."
          onClearFilters={clearAllFilters}
        />
      ) : (
        <table className="table data-table">
          <thead>
            <tr>
              <th>Priority Rank</th>
              <th>Provider Name</th>
              <th>NPI</th>
              <th>Specialty</th>
              <th>Risk Score</th>
              <th>Potential Leakage</th>
              <th>Status</th>
              <th>Assigned Investigator</th>
              <th>Action</th>
            </tr>
          </thead>
          <tbody>
            {sortedQueue.map((row) => (
              <tr key={row.id}>
                <td>{row.priorityRank}</td>
                <td>{row.providerName}</td>
                <td>{row.npi}</td>
                <td>{row.specialty}</td>
                <td><RiskBadge score={row.riskScore} /></td>
                <td>{formatCurrency(row.potentialLeakage)}</td>
                <td><StatusBadge status={row.status} /></td>
                <td>{row.assignedInvestigator}</td>
                <td>
                  <button
                    className="secondary-action"
                    type="button"
                    onClick={() => navigate(`/providers/${row.providerId}`)}
                  >
                    Open
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </section>
  )
}

function ProviderPage() {
  const { providerId } = useParams()
  const navigate = useNavigate()

  const [searchTerm, setSearchTerm] = useState('')
  const [riskFilter, setRiskFilter] = useState('all')
  const [specialtyFilter, setSpecialtyFilter] = useState('all')
  const [stateFilter, setStateFilter] = useState('all')
  const [statusFilter, setStatusFilter] = useState('all')
  const [leakageFilter, setLeakageFilter] = useState('all')
  const [fwaFilter, setFwaFilter] = useState('all')

  // Dynamic filter options from existing provider dataset
  const specialtyOptions = useMemo(() => {
    const list = Array.from(new Set(providers.map((p) => p.specialty))).filter(Boolean)
    return list.map((s) => ({ value: s, label: s }))
  }, [])

  const stateOptions = useMemo(() => {
    const list = Array.from(new Set(providers.map((p) => getStateFromLocation(p.location)))).filter(Boolean)
    return list.map((st) => ({ value: st, label: st }))
  }, [])

  const statusOptions = useMemo(() => {
    const queueStatuses = queueRows.map((q) => q.status)
    const reasonStatuses = providers.flatMap((p) => p.reasons.map((r) => r.status))
    const list = Array.from(new Set([...queueStatuses, ...reasonStatuses])).filter(Boolean)
    return list.map((s) => ({ value: s, label: s }))
  }, [])

  const fwaOptions = useMemo(() => {
    const list = Array.from(
      new Set(
        providers.flatMap((p) => p.reasons.map((r) => r.source)).filter(Boolean)
      )
    )
    return list.map((s) => ({ value: s, label: s }))
  }, [])

  // Filter providers with AND logic
  const filteredProviders = useMemo(() => {
    return providers.filter((p) => {
      // Search: Provider name, NPI, specialty, location/state
      if (searchTerm.trim()) {
        const query = searchTerm.toLowerCase().trim()
        const matchName = p.name.toLowerCase().includes(query)
        const matchNpi = p.npi.includes(query)
        const matchSpec = p.specialty.toLowerCase().includes(query)
        const matchLoc = p.location.toLowerCase().includes(query)
        if (!matchName && !matchNpi && !matchSpec && !matchLoc) return false
      }

      // Risk Level
      if (!matchRiskLevel(p.riskScore, riskFilter)) return false

      // Specialty
      if (specialtyFilter !== 'all' && p.specialty !== specialtyFilter) return false

      // State
      if (stateFilter !== 'all' && getStateFromLocation(p.location) !== stateFilter) return false

      // Potential Leakage
      if (!matchLeakageRange(p.potentialExcessPayment, leakageFilter)) return false

      // Status
      if (statusFilter !== 'all') {
        const queueMatch = queueRows.find((q) => q.providerId === p.id)
        const hasQueueStatus = queueMatch && queueMatch.status === statusFilter
        const hasReasonStatus = p.reasons.some((r) => r.status === statusFilter)
        if (!hasQueueStatus && !hasReasonStatus) return false
      }

      // FWA Type
      if (fwaFilter !== 'all') {
        const hasSource = p.reasons.some((r) => r.source === fwaFilter)
        const hasReasonText = p.reasons.some((r) => r.reason.toLowerCase().includes(fwaFilter.toLowerCase()))
        if (!hasSource && !hasReasonText) return false
      }

      return true
    })
  }, [searchTerm, riskFilter, specialtyFilter, stateFilter, statusFilter, leakageFilter, fwaFilter])

  // Active provider: either the one in URL param or first filtered provider
  const activeProviderId = providerId || (filteredProviders.length > 0 ? filteredProviders[0].id : 'provider-chen')
  const provider = providers.find((entry) => entry.id === activeProviderId) || providers[0]
  const isSelectedProviderInFiltered = filteredProviders.some((p) => p.id === activeProviderId)

  // Active filter chips
  const activeChips: ActiveFilterItem[] = useMemo(() => {
    const chips: ActiveFilterItem[] = []
    if (searchTerm.trim()) {
      chips.push({
        id: 'search',
        label: 'Search',
        value: searchTerm,
        displayLabel: `Search: "${searchTerm}"`,
        onRemove: () => setSearchTerm(''),
      })
    }
    if (riskFilter !== 'all') {
      chips.push({
        id: 'risk',
        label: 'Risk',
        value: riskFilter,
        displayLabel: `Risk: ${riskFilter.charAt(0).toUpperCase() + riskFilter.slice(1)}`,
        onRemove: () => setRiskFilter('all'),
      })
    }
    if (specialtyFilter !== 'all') {
      chips.push({
        id: 'specialty',
        label: 'Specialty',
        value: specialtyFilter,
        displayLabel: `Specialty: ${specialtyFilter}`,
        onRemove: () => setSpecialtyFilter('all'),
      })
    }
    if (stateFilter !== 'all') {
      chips.push({
        id: 'state',
        label: 'State',
        value: stateFilter,
        displayLabel: `State: ${stateFilter}`,
        onRemove: () => setStateFilter('all'),
      })
    }
    if (statusFilter !== 'all') {
      chips.push({
        id: 'status',
        label: 'Status',
        value: statusFilter,
        displayLabel: `Status: ${statusFilter}`,
        onRemove: () => setStatusFilter('all'),
      })
    }
    if (leakageFilter !== 'all') {
      const labelMap: Record<string, string> = {
        under50k: '< ₹50K',
        '50k-100k': '₹50K - ₹100K',
        '100k-250k': '₹100K - ₹250K',
        over250k: '> ₹250K',
      }
      chips.push({
        id: 'leakage',
        label: 'Leakage',
        value: leakageFilter,
        displayLabel: `Leakage: ${labelMap[leakageFilter] || leakageFilter}`,
        onRemove: () => setLeakageFilter('all'),
      })
    }
    if (fwaFilter !== 'all') {
      chips.push({
        id: 'fwa',
        label: 'FWA',
        value: fwaFilter,
        displayLabel: `FWA: ${fwaFilter}`,
        onRemove: () => setFwaFilter('all'),
      })
    }
    return chips
  }, [searchTerm, riskFilter, specialtyFilter, stateFilter, statusFilter, leakageFilter, fwaFilter])

  const clearAllFilters = () => {
    setSearchTerm('')
    setRiskFilter('all')
    setSpecialtyFilter('all')
    setStateFilter('all')
    setStatusFilter('all')
    setLeakageFilter('all')
    setFwaFilter('all')
  }

  const filterConfigs: FilterConfig[] = [
    {
      id: 'risk',
      label: 'Risk Level',
      value: riskFilter,
      options: [
        { value: 'critical', label: 'Critical' },
        { value: 'high', label: 'High' },
        { value: 'medium', label: 'Medium' },
        { value: 'low', label: 'Low' },
      ],
      onChange: setRiskFilter,
    },
    {
      id: 'fwa',
      label: 'FWA Type',
      value: fwaFilter,
      options: fwaOptions,
      onChange: setFwaFilter,
    },
    {
      id: 'specialty',
      label: 'Specialty',
      value: specialtyFilter,
      options: specialtyOptions,
      onChange: setSpecialtyFilter,
    },
    {
      id: 'state',
      label: 'State',
      value: stateFilter,
      options: stateOptions,
      onChange: setStateFilter,
    },
    {
      id: 'status',
      label: 'Status',
      value: statusFilter,
      options: statusOptions,
      onChange: setStatusFilter,
    },
    {
      id: 'leakage',
      label: 'Potential Leakage',
      value: leakageFilter,
      options: [
        { value: 'under50k', label: '< ₹50K' },
        { value: '50k-100k', label: '₹50K - ₹100K' },
        { value: '100k-250k', label: '₹100K - ₹250K' },
        { value: 'over250k', label: '> ₹250K' },
      ],
      onChange: setLeakageFilter,
    },
  ]

  return (
    <div className="page-grid">
      {/* Top Search & Filter Toolbar */}
      <section className="panel panel-12">
        <div className="panel-header stack-header">
          <div>
            <p className="eyebrow">Provider Search &amp; Triage</p>
            <h3>Provider Investigations</h3>
          </div>
        </div>

        <SearchFilterToolbar
          searchPlaceholder="Search provider name, NPI, specialty..."
          searchValue={searchTerm}
          onSearchChange={setSearchTerm}
          filters={filterConfigs}
        />

        <ActiveFilterChips chips={activeChips} onClearAll={clearAllFilters} />

        <div className="result-count-bar">
          <span className="result-count-text">
            {filteredProviders.length} {filteredProviders.length === 1 ? 'provider' : 'providers'} found
          </span>
        </div>

        {filteredProviders.length === 0 ? (
          <EmptyFilterState
            title="No matching records found"
            message="No providers matched your current search query and filter criteria. Adjust your filters or click Clear Filters below."
            onClearFilters={clearAllFilters}
          />
        ) : (
          <table className="table data-table">
            <thead>
              <tr>
                <th>Provider Name</th>
                <th>NPI</th>
                <th>Specialty</th>
                <th>Location</th>
                <th>Risk Score</th>
                <th>Potential Leakage</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {filteredProviders.map((p) => {
                const isCurrentActive = p.id === activeProviderId
                return (
                  <tr key={p.id} className={isCurrentActive ? 'selected-table-row' : ''}>
                    <td>
                      <strong>{p.name}</strong>
                      {isCurrentActive && <span className="active-row-badge">Active View</span>}
                    </td>
                    <td>{p.npi}</td>
                    <td>{p.specialty}</td>
                    <td>{p.location}</td>
                    <td><RiskBadge score={p.riskScore} /></td>
                    <td>{formatCurrency(p.potentialExcessPayment)}</td>
                    <td>
                      <button
                        className="secondary-action"
                        type="button"
                        onClick={() => navigate(`/providers/₹{p.id}`)}
                      >
                        {isCurrentActive ? 'Viewing' : 'Open Profile'}
                      </button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </section>

      {/* Selected Provider Details (Preserved Existing UI & Logic) */}
      {isSelectedProviderInFiltered && provider && (
        <>
          <section className="panel panel-12 summary-panel">
            <div className="panel-header provider-header">
              <div>
                <p className="eyebrow">Provider Summary</p>
                <h3>{provider.name}</h3>
              </div>
              <RiskBadge score={provider.riskScore} />
            </div>
            <div className="summary-grid">
              <div><span>NPI</span><strong>{provider.npi}</strong></div>
              <div><span>Specialty</span><strong>{provider.specialty}</strong></div>
              <div><span>Location</span><strong>{provider.location}</strong></div>
              <div><span>Total Claims</span><strong>{formatNumber(provider.totalClaims)}</strong></div>
              <div><span>Total Paid</span><strong>{formatCurrency(provider.totalPaid)}</strong></div>
              <div><span>Risk Score</span><strong>{provider.riskScore}</strong></div>
            </div>
          </section>

          <section className="panel panel-12 evidence-panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Explainability</p>
                <h3>Why Was This Provider Flagged?</h3>
              </div>
            </div>
            <div className="reason-card">
              <div className="reason-row reason-head">
                <span>Reason</span>
                <span>Source</span>
                <span>Status</span>
              </div>
              {provider.reasons.map((entry) => (
                <div key={entry.reason} className="reason-row">
                  <span>{entry.reason}</span>
                  <span>{entry.source}</span>
                  <StatusBadge status={entry.status} />
                </div>
              ))}
            </div>
          </section>

          <section className="panel panel-12">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Peer Benchmarking</p>
                <h3>Peer Comparison</h3>
              </div>
            </div>
            <div className="peer-meta-row">
              <span>Specialty: {provider.peerGroup.specialty}</span>
              <span>Region: {provider.peerGroup.region}</span>
              <span>Peer Count: {provider.peerGroup.peerCount}</span>
              <span>Threshold: {provider.peerGroup.threshold}</span>
              <span>Confidence: {provider.peerGroup.confidence}</span>
            </div>
            <table className="table data-table">
              <thead>
                <tr>
                  <th>Metric</th>
                  <th>Provider</th>
                  <th>Peer Median</th>
                </tr>
              </thead>
              <tbody>
                {provider.peerMetrics.map((row) => (
                  <tr key={row.metric}>
                    <td>{row.metric}</td>
                    <td>{row.providerDisplay}</td>
                    <td>{row.peerDisplay}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          <section className="panel panel-6 financial-noise">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Financial Impact</p>
                <h3>Potential Excess Payment</h3>
              </div>
            </div>
            <div className="impact-value">{formatCurrency(provider.potentialExcessPayment)}</div>
            <div className="source-tag">Source: POTENTIAL_EXCESS_AMOUNT</div>
          </section>

          <section className="panel panel-6 score-panel">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Scoring</p>
                <h3>This Provider's Score Breakdown</h3>
              </div>
            </div>
            <div className="score-stack">
              {provider.scoreBreakdown.map((item) => (
                <div key={item.label} className={`score-row ₹{item.label === 'Final Score' ? 'final-row' : ''}`}>
                  <span>{item.label}</span>
                  <strong>{item.value}</strong>
                </div>
              ))}
            </div>
          </section>

          <section className="panel panel-12">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Associated Claims</p>
                <h3>Claim Portfolio</h3>
              </div>
            </div>
            <table className="table data-table">
              <thead>
                <tr>
                  <th>Claim ID</th>
                  <th>Service Date</th>
                  <th>Amount</th>
                  <th>Risk Score</th>
                  <th>Action</th>
                </tr>
              </thead>
              <tbody>
                {claims
                  .filter((claim) => claim.providerId === provider.id)
                  .map((claim) => (
                    <tr key={claim.id}>
                      <td>{claim.id}</td>
                      <td>{claim.serviceDate}</td>
                      <td>{formatCurrency(claim.claimAmount)}</td>
                      <td><RiskBadge score={claim.riskScore} /></td>
                      <td>
                        <button className="secondary-action" type="button" onClick={() => navigate(`/claims/₹{claim.id}`)}>
                          Open Claim
                        </button>
                      </td>
                    </tr>
                  ))}
              </tbody>
            </table>
          </section>
        </>
      )}
    </div>
  )
}

function ClaimPage() {
  const { claimId } = useParams()
  const navigate = useNavigate()

  const [searchTerm, setSearchTerm] = useState('')
  const [riskFilter, setRiskFilter] = useState('all')
  const [providerFilter, setProviderFilter] = useState('all')
  const [amountFilter, setAmountFilter] = useState('all')
  const [anomalyFilter, setAnomalyFilter] = useState('all')
  const [dateFilter, setDateFilter] = useState('all')
  const [fwaFilter, setFwaFilter] = useState('all')

  // Dynamic filter options from existing claims
  const providerOptions = useMemo(() => {
    const list = Array.from(new Set(claims.map((c) => c.providerName))).filter(Boolean)
    return list.map((p) => ({ value: p, label: p }))
  }, [])

  const anomalyOptions = useMemo(() => {
    const list = Array.from(
      new Set(
        claims.flatMap((c) => c.reasons.map((r) => r.source)).filter(Boolean)
      )
    )
    return list.map((s) => ({ value: s, label: s }))
  }, [])

  const fwaOptions = useMemo(() => {
    const list = Array.from(
      new Set(
        claims.flatMap((c) => c.reasons.map((r) => r.reason)).filter(Boolean)
      )
    )
    return list.map((r) => ({ value: r, label: r }))
  }, [])

  // Filter claims with AND logic
  const filteredClaims = useMemo(() => {
    return claims.filter((claim) => {
      // Search: Claim ID, Provider, Beneficiary, HCPCS
      if (searchTerm.trim()) {
        const query = searchTerm.toLowerCase().trim()
        const matchId = claim.id.toLowerCase().includes(query)
        const matchProv = claim.providerName.toLowerCase().includes(query)
        const matchBen = claim.beneficiaryId.toLowerCase().includes(query)
        const matchHcpcs = claim.procedureBreakdown.some(
          (p) => p.hcpcs.toLowerCase().includes(query) || p.description.toLowerCase().includes(query)
        )
        if (!matchId && !matchProv && !matchBen && !matchHcpcs) return false
      }

      // Risk Level
      if (!matchRiskLevel(claim.riskScore, riskFilter)) return false

      // Provider
      if (providerFilter !== 'all' && claim.providerName !== providerFilter) return false

      // Claim Amount
      if (!matchClaimAmountRange(claim.claimAmount, amountFilter)) return false

      // Anomaly Type
      if (anomalyFilter !== 'all' && !claim.reasons.some((r) => r.source === anomalyFilter)) return false

      // FWA Type
      if (fwaFilter !== 'all' && !claim.reasons.some((r) => r.reason === fwaFilter || r.source === fwaFilter)) return false

      // Service Date
      if (!matchDateRange(claim.serviceDate, dateFilter)) return false

      return true
    })
  }, [searchTerm, riskFilter, providerFilter, amountFilter, anomalyFilter, fwaFilter, dateFilter])

  // Active claim: either from URL param or first filtered claim
  const activeClaimId = claimId || (filteredClaims.length > 0 ? filteredClaims[0].id : 'CLM-824791')
  const claim = claims.find((entry) => entry.id === activeClaimId) || claims[0]
  const isSelectedClaimInFiltered = filteredClaims.some((c) => c.id === activeClaimId)

  // Active filter chips
  const activeChips: ActiveFilterItem[] = useMemo(() => {
    const chips: ActiveFilterItem[] = []
    if (searchTerm.trim()) {
      chips.push({
        id: 'search',
        label: 'Search',
        value: searchTerm,
        displayLabel: `Search: "₹{searchTerm}"`,
        onRemove: () => setSearchTerm(''),
      })
    }
    if (riskFilter !== 'all') {
      chips.push({
        id: 'risk',
        label: 'Risk',
        value: riskFilter,
        displayLabel: `Risk: ₹{riskFilter.charAt(0).toUpperCase() + riskFilter.slice(1)}`,
        onRemove: () => setRiskFilter('all'),
      })
    }
    if (providerFilter !== 'all') {
      chips.push({
        id: 'provider',
        label: 'Provider',
        value: providerFilter,
        displayLabel: `Provider: ₹{providerFilter}`,
        onRemove: () => setProviderFilter('all'),
      })
    }
    if (amountFilter !== 'all') {
      const labelMap: Record<string, string> = {
        under1k: '< ₹1K',
        '1k-5k': '₹1K - ₹5K',
        '5k-10k': '₹5K - ₹10K',
        over10k: '> ₹10K',
      }
      chips.push({
        id: 'amount',
        label: 'Amount',
        value: amountFilter,
        displayLabel: `Amount: ₹{labelMap[amountFilter] || amountFilter}`,
        onRemove: () => setAmountFilter('all'),
      })
    }
    if (anomalyFilter !== 'all') {
      chips.push({
        id: 'anomaly',
        label: 'Anomaly',
        value: anomalyFilter,
        displayLabel: `Anomaly: ₹{anomalyFilter}`,
        onRemove: () => setAnomalyFilter('all'),
      })
    }
    if (fwaFilter !== 'all') {
      chips.push({
        id: 'fwa',
        label: 'FWA',
        value: fwaFilter,
        displayLabel: `FWA: ₹{fwaFilter}`,
        onRemove: () => setFwaFilter('all'),
      })
    }
    if (dateFilter !== 'all') {
      const labelMap: Record<string, string> = {
        last7: 'Last 7 Days',
        last30: 'Last 30 Days',
        last90: 'Last 90 Days',
      }
      chips.push({
        id: 'date',
        label: 'Date',
        value: dateFilter,
        displayLabel: `Date: ₹{labelMap[dateFilter] || dateFilter}`,
        onRemove: () => setDateFilter('all'),
      })
    }
    return chips
  }, [searchTerm, riskFilter, providerFilter, amountFilter, anomalyFilter, fwaFilter, dateFilter])

  const clearAllFilters = () => {
    setSearchTerm('')
    setRiskFilter('all')
    setProviderFilter('all')
    setAmountFilter('all')
    setAnomalyFilter('all')
    setFwaFilter('all')
    setDateFilter('all')
  }

  const filterConfigs: FilterConfig[] = [
    {
      id: 'risk',
      label: 'Risk Level',
      value: riskFilter,
      options: [
        { value: 'critical', label: 'Critical' },
        { value: 'high', label: 'High' },
        { value: 'medium', label: 'Medium' },
        { value: 'low', label: 'Low' },
      ],
      onChange: setRiskFilter,
    },
    {
      id: 'fwa',
      label: 'FWA Type',
      value: fwaFilter,
      options: fwaOptions,
      onChange: setFwaFilter,
    },
    {
      id: 'provider',
      label: 'Provider',
      value: providerFilter,
      options: providerOptions,
      onChange: setProviderFilter,
    },
    {
      id: 'amount',
      label: 'Claim Amount',
      value: amountFilter,
      options: [
        { value: 'under1k', label: '< ₹1K' },
        { value: '1k-5k', label: '₹1K - ₹5K' },
        { value: '5k-10k', label: '₹5K - ₹10K' },
        { value: 'over10k', label: '> ₹10K' },
      ],
      onChange: setAmountFilter,
    },
    {
      id: 'anomaly',
      label: 'Anomaly Type',
      value: anomalyFilter,
      options: anomalyOptions,
      onChange: setAnomalyFilter,
    },
    {
      id: 'date',
      label: 'Service Date',
      value: dateFilter,
      options: [
        { value: 'last7', label: 'Last 7 Days' },
        { value: 'last30', label: 'Last 30 Days' },
        { value: 'last90', label: 'Last 90 Days' },
      ],
      onChange: setDateFilter,
    },
  ]

  return (
    <div className="page-grid">
      {/* Top Search & Filter Toolbar */}
      <section className="panel panel-12">
        <div className="panel-header stack-header">
          <div>
            <p className="eyebrow">Claim Search &amp; Filtering</p>
            <h3>Claim Investigations</h3>
          </div>
        </div>

        <SearchFilterToolbar
          searchPlaceholder="Search claim ID, provider, beneficiary..."
          searchValue={searchTerm}
          onSearchChange={setSearchTerm}
          filters={filterConfigs}
        />

        <ActiveFilterChips chips={activeChips} onClearAll={clearAllFilters} />

        <div className="result-count-bar">
          <span className="result-count-text">
            {filteredClaims.length} {filteredClaims.length === 1 ? 'claim' : 'claims'} found
          </span>
        </div>

        {filteredClaims.length === 0 ? (
          <EmptyFilterState
            title="No matching records found"
            message="No claims matched your search query and filter criteria. Adjust your filters or click Clear Filters below."
            onClearFilters={clearAllFilters}
          />
        ) : (
          <table className="table data-table">
            <thead>
              <tr>
                <th>Claim ID</th>
                <th>Provider Name</th>
                <th>Beneficiary ID</th>
                <th>Service Date</th>
                <th>Claim Amount</th>
                <th>Paid Amount</th>
                <th>Risk Score</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {filteredClaims.map((c) => {
                const isCurrentActive = c.id === activeClaimId
                return (
                  <tr key={c.id} className={isCurrentActive ? 'selected-table-row' : ''}>
                    <td>
                      <strong>{c.id}</strong>
                      {isCurrentActive && <span className="active-row-badge">Active View</span>}
                    </td>
                    <td>{c.providerName}</td>
                    <td>{c.beneficiaryId}</td>
                    <td>{c.serviceDate}</td>
                    <td>{formatCurrency(c.claimAmount)}</td>
                    <td>{formatCurrency(c.paidAmount)}</td>
                    <td><RiskBadge score={c.riskScore} /></td>
                    <td>
                      <button
                        className="secondary-action"
                        type="button"
                        onClick={() => navigate(`/claims/₹{c.id}`)}
                      >
                        {isCurrentActive ? 'Viewing' : 'Open Claim'}
                      </button>
                    </td>
                  </tr>
                )
              })}
            </tbody>
          </table>
        )}
      </section>

      {/* Selected Claim Details (Preserved Existing UI & Logic) */}
      {isSelectedClaimInFiltered && claim && (
        <>
          <section className="panel panel-12 summary-panel">
            <div className="panel-header provider-header">
              <div>
                <p className="eyebrow">Claim Investigation Profile</p>
                <h3>{claim.id}</h3>
              </div>
              <RiskBadge score={claim.riskScore} />
            </div>
            <div className="summary-grid">
              <div><span>Provider</span><strong>{claim.providerName}</strong></div>
              <div><span>Beneficiary</span><strong>{claim.beneficiaryId}</strong></div>
              <div><span>Claim Amount</span><strong>{formatCurrency(claim.claimAmount)}</strong></div>
              <div><span>Paid Amount</span><strong>{formatCurrency(claim.paidAmount)}</strong></div>
              <div><span>Service Date</span><strong>{claim.serviceDate}</strong></div>
              <div><span>Risk Score</span><strong>{claim.riskScore}</strong></div>
            </div>
          </section>

          <section className="panel panel-6">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Explainability</p>
                <h3>Why Was This Claim Flagged?</h3>
              </div>
            </div>
            <div className="reason-card">
              {claim.reasons.map((reason) => (
                <div key={reason.reason} className="reason-row">
                  <span>{reason.reason}</span>
                  <span>{reason.source}</span>
                  <StatusBadge status={reason.status} />
                </div>
              ))}
            </div>
          </section>

          <section className="panel panel-6">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Procedure Breakdown</p>
                <h3>HCPCS Detail</h3>
              </div>
            </div>
            <table className="table data-table compact-table">
              <thead>
                <tr>
                  <th>HCPCS</th>
                  <th>Description</th>
                  <th>Units</th>
                  <th>Paid Amount</th>
                  <th>Flag</th>
                </tr>
              </thead>
              <tbody>
                {claim.procedureBreakdown.map((item) => (
                  <tr key={item.hcpcs}>
                    <td>{item.hcpcs}</td>
                    <td>{item.description}</td>
                    <td>{item.units}</td>
                    <td>{formatCurrency(item.paidAmount)}</td>
                    <td>{item.flag}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </section>

          <section className="panel panel-6 financial-noise">
            <div className="panel-header">
              <div>
                <p className="eyebrow">Financial Impact</p>
                <h3>Potential Excess Amount</h3>
              </div>
            </div>
            <div className="impact-value">{formatCurrency(claim.potentialExcessPayment)}</div>
          </section>
        </>
      )}
    </div>
  )
}

function AssistantPage() {
  return (
    <div className="page-grid">
      <section className="panel panel-6">
        <div className="panel-header">
          <div>
            <p className="eyebrow">AI Investigation Support</p>
            <h3>Suggested Prompts</h3>
          </div>
        </div>
        <div className="prompt-list">
          {[
            'Why is this provider high risk?',
            'Compare provider with peers.',
            'Summarize investigation.',
            'Generate audit report.',
          ].map((prompt) => (
            <button className="prompt-button" type="button" key={prompt}>{prompt}</button>
          ))}
        </div>
      </section>

      <section className="panel panel-6 assistant-panel">
        <div className="panel-header">
          <div>
            <p className="eyebrow">AI Response</p>
            <h3>Risk Summary</h3>
          </div>
          <RiskBadge score={92} />
        </div>
        <div className="response-stack">
          <div>
            <h4>Evidence</h4>
            <p>Claims materially exceed peer volume and payments, with high-procedure frequency concentrated in a narrow billing window.</p>
          </div>
          <div>
            <h4>Peer Comparison</h4>
            <p>Average paid amount is elevated against the peer median and supported by an increased claims volume.</p>
          </div>
          <div>
            <h4>Financial Impact</h4>
            <p>Potential excess payment is estimated from provider-level leakage and claim-level outlier analysis.</p>
          </div>
          <div>
            <h4>Recommendation</h4>
            <p>Prioritize provider review and targeted claim audit for repeat high-intensity procedure bundles.</p>
          </div>
        </div>
      </section>
    </div>
  )
}

function ReportsPage() {
  return (
    <div className="page-grid">
      <section className="panel panel-6">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Templates</p>
            <h3>Investigation Reports</h3>
          </div>
        </div>
        <ul className="report-list">
          {reportTemplates.map((report) => (
            <li key={report.id} className={report.id === 'provider-investigation-report' ? 'active' : ''}>
              {report.name}
            </li>
          ))}
        </ul>
      </section>

      <section className="panel panel-6">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Export</p>
            <h3>Download Package</h3>
          </div>
        </div>
        <div className="export-grid">
          {['PDF', 'Excel', 'CSV'].map((type) => (
            <button key={type} className="export-button" type="button">
              {type}
            </button>
          ))}
        </div>
      </section>
    </div>
  )
}

function ModelGovernancePage() {
  const qualityRows = [
    {
      label: 'Total Records Processed',
      value: formatNumber(dataQualityMetrics.totalRecordsProcessed),
      percent: 100,
    },
    {
      label: 'Missing Values',
      value: formatNumber(dataQualityMetrics.missingValues),
      percent: 0.9,
    },
    {
      label: 'Duplicate Records',
      value: formatNumber(dataQualityMetrics.duplicateRecords),
      percent: 0.2,
    },
    {
      label: 'Failed Validations',
      value: formatNumber(dataQualityMetrics.failedValidations),
      percent: 0.1,
    },
    {
      label: 'Data Quality Score',
      value: `₹{dataQualityMetrics.dataQualityScore}%`,
      percent: dataQualityMetrics.dataQualityScore,
    },
  ]

  return (
    <div className="page-grid governance-page">
      <section className="panel panel-12 governance-title">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Model Governance</p>
            <h3>Model Governance</h3>
            <p className="governance-subtitle">
              Transparency and oversight for payment integrity scoring, model operations, and data quality.
            </p>
          </div>
        </div>
      </section>

      <section className="panel panel-12">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Risk Scoring Framework</p>
            <h3>Risk Tier Definitions</h3>
          </div>
        </div>
        <table className="table data-table">
          <thead>
            <tr>
              <th>Tier</th>
              <th>Score Range</th>
              <th>Color</th>
            </tr>
          </thead>
          <tbody>
            {riskTierDefinitions.map((row) => (
              <tr key={row.tier}>
                <td>{row.tier}</td>
                <td>{row.range}</td>
                <td>{row.color}</td>
              </tr>
            ))}
          </tbody>
        </table>
        <p className="governance-note">
          These thresholds are applied consistently across dashboard, investigation queue, provider profiles,
          claim investigations, and exported reports.
        </p>
      </section>

      <section className="panel panel-12">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Active Models</p>
            <h3>Model Inventory</h3>
          </div>
        </div>
        <table className="table data-table">
          <thead>
            <tr>
              <th>Model</th>
              <th>Purpose</th>
              <th>Weight</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {modelInventory.map((row) => (
              <tr key={row.model}>
                <td>{row.model}</td>
                <td>{row.purpose}</td>
                <td>{row.weight}</td>
                <td><GovernanceStatusBadge status={row.status} /></td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="panel panel-6">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Model Weights</p>
            <h3>Risk Score Composition</h3>
            <p className="gov-weight-label">GLOBAL MODEL WEIGHTS</p>
          </div>
        </div>
        <div className="donut-wrap">
          <ResponsiveContainer width="100%" height={220}>
            <PieChart>
              <Pie data={globalModelWeights} dataKey="value" nameKey="label" innerRadius={56} outerRadius={82}>
                {globalModelWeights.map((entry) => (
                  <Cell key={entry.label} fill={entry.color} />
                ))}
              </Pie>
              <Tooltip formatter={(value) => [`₹{Number(value ?? 0)}%`, 'Weight']} />
            </PieChart>
          </ResponsiveContainer>
          <div className="legend-list">
            {globalModelWeights.map((row) => (
              <div key={row.label} className="legend-item">
                <span className="color-dot" style={{ background: row.color }} />
                <span>{row.label}</span>
                <strong>{row.value}%</strong>
              </div>
            ))}
          </div>
        </div>
        <p className="governance-formula">
          Final Risk Score = 35% Peer Analysis + 25% Isolation Forest + 20% LOF + 20% Rule Engine
        </p>
      </section>

      <section className="panel panel-6">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Validation &amp; Business Impact</p>
            <h3>Business Validation Metrics</h3>
          </div>
        </div>
        <div className="summary-grid governance-summary-grid">
          <div><span>Flagged Providers</span><strong>{formatNumber(businessValidationMetrics.flaggedProviders)}</strong></div>
          <div><span>LEIE Matches</span><strong>{formatNumber(businessValidationMetrics.leieMatches)}</strong></div>
          <div><span>LEIE Match Rate</span><strong>{businessValidationMetrics.leieMatchRate}%</strong></div>
          <div><span>Potential Excess Amount Detected</span><strong>{formatCurrency(businessValidationMetrics.potentialExcessAmountDetected)}</strong></div>
          <div><span>Claims Reviewed</span><strong>{formatNumber(businessValidationMetrics.claimsReviewed)}</strong></div>
        </div>
        <p className="governance-note">
          LEIE overlap is used for business validation only and is not a formal precision, recall, or accuracy metric.
        </p>
      </section>

      <section className="panel panel-12">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Data Sources</p>
            <h3>Connected Data Sources</h3>
          </div>
        </div>
        <div className="source-grid">
          {governanceDataSources.map((source) => (
            <article key={source.name} className="source-card">
              <h4>{source.name}</h4>
              <p>Records: {formatNumber(source.records)}</p>
              <p>Last Refresh: {source.refreshDate}</p>
              <span className="gov-badge connected">{source.status}</span>
            </article>
          ))}
        </div>
      </section>

      <section className="panel panel-12">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Data Quality Monitoring</p>
            <h3>Data Quality Overview</h3>
          </div>
        </div>
        <div className="quality-stack">
          {qualityRows.map((metric) => (
            <div key={metric.label} className="quality-row">
              <div className="quality-head">
                <span>{metric.label}</span>
                <strong>{metric.value}</strong>
              </div>
              <div className="quality-track">
                <div className="quality-fill" style={{ width: `₹{Math.min(metric.percent, 100)}%` }} />
              </div>
            </div>
          ))}
        </div>
      </section>

      <section className="panel panel-6">
        <div className="panel-header">
          <div>
            <p className="eyebrow">System Health</p>
            <h3>Operational Status</h3>
          </div>
        </div>
        <div className="health-stack">
          {systemHealth.map((row) => (
            <div key={row.component} className="health-row">
              <span>{row.component}</span>
              <strong className="health-pill">
                <span className="health-dot" />
                {row.status}
              </strong>
            </div>
          ))}
        </div>
      </section>

      <section className="panel panel-6">
        <div className="panel-header">
          <div>
            <p className="eyebrow">Audit Log</p>
            <h3>Recent System Activity</h3>
          </div>
        </div>
        <table className="table data-table compact-table">
          <thead>
            <tr>
              <th>Timestamp</th>
              <th>Event</th>
              <th>User</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {auditLogRows.map((row) => (
              <tr key={`₹{row.timestamp}-₹{row.event}`}>
                <td>{row.timestamp}</td>
                <td>{row.event}</td>
                <td>{row.user}</td>
                <td><span className="gov-badge connected">{row.status}</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  )
}

function App() {
  return (
    <BrowserRouter>
      <AppShell />
    </BrowserRouter>
  )
}

export default App
