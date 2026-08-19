import React, { useState, useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import { 
  Send, 
  Bot, 
  User, 
  HeartPulse, 
  Compass,
  Info
} from 'lucide-react';
import { getApiUrl } from '../config';
import './AIAssistant.css';

interface Message {
  sender: 'user' | 'ai';
  text: string;
  timestamp: Date;
}

interface ProviderSummary {
  provider_id: string;
  risk_score: number;
  risk_level: string;
  total_reimbursement: number;
  total_claims: number;
}

const AIAssistant: React.FC = () => {
  const location = useLocation();
  const state = location.state as { providerId?: string } | null;
  
  const [providerId, setProviderId] = useState(state?.providerId || localStorage.getItem('mediclaim_selected_provider') || '');
  const [providersList, setProvidersList] = useState<string[]>([]);
  const [activeProviderStats, setActiveProviderStats] = useState<ProviderSummary | null>(null);
  
  const [query, setQuery] = useState('');
  const [messages, setMessages] = useState<Message[]>([
    {
      sender: 'ai',
      text: "Hello! I am your AI Clinical Audit Assistant. Select a Provider ID to begin. I can answer complex inquiries, dissect statistical deviations, and report peer comparison summaries grounded in the clinical claims database.",
      timestamp: new Date()
    }
  ]);
  const [loading, setLoading] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  // Load suspicious providers to populate select box
  useEffect(() => {
    const fetchSuspiciousList = async () => {
      try {
        const res = await fetch(`${getApiUrl()}/api/v1/dashboard`);
        if (res.ok) {
          const data = await res.json();
          const ids = data.top_suspicious.map((p: any) => p.provider_id);
          setProvidersList(ids);
          // If we had a pre-selected ID from router, load it
          if (state?.providerId) {
            loadProviderStats(state.providerId);
          } else if (ids.length > 0) {
            setProviderId(ids[0]);
            loadProviderStats(ids[0]);
          }
        }
      } catch (e) {
        console.error(e);
      }
    };
    fetchSuspiciousList();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages, loading]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const loadProviderStats = async (id: string) => {
    try {
      const res = await fetch(`${getApiUrl()}/api/v1/providers/${id}`);
      if (res.ok) {
        const data = await res.json();
        setActiveProviderStats({
          provider_id: data.profile.provider_id,
          risk_score: data.profile.risk_score,
          risk_level: data.profile.risk_level,
          total_reimbursement: data.profile.total_reimbursement,
          total_claims: data.profile.total_claims
        });
      }
    } catch (e) {
      console.error(e);
    }
  };

  const handleProviderChange = (id: string) => {
    setProviderId(id);
    localStorage.setItem('mediclaim_selected_provider', id);
    window.dispatchEvent(new Event('mediclaim-provider-selected'));
    loadProviderStats(id);
    setMessages([
      {
        sender: 'ai',
        text: `AI Context loaded for Provider **${id}**. Ask me questions about their anomalies, statistical deviations, or how they compare to the local peer group.`,
        timestamp: new Date()
      }
    ]);
  };

  const handleSendMessage = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!query.trim() || !providerId) return;

    const userMsg = query;
    setMessages(prev => [...prev, { sender: 'user', text: userMsg, timestamp: new Date() }]);
    setQuery('');
    setLoading(true);

    try {
      const res = await fetch(`${getApiUrl()}/api/v1/ai-assistant/query`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          provider_id: providerId,
          query: userMsg
        })
      });

      if (res.ok) {
        const json = await res.json();
        setMessages(prev => [...prev, { sender: 'ai', text: json.response, timestamp: new Date() }]);
      } else {
        setMessages(prev => [...prev, { sender: 'ai', text: "Error: Could not retrieve a response from the API.", timestamp: new Date() }]);
      }
    } catch (err) {
      setMessages(prev => [...prev, { sender: 'ai', text: "Error: Failed to connect to server.", timestamp: new Date() }]);
    } finally {
      setLoading(false);
    }
  };

  const suggestQuery = (q: string) => {
    setQuery(q);
  };

  // Basic markdown bold & table parsing helper for mock formatting
  const formatText = (text: string) => {
    // Split by lines to parse lists and markdown tables
    const lines = text.split('\n');
    let inTable = false;
    let tableHeaders: string[] = [];
    let tableRows: string[][] = [];

    return lines.map((line, i) => {
      // Bold text replacements
      let formattedLine = line.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
      formattedLine = formattedLine.replace(/\*(.*?)\*/g, '<em>$1</em>');

      // Detect table rows e.g. | Col 1 | Col 2 |
      if (line.trim().startsWith('|')) {
        const cells = line.split('|').map(c => c.trim()).filter((_, idx, arr) => idx > 0 && idx < arr.length - 1);
        if (line.includes('---')) {
          // Separator row, ignore
          return null;
        }
        if (!inTable) {
          inTable = true;
          tableHeaders = cells;
          return null;
        } else {
          tableRows.push(cells);
          
          // Check if next line is not a table row to render the accumulated table
          const nextLine = lines[i + 1];
          if (!nextLine || !nextLine.trim().startsWith('|')) {
            inTable = false;
            const headers = tableHeaders;
            const rows = tableRows;
            tableHeaders = [];
            tableRows = [];
            return (
              <div className="table-container chat-table-container" key={`table-${i}`}>
                <table>
                  <thead>
                    <tr>
                      {headers.map((h, idx) => <th key={idx} dangerouslySetInnerHTML={{ __html: h }}></th>)}
                    </tr>
                  </thead>
                  <tbody>
                    {rows.map((row, rIdx) => (
                      <tr key={rIdx}>
                        {row.map((cell, cIdx) => <td key={cIdx} dangerouslySetInnerHTML={{ __html: cell }}></td>)}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            );
          }
          return null;
        }
      }

      // Detect bullet points
      if (line.trim().startsWith('- ')) {
        const content = formattedLine.trim().substring(2);
        return <li key={i} className="chat-bullet" dangerouslySetInnerHTML={{ __html: content }}></li>;
      }

      // Regular line
      if (formattedLine.trim() === '') return <br key={i} />;
      return <p key={i} dangerouslySetInnerHTML={{ __html: formattedLine }}></p>;
    }).filter(el => el !== null);
  };

  return (
    <div className="assistant-page animate-fade-in">
      <div className="page-header">
        <div>
          <h1 className="page-title">AI Clinical Audit Assistant</h1>
          <p className="page-subtitle">Ground clinical investigation queries using trained model parameters and peer statistical matrices.</p>
        </div>
      </div>

      <div className="assistant-grid">
        {/* Chat Area */}
        <div className="card chat-card-container">
          <div className="chat-header">
            <div className="provider-select-flex">
              <Bot size={20} className="bot-header-icon" />
              <span className="select-lbl">Target Case Context:</span>
              <select 
                value={providerId} 
                onChange={(e) => handleProviderChange(e.target.value)}
                className="provider-selector-assistant"
              >
                <option value="">Select Provider ID...</option>
                {providersList.map(id => (
                  <option key={id} value={id}>Provider {id}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="chat-messages-container">
            {messages.map((msg, index) => (
              <div className={`message-bubble-wrapper ${msg.sender}`} key={index}>
                <div className="message-avatar">
                  {msg.sender === 'ai' ? <Bot size={16} /> : <User size={16} />}
                </div>
                <div className="message-content">
                  {formatText(msg.text)}
                  <span className="msg-time">
                    {msg.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </span>
                </div>
              </div>
            ))}
            {loading && (
              <div className="message-bubble-wrapper ai">
                <div className="message-avatar">
                  <Bot size={16} className="spinning-bot" />
                </div>
                <div className="message-content loading-content">
                  <div className="typing-dots">
                    <span></span>
                    <span></span>
                    <span></span>
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Quick suggestions */}
          {providerId && (
            <div className="quick-suggestions-row">
              <Compass size={14} className="suggestion-icon" />
              <button className="suggestion-chip" onClick={() => suggestQuery("Why was this provider flagged?")}>
                Why was this provider flagged?
              </button>
              <button className="suggestion-chip" onClick={() => suggestQuery("Compare this provider to their peers.")}>
                Compare to peers
              </button>
              <button className="suggestion-chip" onClick={() => suggestQuery("Which machine learning models flagged them?")}>
                Explain ML findings
              </button>
            </div>
          )}

          {/* Input form */}
          <form onSubmit={handleSendMessage} className="chat-input-form">
            <input 
              type="text" 
              placeholder={providerId ? "Ask a clinical question about this provider..." : "Please select a provider first..."}
              value={query}
              onChange={(e) => setQuery(e.target.value)}
              disabled={!providerId || loading}
            />
            <button 
              type="submit" 
              className="btn btn-primary send-chat-btn"
              disabled={!providerId || !query.trim() || loading}
            >
              <Send size={16} />
            </button>
          </form>
        </div>

        {/* Live Context Sidebar */}
        <div className="card assistant-sidebar-card">
          <h3 className="sidebar-title">Grounded Context File</h3>
          
          {activeProviderStats ? (
            <div className="active-sidebar-details">
              <div className="active-sidebar-header">
                <HeartPulse size={24} className="sidebar-header-icon" />
                <div>
                  <h4>{activeProviderStats.provider_id}</h4>
                  <p className="sidebar-type">Case Metadata</p>
                </div>
              </div>

              <div className="sidebar-kpi-list">
                <div className="sidebar-kpi-item">
                  <span className="lbl">Composite Risk Score</span>
                  <span className={`val badge badge-${activeProviderStats.risk_level.toLowerCase()}`}>
                    {activeProviderStats.risk_score.toFixed(1)}
                  </span>
                </div>
                <div className="sidebar-kpi-item">
                  <span className="lbl">Risk Classification</span>
                  <span className="val bold text-uppercase">{activeProviderStats.risk_level}</span>
                </div>
                <div className="sidebar-kpi-item">
                  <span className="lbl">Covered Billing</span>
                  <span className="val bold">${activeProviderStats.total_reimbursement.toLocaleString(undefined, {maximumFractionDigits: 0})}</span>
                </div>
                <div className="sidebar-kpi-item">
                  <span className="lbl">Submitted Claims</span>
                  <span className="val bold">{activeProviderStats.total_claims} Claims</span>
                </div>
              </div>

              <div className="sidebar-instruction-box">
                <Info size={14} className="instruction-icon" />
                <p>All questions are parsed against pre-calculated state peer statistics and model thresholds for Provider {activeProviderStats.provider_id}.</p>
              </div>
            </div>
          ) : (
            <div className="empty-sidebar-context">
              <Compass size={32} className="empty-icon" />
              <p>Select a provider to load active metadata and context files.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AIAssistant;
