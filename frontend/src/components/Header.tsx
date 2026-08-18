import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { Search, Bell, ShieldCheck } from 'lucide-react';
import './Header.css';

const Header: React.FC = () => {
  const navigate = useNavigate();
  const [search, setSearch] = useState('');

  const handleSearchSubmit = (e: React.FormEvent) => {
    e.preventDefault();
    if (search.trim()) {
      // Redirect to the provider investigation page if the search input matches a potential provider ID format
      const query = search.trim().toUpperCase();
      if (query.startsWith('PRV')) {
        navigate(`/provider/${query}`);
      } else {
        navigate(`/queue?search=${query}`);
      }
      setSearch('');
    }
  };

  return (
    <header className="global-header">
      <form onSubmit={handleSearchSubmit} className="global-search-form">
        <Search size={16} className="search-icon" />
        <input 
          type="text" 
          placeholder="Global Provider Search (e.g. PRV51003)..."
          value={search}
          onChange={(e) => setSearch(e.target.value)}
        />
      </form>
      
      <div className="header-actions">
        <div className="system-health">
          <ShieldCheck size={16} className="health-icon active" />
          <span className="health-label">System Active</span>
        </div>
        
        <div className="notification-bell">
          <Bell size={18} />
          <span className="bell-badge"></span>
        </div>
        
        <div className="user-profile-badge">
          <div className="avatar-letter">A</div>
          <span>Auditor Agent</span>
        </div>
      </div>
    </header>
  );
};

export default Header;
