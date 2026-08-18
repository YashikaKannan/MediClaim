import React from 'react';
import { NavLink } from 'react-router-dom';
import { 
  Shield, 
  LayoutDashboard, 
  UploadCloud, 
  ClipboardList, 
  BarChart3, 
  MessageSquareCode, 
  HelpCircle, 
  Settings as SettingsIcon,
  Database,
  UserCheck
} from 'lucide-react';
import './Sidebar.css';

const Sidebar: React.FC = () => {
  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <Shield className="brand-icon" />
        <span className="brand-text">Medi<span className="accent">Claim</span></span>
      </div>
      
      <nav className="sidebar-menu">
        <NavLink to="/" className={({ isActive }) => `menu-item ${isActive ? 'active' : ''}`}>
          <LayoutDashboard className="menu-icon" />
          <span>Dashboard</span>
        </NavLink>
        
        <NavLink to="/ingestion" className={({ isActive }) => `menu-item ${isActive ? 'active' : ''}`}>
          <UploadCloud className="menu-icon" />
          <span>Data Ingestion</span>
        </NavLink>
        
        <NavLink to="/queue" className={({ isActive }) => `menu-item ${isActive ? 'active' : ''}`}>
          <ClipboardList className="menu-icon" />
          <span>Investigation Queue</span>
        </NavLink>
        
        <NavLink to="/analytics" className={({ isActive }) => `menu-item ${isActive ? 'active' : ''}`}>
          <BarChart3 className="menu-icon" />
          <span>Analytics & Reports</span>
        </NavLink>
        
        <NavLink to="/assistant" className={({ isActive }) => `menu-item ${isActive ? 'active' : ''}`}>
          <MessageSquareCode className="menu-icon" />
          <span>AI Assistant</span>
        </NavLink>
        
        <NavLink to="/about" className={({ isActive }) => `menu-item ${isActive ? 'active' : ''}`}>
          <HelpCircle className="menu-icon" />
          <span>How It Works</span>
        </NavLink>
        
        <NavLink to="/settings" className={({ isActive }) => `menu-item ${isActive ? 'active' : ''}`}>
          <SettingsIcon className="menu-icon" />
          <span>Settings</span>
        </NavLink>
      </nav>
      
      <div className="sidebar-footer">
        <div className="status-panel">
          <div className="status-item">
            <Database size={14} className="status-icon" />
            <div>
              <p className="status-label">Database</p>
              <p className="status-value">Local SQLite</p>
            </div>
          </div>
          <div className="status-item">
            <UserCheck size={14} className="status-icon" />
            <div>
              <p className="status-label">Role</p>
              <p className="status-value">Senior Auditor</p>
            </div>
          </div>
        </div>
      </div>
    </aside>
  );
};

export default Sidebar;
