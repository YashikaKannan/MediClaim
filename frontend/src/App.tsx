import React from 'react';
import { HashRouter as Router, Routes, Route } from 'react-router-dom';
import Sidebar from './components/Sidebar';
import Header from './components/Header';

// Pages
import Dashboard from './pages/Dashboard';
import DataIngestion from './pages/DataIngestion';
import InvestigationQueue from './pages/InvestigationQueue';
import ProviderInvestigation from './pages/ProviderInvestigation';
import Analytics from './pages/Analytics';
import HowItWorks from './pages/HowItWorks';
import AIAssistant from './pages/AIAssistant';
import Settings from './pages/Settings';

const App: React.FC = () => {
  return (
    <Router>
      <div className="app-container">
        {/* Navigation Sidebar */}
        <Sidebar />
        
        {/* Main Content Area */}
        <main className="main-content">
          {/* Global Header Search & Actions */}
          <Header />
          
          {/* Dynamic Page Router */}
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/ingestion" element={<DataIngestion />} />
            <Route path="/queue" element={<InvestigationQueue />} />
            <Route path="/provider/:id" element={<ProviderInvestigation />} />
            <Route path="/analytics" element={<Analytics />} />
            <Route path="/about" element={<HowItWorks />} />
            <Route path="/assistant" element={<AIAssistant />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </main>
      </div>
    </Router>
  );
};

export default App;
