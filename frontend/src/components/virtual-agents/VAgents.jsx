import React, { useState, useEffect } from 'react';
import axios from 'axios';
import { useNavigate } from 'react-router-dom';
import Cookies from 'js-cookie';
import { Cpu } from 'lucide-react';
import styles from './VAgents.styles.js';
import './VAgents.css';
import { RAG_BACKEND_URL } from './VAgents.utils.js';

// Import Split Components
import { DashboardPage } from './AdminDashboard.jsx';
import { QueryView } from './ChatInterface.jsx';

// ==============================================================================
// Main App Component
// ==============================================================================
const VAgents = () => {
    const [view, setView] = useState('dashboard');
    const [currentUser, setCurrentUser] = useState(null);
    const navigate = useNavigate();

    useEffect(() => {
        const footer = document.querySelector("footer");
        if (footer) {
            const updateFooterHeight = () => {
                document.documentElement.style.setProperty(
                    "--footer-height",
                    `${footer.getBoundingClientRect().height}px`
                );
            };
            updateFooterHeight();
            window.addEventListener("resize", updateFooterHeight);
            return () => window.removeEventListener("resize", updateFooterHeight);
        }
    }, []);

    useEffect(() => {
        const syncUser = () => {
            const id = Cookies.get('userid');
            const name = Cookies.get('name');
            const type = Cookies.get('usertype');
            const firmId = Cookies.get('firmid');

            if (id && name && type && firmId) {
                let role = 'basic';
                if (type === 'ADMINAPP') role = 'admin';
                if (type === 'BUSINESSAPP') role = 'business';

                const user = { id, name, role, firmId };
                setCurrentUser(user);

                axios.post(`${RAG_BACKEND_URL}/api/users/sync`, user).catch(err => console.error("Failed to sync user", err));

                if (role === 'business' || role === 'basic') {
                    setView('query');
                }
            } else {
                navigate('/');
            }
        };
        syncUser();
    }, [navigate]);

    if (!currentUser) {
        return <div style={styles.appContainer}><div style={styles.loadingContainer}><h2>Initializing V-Agents...</h2></div></div>;
    }

    const renderNav = () => currentUser.role === 'admin' ? (
        <div style={styles.navButtonGroup}>
            <button onClick={() => setView('dashboard')} style={view === 'dashboard' ? styles.navButtonActive : styles.navButton}>Agent Command Center</button>
            <button onClick={() => setView('query')} style={view === 'query' ? styles.navButtonActive : styles.navButton}>Live Agents</button>
        </div>
    ) : null;

    const renderView = () => {
        switch (currentUser.role) {
            case 'admin':
                return view === 'query' ? <QueryView currentUser={currentUser} /> : <DashboardPage currentUser={currentUser} />;
            case 'business':
            case 'basic':
                return <QueryView currentUser={currentUser} />;
            default:
                return <div>Invalid user role.</div>
        }
    }

    return (
        <div style={styles.appContainer}>
            <nav style={styles.navbar}>
                <div style={styles.navLeft}>
                    <div style={{ background: 'var(--primary-light)', padding: '6px', borderRadius: '8px', display: 'flex' }}>
                        <Cpu size={24} style={{ color: 'var(--primary)' }} />
                    </div>
                    <h1 style={styles.navTitle}>V-Agents <span style={{ fontSize: '0.8em', opacity: 0.7, fontWeight: 400 }}>Enterprise</span></h1>
                </div>
                <div style={styles.navCenter}>
                    {renderNav()}
                </div>
                <div style={styles.navRight}>
                    <div style={styles.userBadge}>
                        <span style={styles.loggedInAs}>{currentUser.name}</span>
                        <span style={styles.roleBadge}>{currentUser.role}</span>
                    </div>
                </div>
            </nav>
            <main style={styles.mainContent}>{renderView()}</main>
        </div>
    );
};

export default VAgents;