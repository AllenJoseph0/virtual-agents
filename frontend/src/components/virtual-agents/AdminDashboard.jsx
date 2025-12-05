import React, { useState, useEffect, useCallback, useMemo } from 'react';
import axios from 'axios';
import Cookies from 'js-cookie';
import {
    Cpu, BrainCircuit, ShieldAlert, Key, Loader2, Sparkles,
    UploadCloud, Bot, FileText, PlusSquare, RefreshCw, Trash2,
    Settings, ChevronDown, ShieldCheck, Save, FileEdit, X, Search, Database, Beaker, PlayCircle, CheckCircle, AlertCircle
} from 'lucide-react';
import styles from './VAgents.styles.js';
import { RAG_BACKEND_URL, generateUUID, formatBytes } from './VAgents.utils.js';

// ==============================================================================
// BENCHMARK RESULTS MODAL (FIXED)
// ==============================================================================
const BenchmarkModal = ({ isOpen, onClose, categoryName, onRunTest, isLoading, results, overallScore }) => {
    const [numQuestions, setNumQuestions] = useState(5);
    const [viewState, setViewState] = useState('config'); // config | loading | results

    useEffect(() => {
        if (!isOpen) return;

        // Force 'config' state if we have no results yet
        if (!isLoading && (!results || results.length === 0)) {
            setViewState('config');
        }
        // Force 'loading' state if loading
        else if (isLoading) {
            setViewState('loading');
        }
        // Force 'results' state ONLY if we have actual data
        else if (results && results.length > 0) {
            setViewState('results');
        }
    }, [isOpen, isLoading, results]);

    if (!isOpen) return null;

    const handleStart = () => {
        onRunTest(numQuestions);
    };

    const getScoreColor = (score) => {
        if (score >= 80) return 'var(--success)';
        if (score >= 50) return 'var(--warning)';
        return 'var(--danger)';
    };

    // Helper to safely render content that might accidentally be an object
    // Fixes: "Objects are not valid as a React child (found: object with keys {type, question})"
    const safeRender = (content) => {
        if (typeof content === 'object' && content !== null) {
            return content.question || content.text || JSON.stringify(content);
        }
        return content;
    };

    return (
        <div style={styles.modalOverlay}>
            <div style={styles.modalContent} onClick={(e) => e.stopPropagation()}>
                <div style={styles.modalHeader}>
                    <div>
                        <h2 style={styles.modalTitle}>Diagnostic Benchmark</h2>
                        <span style={{ fontSize: '0.85rem', color: 'var(--muted-foreground)' }}>Agent: {categoryName}</span>
                    </div>
                    <button onClick={onClose} style={styles.modalCloseButton}><X size={20} /></button>
                </div>

                <div style={styles.modalBody}>
                    {viewState === 'config' && (
                        <div style={styles.benchmarkConfig}>
                            <Beaker size={64} color="var(--primary)" style={{ marginBottom: '1rem' }} />
                            <div>
                                <h3 style={{ color: 'var(--foreground-heavy)', margin: '0 0 0.5rem 0', fontSize: '1.2rem' }}>Start New Test</h3>
                                <p style={{ color: 'var(--muted-foreground)', margin: '0 auto', maxWidth: '400px', lineHeight: '1.5' }}>
                                    Select how many questions to generate for the test run.
                                </p>
                            </div>

                            <div style={{ width: '100%', maxWidth: '300px', marginTop: '1.5rem' }}>
                                <label style={{ ...styles.label, textAlign: 'left' }}>Number of Questions</label>
                                <div style={{ display: 'flex', gap: '0.5rem', justifyContent: 'center' }}>
                                    {[3, 5, 10].map(n => (
                                        <button
                                            key={n}
                                            onClick={() => setNumQuestions(n)}
                                            style={numQuestions === n ? styles.buttonPrimary : styles.buttonSecondary}
                                        >
                                            {n} Questions
                                        </button>
                                    ))}
                                </div>
                            </div>

                            <button onClick={handleStart} style={{ ...styles.buttonPrimary, padding: '0.8rem 2rem', marginTop: '2rem', width: '100%', justifyContent: 'center' }}>
                                <PlayCircle size={20} /> Run Benchmark
                            </button>
                        </div>
                    )}

                    {viewState === 'loading' && (
                        <div style={{ ...styles.loadingContainer, padding: '4rem 0', flexDirection: 'column' }}>
                            <Loader2 style={{ ...styles.spinner, color: 'var(--primary)' }} size={48} />
                            <h3 style={{ fontWeight: 600, marginTop: '1.5rem', color: 'var(--foreground-heavy)' }}>Running Diagnostics...</h3>
                            <p style={{ fontSize: '0.9rem', color: 'var(--muted-foreground)', maxWidth: '300px', margin: '0.5rem auto 0', textAlign: 'center' }}>
                                Generating {numQuestions} questions and evaluating responses against the knowledge base context.
                            </p>
                        </div>
                    )}

                    {viewState === 'results' && (
                        <div style={styles.benchmarkContainer}>
                            <div style={styles.scoreCard}>
                                <div style={{ ...styles.scoreBig, color: getScoreColor(overallScore) }}>{overallScore}%</div>
                                <div style={styles.scoreLabel}>Overall Accuracy Score</div>
                            </div>

                            {(!results || results.length === 0) ? (
                                <p style={styles.p}>No results data found.</p>
                            ) : (
                                <div>
                                    {results.map((item, idx) => (
                                        <div key={idx} style={styles.benchmarkResultItem}>
                                            <div style={{ ...styles.benchmarkScoreBadge, background: getScoreColor(item.score), color: '#fff' }}>
                                                {item.score}%
                                            </div>
                                            <div style={styles.benchmarkLabel}>Test Question {idx + 1}</div>
                                            <div style={styles.benchmarkQuestion}>
                                                <span style={{ color: 'var(--warning-dark)', marginRight: '0.5rem' }}>Q:</span>
                                                {safeRender(item.question)}
                                            </div>
                                            <div style={styles.benchmarkAnswer}>
                                                <span style={{ fontWeight: 600, color: 'var(--primary)', display: 'block', marginBottom: '0.25rem', fontSize: '0.75rem' }}>AGENT ANSWER:</span>
                                                {safeRender(item.answer)}
                                            </div>
                                            <div style={{ ...styles.benchmarkReason, color: getScoreColor(item.score) }}>
                                                {item.score >= 80 ? <CheckCircle size={16} /> : <AlertCircle size={16} />}
                                                <span style={{ fontWeight: 600, marginLeft: '0.5rem' }}>Critique:</span> {safeRender(item.reason)}
                                            </div>
                                        </div>
                                    ))}
                                </div>
                            )}
                        </div>
                    )}
                </div>

                <div style={styles.modalFooter}>
                    <button onClick={onClose} style={styles.buttonSecondary}>Close</button>
                </div>
            </div>
        </div>
    );
};

// ==============================================================================
// FILE MANAGER POPUP MODAL
// ==============================================================================
const FileManagerModal = ({ isOpen, onClose, categoryName, files, onDeleteFile }) => {
    const [searchTerm, setSearchTerm] = useState('');

    if (!isOpen) return null;

    const filteredFiles = files.filter(f =>
        f.name.toLowerCase().includes(searchTerm.toLowerCase())
    );

    return (
        <div style={styles.modalOverlay}>
            <div style={styles.modalContent}>
                <div style={styles.modalHeader}>
                    <div>
                        <h2 style={styles.modalTitle}>Manage Knowledge Base</h2>
                        <span style={{ fontSize: '0.85rem', color: 'var(--muted-foreground)' }}>Agent: {categoryName}</span>
                    </div>
                    <button onClick={onClose} style={styles.modalCloseButton}><X size={20} /></button>
                </div>

                <div style={{ padding: '1.5rem 2rem 0' }}>
                    <div style={styles.searchWrapper}>
                        <Search size={18} style={styles.searchIcon} />
                        <input
                            style={styles.searchInput}
                            placeholder="Search files..."
                            value={searchTerm}
                            onChange={(e) => setSearchTerm(e.target.value)}
                            autoFocus
                        />
                    </div>
                </div>

                <div style={styles.modalBody}>
                    <div style={styles.fileListContainer}>
                        {filteredFiles.length === 0 ? (
                            <div style={styles.emptyStateSmall}>
                                {searchTerm ? 'No matching files found.' : 'No files in this knowledge base.'}
                            </div>
                        ) : (
                            <ul style={styles.fileList}>
                                {filteredFiles.map((fileObj) => (
                                    <li key={fileObj.name} style={styles.fileListItemCompact}>
                                        <div style={styles.fileInfoCompact} title={fileObj.name}>
                                            <FileText size={16} style={{ flexShrink: 0, color: 'var(--primary)' }} />
                                            <span style={styles.fileNameText}>{fileObj.name}</span>
                                        </div>
                                        <div style={styles.fileMetaCompact}>
                                            <span>{formatBytes(fileObj.size)}</span>
                                            <span style={{ fontSize: '0.7rem', opacity: 0.7 }}>{fileObj.date}</span>
                                            <button
                                                onClick={() => onDeleteFile(categoryName, fileObj.name)}
                                                style={styles.iconButtonDanger}
                                                title="Delete File"
                                            >
                                                <Trash2 size={16} />
                                            </button>
                                        </div>
                                    </li>
                                ))}
                            </ul>
                        )}
                    </div>
                    <div style={{ marginTop: '1rem', fontSize: '0.85rem', color: 'var(--muted-foreground)', textAlign: 'center' }}>
                        Showing {filteredFiles.length} of {files.length} files.
                    </div>
                </div>

                <div style={styles.modalFooter}>
                    <button onClick={onClose} style={styles.buttonSecondary}>Close</button>
                </div>
            </div>
        </div>
    );
};


// ==============================================================================
// Admin Dashboard Main Component
// ==============================================================================
export const DashboardPage = ({ currentUser }) => {
    const [dashboardView, setDashboardView] = useState('knowledge');
    return (
        <div style={{ maxWidth: '1200px', margin: '0 auto', padding: '0 1rem', width: '100%' }}>
            <header style={styles.header}>
                <h2 style={styles.headerH2}>Agent Command Center</h2>
                <p style={styles.headerSubtitle}>Design, train, and govern your fleet of Virtual Agents.</p>
            </header>
            <div style={styles.dashboardNav}>
                <button onClick={() => setDashboardView('knowledge')} style={dashboardView === 'knowledge' ? styles.dashboardNavButtonActive : styles.dashboardNavButton}>
                    <Cpu size={18} /> Agents & Memory
                </button>
                <button onClick={() => setDashboardView('personas')} style={dashboardView === 'personas' ? styles.dashboardNavButtonActive : styles.dashboardNavButton}>
                    <BrainCircuit size={18} /> Persona Hub
                </button>
                <button onClick={() => setDashboardView('compliance')} style={dashboardView === 'compliance' ? styles.dashboardNavButtonActive : styles.dashboardNavButton}>
                    <ShieldAlert size={18} /> Safety Protocols
                </button>
                <button onClick={() => setDashboardView('apiKeys')} style={dashboardView === 'apiKeys' ? styles.dashboardNavButtonActive : styles.dashboardNavButton}>
                    <Key size={18} /> Credentials
                </button>
            </div>
            {dashboardView === 'knowledge' && <AgentManager currentUser={currentUser} />}
            {dashboardView === 'personas' && <PersonaManager currentUser={currentUser} />}
            {dashboardView === 'compliance' && <ComplianceManager />}
            {dashboardView === 'apiKeys' && <ApiKeyManager currentUser={currentUser} />}
        </div>
    );
};

// ==============================================================================
// Category Access Control Component
// ==============================================================================
const CategoryAccessControl = ({ categoryName, initialPermissions, adminId, onPermissionsChange, currentUserRole }) => {
    const [permissions, setPermissions] = useState(initialPermissions || { business: false, basic: false });
    const [error, setError] = useState('');

    const handlePermissionChange = async (role, hasAccess) => {
        const newPermissions = { ...permissions, [role]: hasAccess };
        setPermissions(newPermissions);

        try {
            await axios.put(`${RAG_BACKEND_URL}/api/permissions/category`, {
                adminId,
                category: categoryName,
                roleToUpdate: role,
                hasAccess: hasAccess
            });
            onPermissionsChange(categoryName, newPermissions);
        } catch (err) {
            setError('Failed to update. Please refresh.');
            setPermissions(permissions);
        }
    };

    if (currentUserRole !== 'admin') {
        return null;
    }

    return (
        <div style={styles.permissionsContainer}>
            <div style={styles.permissionsHeader}><ShieldCheck size={18} /> Agent Deployment</div>
            {error && <div style={{ padding: '0 1.25rem', color: 'var(--danger)' }}>{error}</div>}
            <div style={styles.permissionsBody}>
                <div style={styles.permissionUserRow}>
                    <span>Deploy to Business Users</span>
                    <label className="switch">
                        <input
                            type="checkbox"
                            checked={permissions.business}
                            onChange={(e) => handlePermissionChange('business', e.target.checked)}
                        />
                        <span className="slider"></span>
                    </label>
                </div>
                <div style={styles.permissionUserRow}>
                    <span>Deploy to Basic Users</span>
                    <label className="switch">
                        <input
                            type="checkbox"
                            checked={permissions.basic}
                            onChange={(e) => handlePermissionChange('basic', e.target.checked)}
                        />
                        <span className="slider"></span>
                    </label>
                </div>
            </div>
        </div>
    );
};

// ==============================================================================
// Persona Selector (for Admin)
// ==============================================================================
const PersonaSelector = ({ categoryName, adminId, currentPersonaId, personas, onPersonaChange }) => {
    const currentPersona = personas.find(p => p.id === currentPersonaId);
    let displayValue = currentPersonaId && currentPersona ? currentPersona.name : (currentPersonaId ? categoryName : "Generic Assistant");

    return (
        <div style={styles.personaSelectorContainer}>
            <div style={styles.permissionsHeader}><BrainCircuit size={18} /> Agent Identity</div>
            <div style={styles.personaSelectorBody}>
                <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.95rem' }}>
                    <span style={{ fontWeight: 600, color: 'var(--foreground-heavy)' }}>Agent Identity:</span>
                    <span style={{ color: 'var(--foreground)' }}>{displayValue}</span>
                </div>
            </div>
        </div>
    );
};

// ==============================================================================
// Compliance Selector (for Admin)
// ==============================================================================
const ComplianceSelector = ({ categoryName, adminId, currentProfileId, profiles, onProfileChange }) => {
    const [selectedId, setSelectedId] = useState(currentProfileId || '');
    const [isSaving, setIsSaving] = useState(false);

    const handleChange = async (e) => {
        const newProfileId = e.target.value;
        setSelectedId(newProfileId);
        setIsSaving(true);
        try {
            await axios.put(`${RAG_BACKEND_URL}/api/category/settings`, {
                adminId,
                categoryName,
                settings: { complianceProfileId: newProfileId || null }
            });
            onProfileChange(categoryName, newProfileId);
        } catch (err) {
            console.error("Failed to save compliance setting", err);
            setSelectedId(currentProfileId || '');
        } finally {
            setIsSaving(false);
        }
    };

    useEffect(() => {
        setSelectedId(currentProfileId || '');
    }, [currentProfileId]);

    return (
        <div style={styles.personaSelectorContainer}>
            <div style={styles.permissionsHeader}><ShieldAlert size={18} /> Safety & Compliance Protocol</div>
            <div style={styles.personaSelectorBody}>
                <label style={{ ...styles.label, marginBottom: 0, flexShrink: 0 }}>Active Protocol:</label>
                <select value={selectedId} onChange={handleChange} style={{ ...styles.input, flexGrow: 1 }}>
                    <option value="">-- Unrestricted --</option>
                    {profiles.map(p => (
                        <option key={p.id} value={p.id}>{p.name}</option>
                    ))}
                </select>
                {isSaving && <Loader2 size={16} style={{ ...styles.spinner, marginLeft: '1rem' }} />}
            </div>
        </div>
    );
};

// ==============================================================================
// Agent Manager (Updated with Modal File Manager & Benchmark)
// ==============================================================================
const AgentManager = ({ currentUser }) => {
    const [structure, setStructure] = useState([]);
    const [personas, setPersonas] = useState([]);
    const [complianceProfiles, setComplianceProfiles] = useState([]);
    const [loading, setLoading] = useState(true);
    const [message, setMessage] = useState('');
    const [error, setError] = useState('');

    const [selectedCategory, setSelectedCategory] = useState('');
    const [newCategoryName, setNewCategoryName] = useState('');

    const [files, setFiles] = useState(null);
    const [isUploading, setIsUploading] = useState(false);
    const [activeJob, setActiveJob] = useState(null);
    const [openSettings, setOpenSettings] = useState({});

    // File Manager Modal State
    const [fileManagerOpen, setFileManagerOpen] = useState(false);
    const [fileManagerData, setFileManagerData] = useState({ categoryName: '', files: [] });

    // Benchmark Modal State
    const [benchmarkOpen, setBenchmarkOpen] = useState(false);
    // CRITICAL: Initialize results as NULL to indicate "not run yet"
    const [benchmarkData, setBenchmarkData] = useState({ categoryName: '', results: null, overallScore: 0 });
    const [isBenchmarking, setIsBenchmarking] = useState(false);
    const [currentBenchmarkCat, setCurrentBenchmarkCat] = useState(null);

    const fetchData = useCallback(async () => {
        setLoading(true);
        setError('');
        try {
            const [structureResp, personasResp, complianceResp] = await Promise.all([
                axios.get(`${RAG_BACKEND_URL}/api/rag/structure?username=${currentUser.id}`),
                axios.get(`${RAG_BACKEND_URL}/api/personas`),
                axios.get(`${RAG_BACKEND_URL}/api/compliance`)
            ]);
            setStructure(structureResp.data?.[currentUser.id] || []);
            setPersonas(personasResp.data || []);
            setComplianceProfiles(complianceResp.data || []);
        } catch (err) {
            setError(err.response?.data?.error || 'Could not fetch data.');
        } finally {
            setLoading(false);
        }
    }, [currentUser.id]);

    useEffect(() => { fetchData(); }, [fetchData]);

    const handlePermissionsChange = (categoryName, newPermissions) => {
        setStructure(currentStructure =>
            currentStructure.map(cat =>
                cat.name === categoryName ? { ...cat, permissions: newPermissions } : cat
            )
        );
    };

    const handlePersonaChange = (categoryName, newPersonaId) => {
        setStructure(currentStructure =>
            currentStructure.map(cat =>
                cat.name === categoryName ? { ...cat, personaId: newPersonaId } : cat
            )
        );
    };

    const handleComplianceChange = (categoryName, newProfileId) => {
        setStructure(currentStructure =>
            currentStructure.map(cat =>
                cat.name === categoryName ? { ...cat, complianceProfileId: newProfileId } : cat
            )
        );
    };

    const handleAction = async (action, category) => {
        const confirmationText = {
            'update-index': `Retrain '${category}' agent with new knowledge files?`,
            'create-index': `Full Retrain: Rebuild knowledge index for '${category}'? This may take a while.`,
            'delete-index': `Wipe knowledge index for '${category}'? Files will remain.`,
            'delete-category': `Delete Agent '${category}'? This removes all knowledge files and configuration.`
        };
        if (!window.confirm(confirmationText[action])) return;

        setMessage(''); setError('');
        setActiveJob({ type: action, category });

        try {
            const payload = {
                username: currentUser.id,
                category,
                firm_id: currentUser.firmId
            };
            const resp = await axios.post(`${RAG_BACKEND_URL}/api/rag/${action}`, payload);
            setMessage(resp.data.message || 'Operation successful.');
            fetchData();
        } catch (err) {
            setError(err.response?.data?.error || `Failed to ${action.replace('-', ' ')}.`);
        } finally {
            setActiveJob(null);
        }
    };

    // --- BENCHMARK FUNCTIONALITY ---
    const openBenchmarkModal = (cat) => {
        setCurrentBenchmarkCat(cat);
        // FORCE RESET: results must be NULL to show config screen
        setBenchmarkData({ categoryName: cat.name, results: null, overallScore: 0 });
        setBenchmarkOpen(true);
    };

    const handleRunBenchmark = async (numQuestions) => {
        if (!currentBenchmarkCat) return;
        setIsBenchmarking(true);
        try {
            const payload = {
                adminId: currentUser.id,
                owner_id: currentUser.id,
                category: currentBenchmarkCat.name,
                personaId: currentBenchmarkCat.personaId,
                complianceProfileId: currentBenchmarkCat.complianceProfileId,
                num_questions: numQuestions,
                firmId: currentUser.firmId
            };
            const resp = await axios.post(`${RAG_BACKEND_URL}/api/rag/run-test`, payload);
            setBenchmarkData({
                categoryName: currentBenchmarkCat.name,
                results: resp.data.results || [],
                overallScore: resp.data.overall_score || 0
            });
        } catch (err) {
            console.error("Benchmark failed", err);
            setError("Benchmark failed to run. Ensure Agent is trained and active.");
            setBenchmarkOpen(false);
        } finally {
            setIsBenchmarking(false);
        }
    };

    const handleUpload = async (e) => {
        e.preventDefault();
        const finalCategoryName = selectedCategory === '__NEW__' ? newCategoryName.trim() : selectedCategory;

        if (!files || !finalCategoryName) {
            setError('Please provide an Agent name and select knowledge files.');
            return;
        }
        const fd = new FormData();
        fd.append('username', currentUser.id);
        fd.append('category', finalCategoryName);
        for (let i = 0; i < files.length; i++) fd.append('files', files[i]);

        setIsUploading(true); setError(''); setMessage('');
        try {
            const resp = await axios.post(`${RAG_BACKEND_URL}/api/rag/upload`, fd);

            if (selectedCategory === '__NEW__') {
                try {
                    const defaultPrompt = `You are an intelligent assistant specialized in ${finalCategoryName}. Use the provided knowledge base to answer questions accurately and concisely.`;
                    const createPersonaResp = await axios.post(`${RAG_BACKEND_URL}/api/personas`, {
                        name: finalCategoryName,
                        prompt: defaultPrompt,
                        voice_prompt: defaultPrompt,
                        firm_id: currentUser.firmId,
                        stages: []
                    });
                    const createdPersonaId = createPersonaResp.data.id;
                    if (createdPersonaId) {
                        await axios.put(`${RAG_BACKEND_URL}/api/category/settings`, {
                            adminId: currentUser.id,
                            categoryName: finalCategoryName,
                            settings: { personaId: createdPersonaId }
                        });
                        setMessage(`${resp.data.message} Agent created and default Persona assigned.`);
                    }
                } catch (personaErr) {
                    console.error("Auto-persona creation failed", personaErr);
                    setMessage(`${resp.data.message} Agent created, but failed to auto-generate Persona.`);
                }
            } else {
                setMessage(`${resp.data.message} Knowledge updated.`);
            }

            setSelectedCategory('');
            setNewCategoryName('');
            setFiles(null);
            if (e.target.reset) e.target.reset();

            setTimeout(() => { fetchData(); }, 800);

        } catch (err) {
            setError(err.response?.data?.error || 'Knowledge upload failed.');
        } finally {
            setIsUploading(false);
        }
    };

    const openFileManager = (cat) => {
        setFileManagerData({ categoryName: cat.name, files: cat.files });
        setFileManagerOpen(true);
    };

    const handleDeleteFile = async (categoryName, fileName) => {
        if (!window.confirm(`Delete "${fileName}"? This cannot be undone.`)) return;
        try {
            await axios.delete(`${RAG_BACKEND_URL}/api/rag/file`, {
                data: { username: currentUser.id, category: categoryName, filename: fileName }
            });

            // Remove file locally from modal state to UI updates instantly
            setFileManagerData(prev => ({
                ...prev,
                files: prev.files.filter(f => f.name !== fileName)
            }));

            // Refresh main data in background
            fetchData();
        } catch (err) {
            alert('Failed to delete file.');
        }
    };

    return (
        <div className="fade-in" style={{ maxWidth: '900px', margin: '0 auto', width: '100%' }}>
            {message && <div style={{ ...styles.alert, ...styles.alertSuccess, marginTop: '1rem' }}>{message}</div>}
            {error && <div style={{ ...styles.alert, ...styles.alertDanger, marginTop: '1rem' }}>{error}</div>}

            <div style={styles.card}>
                <div style={styles.cardHeader}><Sparkles size={20} /> Initialize or Train Agent</div>
                <div style={styles.cardBody}>
                    <form onSubmit={handleUpload}>
                        <label style={styles.label}>1. Select Target Agent</label>
                        <select
                            style={styles.input}
                            value={selectedCategory}
                            onChange={e => setSelectedCategory(e.target.value)}
                            required
                        >
                            <option value="" disabled>Select agent to train...</option>
                            {structure.map(cat => <option key={cat.name} value={cat.name}>{cat.name}</option>)}
                            <option value="__NEW__">+ Create New Agent</option>
                        </select>

                        {selectedCategory === '__NEW__' && (
                            <input
                                style={{ ...styles.input, marginTop: '1rem' }}
                                value={newCategoryName}
                                onChange={e => setNewCategoryName(e.target.value)}
                                placeholder="Name your new agent (e.g., 'Journalist', 'Coach')"
                                required
                            />
                        )}

                        <label style={{ ...styles.label, marginTop: '1rem' }}>2. Upload Knowledge Documents</label>
                        <input type="file" multiple required onChange={e => setFiles(e.target.files)} style={styles.fileInput} />
                        <div style={{ marginTop: '1rem' }}>
                            <button type="submit" style={styles.buttonPrimary} disabled={isUploading}>
                                {isUploading ? <><Loader2 style={styles.spinner} size={16} /> Ingesting Data...</> : <><UploadCloud size={16} /> Upload & Train</>}
                            </button>
                        </div>
                    </form>
                </div>
            </div>

            {loading && <div style={styles.loadingContainer}><Loader2 style={styles.spinner} size={24} /> Fetching Agents...</div>}
            {!loading && structure.length === 0 && <p style={styles.p}>No Agents created. Create one above.</p>}

            <div style={styles.agentGrid}>
                {structure.map(cat => (
                    <div key={cat.name} style={styles.card}>
                        <div style={styles.categoryHeader}>
                            <div style={styles.categoryTitle}>
                                <div style={styles.messageAvatar}><Bot size={20} /></div>
                                {cat.name}
                                <span style={{ ...styles.indexStatus, backgroundColor: cat.indexStatus === 'ACTIVE' ? 'var(--success)' : 'var(--warning-dark)' }}>
                                    {cat.indexStatus === 'ACTIVE' ? 'Online' : 'Needs Training'}
                                </span>
                            </div>
                        </div>

                        <div style={{ padding: '0 1.75rem 1.25rem' }}>
                            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '1rem', marginTop: '1.25rem' }}>
                                <div>
                                    <h4 style={{ margin: 0, fontSize: '0.9rem', color: 'var(--foreground-heavy)', fontWeight: 600 }}>Knowledge Base</h4>
                                    <span style={{ fontSize: '0.8rem', color: 'var(--muted-foreground)' }}>{cat.files.length} documents indexed</span>
                                </div>
                                <button
                                    onClick={() => openFileManager(cat)}
                                    style={styles.buttonSecondary}
                                    title="View and Manage Files"
                                >
                                    <Database size={14} /> Manage Files
                                </button>
                            </div>
                        </div>

                        <div style={styles.cardFooter}>
                            <div style={styles.buttonGroup}>
                                <button style={styles.buttonSuccess} onClick={() => handleAction('update-index', cat.name)} disabled={!!activeJob} title="Incremental Training">
                                    {activeJob?.type === 'update-index' && activeJob?.category === cat.name ? <><Loader2 style={styles.spinner} size={16} /></> : <PlusSquare size={16} />} Train
                                </button>
                                <button style={styles.buttonWarning} onClick={() => handleAction('create-index', cat.name)} disabled={!!activeJob} title="Full Retraining">
                                    {activeJob?.type === 'create-index' && activeJob?.category === cat.name ? <><Loader2 style={styles.spinner} size={16} /></> : <RefreshCw size={16} />} Rebuild
                                </button>
                                {/* NEW BENCHMARK BUTTON */}
                                <button
                                    style={styles.buttonSecondary}
                                    onClick={() => openBenchmarkModal(cat)}
                                    disabled={!!activeJob || cat.indexStatus !== 'ACTIVE'}
                                    title="Run Diagnostic Benchmark"
                                >
                                    <Beaker size={16} /> Test
                                </button>
                                <button style={styles.iconButtonDanger} onClick={() => handleAction('delete-category', cat.name)} disabled={!!activeJob} title="Delete Agent">
                                    <Trash2 size={18} />
                                </button>
                            </div>
                        </div>
                        {currentUser.role === 'admin' && (
                            <>
                                <button
                                    onClick={() => setOpenSettings(prev => ({ ...prev, [cat.name]: !prev[cat.name] }))}
                                    style={{
                                        width: '100%',
                                        padding: '1rem 1.75rem',
                                        background: 'var(--background)',
                                        border: 'none',
                                        borderTop: '1px solid var(--border)',
                                        cursor: 'pointer',
                                        display: 'flex',
                                        alignItems: 'center',
                                        justifyContent: 'space-between',
                                        fontSize: '0.875rem',
                                        fontWeight: 600,
                                        color: 'var(--foreground)',
                                        transition: 'all 0.2s'
                                    }}
                                >
                                    <span style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                                        <Settings size={16} /> Advanced Settings
                                    </span>
                                    <ChevronDown size={16} style={{ transform: openSettings[cat.name] ? 'rotate(180deg)' : 'rotate(0)', transition: 'transform 0.2s' }} />
                                </button>
                                {openSettings[cat.name] && (
                                    <div style={{ background: 'var(--chat-bg)', borderTop: '1px solid var(--border)' }}>
                                        <CategoryAccessControl
                                            categoryName={cat.name}
                                            initialPermissions={cat.permissions}
                                            adminId={currentUser.id}
                                            onPermissionsChange={handlePermissionsChange}
                                            currentUserRole={currentUser.role}
                                        />
                                        <PersonaSelector
                                            categoryName={cat.name}
                                            adminId={currentUser.id}
                                            currentPersonaId={cat.personaId}
                                            personas={personas}
                                            onPersonaChange={handlePersonaChange}
                                        />
                                        <ComplianceSelector
                                            categoryName={cat.name}
                                            adminId={currentUser.id}
                                            currentProfileId={cat.complianceProfileId}
                                            profiles={complianceProfiles}
                                            onProfileChange={handleComplianceChange}
                                        />
                                    </div>
                                )}
                            </>
                        )}
                    </div>
                ))}
            </div>

            <FileManagerModal
                isOpen={fileManagerOpen}
                onClose={() => setFileManagerOpen(false)}
                categoryName={fileManagerData.categoryName}
                files={fileManagerData.files}
                onDeleteFile={handleDeleteFile}
            />

            <BenchmarkModal
                isOpen={benchmarkOpen}
                onClose={() => setBenchmarkOpen(false)}
                categoryName={currentBenchmarkCat?.name}
                onRunTest={handleRunBenchmark}
                isLoading={isBenchmarking}
                results={benchmarkData?.results}
                overallScore={benchmarkData?.overallScore}
            />
        </div>
    );
};

// ==============================================================================
// Persona Manager Component
// ==============================================================================
const PersonaManager = ({ currentUser }) => {
    const [personas, setPersonas] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState('');
    const [editingPersona, setEditingPersona] = useState(null);

    const fetchPersonas = useCallback(async () => {
        setIsLoading(true);
        try {
            const res = await axios.get(`${RAG_BACKEND_URL}/api/personas`);
            setPersonas(res.data);
        } catch (err) {
            setError('Failed to fetch personas.');
        } finally {
            setIsLoading(false);
        }
    }, []);

    useEffect(() => {
        fetchPersonas();
    }, [fetchPersonas]);

    const handleSave = async (personaData) => {
        const isUpdate = !!personaData.id;
        const url = isUpdate ? `${RAG_BACKEND_URL}/api/personas/${personaData.id}` : `${RAG_BACKEND_URL}/api/personas`;
        const method = isUpdate ? 'put' : 'post';

        const payload = { ...personaData, firm_id: currentUser.firmId };
        try {
            await axios[method](url, payload);
            setEditingPersona(null);
            fetchPersonas();
        } catch (err) {
            setError(err.response?.data?.error || 'Failed to save persona.');
        }
    };

    const handleDelete = async (personaId) => {
        if (window.confirm('Delete this persona? Agents using it will revert to default behavior.')) {
            try {
                await axios.delete(`${RAG_BACKEND_URL}/api/personas/${personaId}`);
                fetchPersonas();
            } catch (err) {
                setError('Failed to delete persona.');
            }
        }
    };

    const PersonaForm = ({ persona, onSave, onCancel }) => {
        const [name, setName] = useState(persona?.name || '');
        const [prompt, setPrompt] = useState(persona?.prompt || '');
        const [voicePrompt, setVoicePrompt] = useState(persona?.voice_prompt || '');
        const [stages, setStages] = useState((persona?.stages || []).join(', '));
        const [isSaving, setIsSaving] = useState(false);
        const isEditing = !!persona?.id;

        const handleSubmit = (e) => {
            e.preventDefault();
            if (!name.trim()) return;
            setIsSaving(true);
            const stagesArray = stages.split(',').map(s => s.trim()).filter(Boolean);
            onSave({ ...persona, name, prompt, voice_prompt: voicePrompt, stages: stagesArray }).finally(() => setIsSaving(false));
        };

        return (
            <div style={styles.modalOverlay}>
                <div style={styles.modalContent} onClick={(e) => e.stopPropagation()}>
                    <div style={styles.modalHeader}>
                        <h2 style={styles.modalTitle}>{isEditing ? 'Edit Agent Persona' : 'New Agent Persona'}</h2>
                        <button onClick={onCancel} style={styles.modalCloseButton}><X size={20} /></button>
                    </div>
                    <form onSubmit={handleSubmit} style={styles.modalBody}>
                        <div style={styles.formGroup}>
                            <label style={styles.label}>Persona Name</label>
                            <input className="modal-input" style={styles.input} value={name} onChange={e => setName(e.target.value)} placeholder="e.g., 'Empathetic Support Agent'" required />
                        </div>
                        <div style={styles.promptContainer}>
                            <div style={{ ...styles.formGroup, ...styles.promptColumn }}>
                                <label style={styles.label}>Voice Interaction Instructions</label>
                                <textarea className="modal-input" style={styles.textarea} value={voicePrompt} onChange={e => setVoicePrompt(e.target.value)} rows={10} placeholder="You are a helpful assistant. Speak concisely. Use a warm tone..." />
                            </div>
                            <div style={{ ...styles.formGroup, ...styles.promptColumn }}>
                                <label style={styles.label}>Text Chat Instructions</label>
                                <textarea className="modal-input" style={styles.textarea} value={prompt} onChange={e => setPrompt(e.target.value)} rows={10} placeholder="You are a helpful assistant. Use markdown for formatting. Be detailed..." />
                            </div>
                        </div>
                        <div style={styles.formGroup}>
                            <label style={styles.label}>Conversation Stages (Optional)</label>
                            <input className="modal-input" style={styles.input} value={stages} onChange={e => setStages(e.target.value)} placeholder="greeting, inquiry, resolution, closing" />
                        </div>
                    </form>
                    <div style={styles.modalFooter}>
                        <button type="button" onClick={onCancel} style={styles.buttonSecondary}>Cancel</button>
                        <button type="button" onClick={handleSubmit} style={styles.buttonSuccess} disabled={isSaving || !name.trim()}>
                            {isSaving ? <><Loader2 style={styles.spinner} size={16} /> Saving...</> : <><Save size={16} /> Save Persona</>}
                        </button>
                    </div>
                </div>
            </div>
        );
    };

    if (isLoading) return <div style={styles.loadingContainer}><Loader2 style={styles.spinner} size={24} /> Loading personas...</div>;

    return (
        <div className="fade-in" style={{ maxWidth: '900px', margin: '0 auto', width: '100%' }}>
            {error && <div style={{ ...styles.alert, ...styles.alertDanger, marginTop: '1rem' }}>{error}</div>}
            {editingPersona && <PersonaForm persona={editingPersona} onSave={handleSave} onCancel={() => setEditingPersona(null)} />}
            <div style={styles.card}>
                <div style={styles.cardHeader}><BrainCircuit size={20} /> Agent Personas</div>
                {personas.length === 0 ? <p style={styles.p}>No personas defined.</p> : personas.map(p => (
                    <div key={p.id} style={styles.personaItem}>
                        <div>
                            <h3 style={styles.personaName}>{p.name}</h3>
                            <p style={styles.personaPrompt}><strong>Core Directive:</strong> {p.prompt?.substring(0, 100)}{p.prompt?.length > 100 ? '...' : ''}</p>
                        </div>
                        <div style={styles.personaActions}>
                            <button onClick={() => setEditingPersona(p)} style={styles.iconButton} title="Edit Persona"><FileEdit size={18} /></button>
                            <button onClick={() => handleDelete(p.id)} style={styles.iconButtonDanger} title="Delete Persona"><Trash2 size={18} /></button>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

// ==============================================================================
// Compliance Profile Manager (Included for completeness)
// ==============================================================================
const ComplianceManager = ({ currentUser }) => {
    const [profiles, setProfiles] = useState([]);
    const [isLoading, setIsLoading] = useState(true);
    const [error, setError] = useState('');
    const [editingProfile, setEditingProfile] = useState(null);
    const [isCreating, setIsCreating] = useState(false);

    const fetchProfiles = useCallback(async () => {
        setIsLoading(true);
        try {
            const res = await axios.get(`${RAG_BACKEND_URL}/api/compliance`);
            setProfiles(res.data);
        } catch (err) { setError('Failed to fetch compliance profiles.'); } finally { setIsLoading(false); }
    }, []);

    useEffect(() => { fetchProfiles(); }, [fetchProfiles]);

    const handleSave = async (profileData) => {
        try {
            const payload = { id: profileData.id || generateUUID(), name: profileData.name, content: profileData.content };
            await axios.post(`${RAG_BACKEND_URL}/api/compliance`, payload);
            setEditingProfile(null); setIsCreating(false); fetchProfiles();
        } catch (err) { setError(err.response?.data?.error || 'Failed to save profile.'); }
    };

    const handleDelete = async (profileId) => {
        if (window.confirm('Delete this compliance profile?')) {
            try { await axios.delete(`${RAG_BACKEND_URL}/api/compliance/${profileId}`); fetchProfiles(); } catch (err) { setError('Failed to delete profile.'); }
        }
    };

    const ComplianceProfileForm = ({ profile, onSave, onCancel }) => {
        const [name, setName] = useState(profile?.name || '');
        const [rules, setRules] = useState(profile?.content || '');
        const [isSaving, setIsSaving] = useState(false);
        const isEditing = !!profile?.id;

        const handleSubmit = (e) => {
            e.preventDefault(); if (!name.trim()) return;
            setIsSaving(true); onSave({ ...profile, name, content: rules }).finally(() => setIsSaving(false));
        };

        return (
            <div style={styles.modalOverlay}>
                <div style={styles.modalContent} onClick={(e) => e.stopPropagation()}>
                    <div style={styles.modalHeader}>
                        <h2 style={styles.modalTitle}>{isEditing ? 'Edit' : 'Create'} Safety Protocol</h2>
                        <button onClick={onCancel} style={styles.modalCloseButton}><X size={20} /></button>
                    </div>
                    <form onSubmit={handleSubmit} style={styles.modalBody}>
                        <div style={styles.formGroup}>
                            <label style={styles.label}>Protocol Name</label>
                            <input className="modal-input" style={styles.input} value={name} onChange={e => setName(e.target.value)} placeholder="e.g., GDPR Strict" required />
                        </div>
                        <div style={styles.formGroup}>
                            <label style={styles.label}>Rules & Constraints</label>
                            <textarea className="modal-input" style={styles.textarea} value={rules} onChange={(e) => setRules(e.target.value)} rows={8} placeholder={"No financial advice, 100%\nAvoid political topics, 80%"} />
                        </div>
                    </form>
                    <div style={styles.modalFooter}>
                        <button type="button" onClick={onCancel} style={styles.buttonSecondary}>Cancel</button>
                        <button type="button" onClick={handleSubmit} style={styles.buttonSuccess} disabled={isSaving || !name.trim()}>
                            {isSaving ? <><Loader2 style={styles.spinner} size={16} /> Saving...</> : <><Save size={16} /> Save Protocol</>}
                        </button>
                    </div>
                </div>
            </div>
        );
    };

    if (isLoading) return <div style={styles.loadingContainer}><Loader2 style={styles.spinner} size={24} /> Loading protocols...</div>;

    return (
        <div className="fade-in">
            {error && <div style={{ ...styles.alert, ...styles.alertDanger, marginTop: '1rem' }}>{error}</div>}
            {(isCreating || editingProfile) && <ComplianceProfileForm profile={editingProfile} onSave={handleSave} onCancel={() => { setEditingProfile(null); setIsCreating(false); }} />}
            <div style={styles.card}>
                <div style={{ ...styles.cardHeader, justifyContent: 'space-between' }}>
                    <div>Safety Protocols</div>
                    <button onClick={() => setIsCreating(true)} style={styles.buttonPrimary}><PlusSquare size={16} /> New Protocol</button>
                </div>
                {profiles.length === 0 ? <p style={styles.p}>No protocols defined.</p> : profiles.map(p => (
                    <div key={p.id} style={styles.personaItem}>
                        <div>
                            <h3 style={styles.personaName}>{p.name}</h3>
                            <pre style={styles.complianceContentPreview}>{p.content?.substring(0, 200)}{p.content?.length > 200 ? '...' : ''}</pre>
                        </div>
                        <div style={styles.personaActions}>
                            <button onClick={() => setEditingProfile(p)} style={styles.iconButton} title="Edit Protocol"><FileEdit size={18} /></button>
                            <button onClick={() => handleDelete(p.id)} style={styles.iconButtonDanger} title="Delete Protocol"><Trash2 size={18} /></button>
                        </div>
                    </div>
                ))}
            </div>
        </div>
    );
};

// ==============================================================================
// API Key Manager (Included for completeness)
// ==============================================================================
const ApiKeyManager = ({ currentUser }) => {
    const [apiKeys, setApiKeys] = useState([]);
    const [llmProviders, setLlmProviders] = useState([]);
    const [llmProviderTypes, setLlmProviderTypes] = useState([]);
    const [editingKey, setEditingKey] = useState(null);
    const [selectedProvider, setSelectedProvider] = useState('');
    const [selectedProviderType, setSelectedProviderType] = useState('');
    const [newApiKey, setNewApiKey] = useState('');
    const [isLoading, setIsLoading] = useState(true);
    const [isSaving, setIsSaving] = useState(false);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
    const firmid = Cookies.get('firmid');

    const fetchLlmOptions = useCallback(async () => {
        try {
            const res = await axios.get(`${RAG_BACKEND_URL}/api/llm/options`);
            setLlmProviders(res.data.providers || []);
            setLlmProviderTypes(res.data.types || []);
            if (res.data.providers?.length > 0) setSelectedProvider(res.data.providers[0]);
            if (res.data.types?.length > 0) setSelectedProviderType(res.data.types[0]);
        } catch (err) { setError('Failed to fetch LLM options.'); }
    }, []);

    const fetchApiKeys = useCallback(async () => {
        setIsLoading(true);
        if (!firmid || !currentUser.id) return;
        try {
            const res = await axios.get(`${RAG_BACKEND_URL}/api/llm/keys?userId=${currentUser.id}&firmId=${firmid}`);
            setApiKeys(res.data || []);
        } catch (err) { setError('Failed to fetch API keys.'); } finally { setIsLoading(false); }
    }, [currentUser.id, firmid]);

    useEffect(() => { fetchLlmOptions(); fetchApiKeys(); }, [fetchLlmOptions, fetchApiKeys]);

    const handleSaveKey = async (e) => {
        e.preventDefault(); if (!selectedProvider || !selectedProviderType || !newApiKey.trim()) { setError('Please fill all fields.'); return; }
        setIsSaving(true); setError(''); setSuccess('');
        try {
            await axios.post(`${RAG_BACKEND_URL}/api/llm/keys`, { userId: currentUser.id, firmId: firmid, llmProvider: selectedProvider, llmProviderType: selectedProviderType, apiKey: newApiKey.trim() });
            setSuccess('API Key saved!'); setNewApiKey(''); fetchApiKeys(); setTimeout(() => setSuccess(''), 3000);
        } catch (err) { setError(err.response?.data?.error || 'Failed to save API key.'); } finally { setIsSaving(false); }
    };

    const handleDeleteKey = async (keyId) => {
        if (!window.confirm('Delete this API key?')) return;
        try {
            await axios.delete(`${RAG_BACKEND_URL}/api/llm/keys/${keyId}`, { data: { userId: currentUser.id, firmId: firmid } });
            setSuccess('Deleted!'); fetchApiKeys(); setTimeout(() => setSuccess(''), 3000);
        } catch (err) { setError('Failed to delete.'); }
    };

    const handleUpdateKey = async (keyData) => {
        try {
            await axios.put(`${RAG_BACKEND_URL}/api/llm/keys/${keyData.ID}`, { ...keyData, userId: currentUser.id, firmId: firmid });
            setSuccess('Updated!'); setEditingKey(null); fetchApiKeys(); setTimeout(() => setSuccess(''), 3000);
        } catch (err) { setError('Failed to update.'); }
    };

    const maskApiKey = (key) => (!key || key.length < 8) ? '********' : `${key.substring(0, 4)}...${key.substring(key.length - 4)}`;

    const ApiKeyEditModal = ({ keyData, onSave, onCancel }) => {
        const [apiKey, setApiKey] = useState(keyData.API_KEY);
        const [status, setStatus] = useState(keyData.STATUS);
        const [isSaving, setIsSaving] = useState(false);
        const handleSubmit = async (e) => { e.preventDefault(); setIsSaving(true); await onSave({ ...keyData, API_KEY: apiKey, STATUS: status }); setIsSaving(false); };
        return (
            <div style={styles.modalOverlay}>
                <div style={styles.modalContent} onClick={(e) => e.stopPropagation()}>
                    <div style={styles.modalHeader}><h2 style={styles.modalTitle}>Edit Credentials</h2><button onClick={onCancel} style={styles.modalCloseButton}><X size={20} /></button></div>
                    <form onSubmit={handleSubmit} style={styles.modalBody}>
                        <div style={styles.formGroup}><label style={styles.label}>API Key</label><input className="modal-input" type="password" style={styles.input} value={apiKey} onChange={e => setApiKey(e.target.value)} /></div>
                        <div style={styles.formGroup}><label style={styles.label}>Status</label><select className="modal-input" style={styles.input} value={status} onChange={e => setStatus(e.target.value)}><option value="ACTIVE">ACTIVE</option><option value="INACTIVE">INACTIVE</option></select></div>
                    </form>
                    <div style={styles.modalFooter}><button onClick={onCancel} style={styles.buttonSecondary}>Cancel</button><button onClick={handleSubmit} style={styles.buttonSuccess} disabled={isSaving}>{isSaving ? 'Updating...' : 'Update'}</button></div>
                </div>
            </div>
        );
    };

    return (
        <div className="fade-in">
            {success && <div style={{ ...styles.alert, ...styles.alertSuccess, marginTop: '1rem' }}>{success}</div>}
            {error && <div style={{ ...styles.alert, ...styles.alertDanger, marginTop: '1rem' }}>{error}</div>}
            <div style={styles.card}>
                <div style={styles.cardHeader}><PlusSquare size={20} /> Add Provider Key</div>
                <form onSubmit={handleSaveKey}>
                    <div style={styles.cardBody}>
                        <div style={styles.formGroup}><label style={styles.label}>Service Type</label><select style={styles.input} value={selectedProviderType} onChange={e => setSelectedProviderType(e.target.value)} required > <option value="" disabled>Select type...</option> {llmProviderTypes.map(type => <option key={type} value={type}>{type}</option>)} </select></div>
                        <div style={styles.formGroup}><label style={styles.label}>Service Provider</label><select style={styles.input} value={selectedProvider} onChange={e => setSelectedProvider(e.target.value)} required > <option value="" disabled>Select provider...</option> {llmProviders.map(provider => <option key={provider} value={provider}>{provider}</option>)} </select></div>
                        <div style={styles.formGroup}><label style={styles.label}>Secret Key</label><input type="password" style={styles.input} value={newApiKey} onChange={e => setNewApiKey(e.target.value)} placeholder="sk-..." required /></div>
                    </div>
                    <div style={styles.cardFooter}><button type="submit" style={styles.buttonPrimary} disabled={isSaving}>{isSaving ? 'Saving...' : 'Save Credentials'}</button></div>
                </form>
            </div>
            <div style={styles.card}>
                <div style={styles.cardHeader}><Key size={20} /> Configured Credentials</div>
                {isLoading ? <div style={styles.loadingContainer}>Loading...</div> : apiKeys.length === 0 ? <p style={styles.p}>No API keys configured.</p> : apiKeys.map(key => (
                    <div key={key.ID} style={styles.personaItem}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem' }}><span style={{ ...styles.statusIndicator, backgroundColor: key.STATUS === 'ACTIVE' ? 'var(--success)' : 'var(--muted)' }}></span><div><h3 style={styles.personaName}>{key.LLM_PROVIDER} <span style={styles.apiKeyTypeChip}>{key.LLM_PROVIDER_TYPE}</span></h3><p style={{ ...styles.personaPrompt, fontFamily: 'monospace' }}>{maskApiKey(key.API_KEY)}</p></div></div>
                        <div style={styles.personaActions}>
                            <button onClick={() => setEditingKey(key)} style={styles.iconButton} title="Edit Key"><FileEdit size={18} /></button>
                            <button onClick={() => handleDeleteKey(key.ID)} style={styles.iconButtonDanger} title="Revoke Key"><Trash2 size={18} /></button>
                        </div>
                    </div>
                ))}
            </div>
            {editingKey && <ApiKeyEditModal keyData={editingKey} onSave={handleUpdateKey} onCancel={() => setEditingKey(null)} />}
        </div>
    );
};