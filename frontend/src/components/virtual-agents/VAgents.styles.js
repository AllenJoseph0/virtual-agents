//rag.styles.js

const styles = {
  // --- APP SHELL LAYOUT ---
  appContainer: {
    background: 'var(--background)',
    color: 'var(--foreground)',
    display: 'flex',
    flexDirection: 'column',
    height: '100vh',
    overflow: 'hidden',
    paddingBottom: 'var(--footer-height, 0px)',
  },
  navbar: {
    display: 'flex',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: '0.75rem 2rem',
    background: 'var(--card)',
    borderBottom: '1px solid var(--border)',
    flexShrink: 0,
    zIndex: 1000,
    height: '73px',
    boxSizing: 'border-box'
  },
  mainContent: {
    flexGrow: 1,
    overflowY: 'auto',
    display: 'flex',
    flexDirection: 'column',
    position: 'relative',
    paddingBottom: '2rem'
  },

  // --- NAVIGATION ---
  navLeft: { flex: 1, display: 'flex', alignItems: 'center', gap: '1rem' },
  navCenter: { flex: 2, display: 'flex', justifyContent: 'center' },
  navRight: { flex: 1, display: 'flex', justifyContent: 'flex-end', alignItems: 'center', gap: '0.5rem' },
  navButtonGroup: { display: 'flex', gap: '0.25rem', background: 'var(--secondary)', padding: '4px', borderRadius: 'var(--radius)', border: '1px solid var(--border)' },
  navTitle: { margin: 0, fontSize: '1.25rem', fontWeight: 700, letterSpacing: '-0.5px', color: 'var(--foreground-heavy)' },
  navButton: { padding: '0.5rem 1rem', border: 'none', background: 'transparent', color: 'var(--muted-foreground)', borderRadius: 'var(--radius-sm)', cursor: 'pointer', transition: 'all 0.2s ease', fontWeight: 500, fontSize: '0.9rem' },
  navButtonActive: { padding: '0.5rem 1rem', border: 'none', background: 'var(--card)', color: 'var(--primary)', borderRadius: 'var(--radius-sm)', cursor: 'pointer', fontWeight: 600, boxShadow: 'var(--shadow-sm)', fontSize: '0.9rem' },

  // --- HEADERS & BADGES ---
  userBadge: { display: 'flex', flexDirection: 'column', alignItems: 'flex-end', lineHeight: 1.2 },
  loggedInAs: { color: 'var(--foreground-heavy)', fontWeight: 600, fontSize: '0.9rem' },
  roleBadge: { fontSize: '0.7rem', textTransform: 'uppercase', color: 'var(--muted-foreground)', fontWeight: 500, letterSpacing: '0.5px' },
  header: { textAlign: 'center', marginBottom: '2.5rem', marginTop: '3rem' },
  headerH2: { margin: '0 0 0.5rem 0', color: 'var(--foreground-heavy)', fontSize: '2rem', letterSpacing: '-1px' },
  headerSubtitle: { margin: 0, color: 'var(--muted-foreground)', fontSize: '1.1rem' },
  loadingContainer: { display: 'flex', alignItems: 'center', justifyContent: 'center', gap: '0.5rem', textAlign: 'center', padding: '4rem', color: 'var(--muted-foreground)' },

  // --- BUTTONS ---
  buttonPrimary: { background: 'var(--primary)', color: 'var(--primary-foreground)', padding: '0.65rem 1.25rem', border: 'none', borderRadius: 'var(--radius)', cursor: 'pointer', fontSize: '0.875rem', display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 600, transition: 'all 0.2s', boxShadow: 'var(--shadow-sm)' },
  buttonSuccess: { background: 'var(--success)', color: 'var(--primary-foreground)', border: 'none', padding: '0.6rem 1.1rem', borderRadius: 'var(--radius)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 600, fontSize: '0.875rem', transition: 'all 0.2s', boxShadow: 'var(--shadow-sm)' },
  buttonSecondary: { background: 'var(--card)', color: 'var(--foreground)', border: '1px solid var(--border)', padding: '0.6rem 1.1rem', borderRadius: 'var(--radius)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 500, fontSize: '0.875rem', transition: 'all 0.2s' },
  buttonWarning: { background: 'var(--warning)', color: '#000', border: 'none', padding: '0.6rem 1.1rem', borderRadius: 'var(--radius)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 600, fontSize: '0.875rem', transition: 'all 0.2s', boxShadow: 'var(--shadow-sm)' },
  buttonDanger: { background: 'var(--danger)', color: 'var(--primary-foreground)', border: 'none', padding: '0.6rem 1.1rem', borderRadius: 'var(--radius)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 600, fontSize: '0.875rem', transition: 'all 0.2s', boxShadow: 'var(--shadow-sm)' },
  buttonDangerOutline: { background: 'transparent', color: 'var(--danger)', border: '1px solid var(--danger)', padding: '0.6rem 1.1rem', borderRadius: 'var(--radius)', cursor: 'pointer', display: 'flex', alignItems: 'center', gap: '8px', fontWeight: 500, fontSize: '0.875rem', transition: 'all 0.2s' },
  buttonGroup: { display: 'flex', gap: '0.75rem', flexWrap: 'wrap', alignItems: 'center' },

  iconButton: {
    padding: '0.6rem',
    background: 'var(--background)',
    border: '1px solid var(--border)',
    color: 'var(--foreground)',
    cursor: 'pointer',
    borderRadius: 'var(--radius)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'all 0.2s'
  },
  iconButtonDanger: {
    padding: '0.6rem',
    background: 'rgba(217, 45, 32, 0.05)',
    border: '1px solid transparent',
    color: 'var(--danger)',
    cursor: 'pointer',
    borderRadius: 'var(--radius)',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    transition: 'all 0.2s'
  },

  // --- CARDS & INPUTS ---
  card: { background: 'var(--card)', border: '1px solid var(--border)', borderRadius: 'var(--radius-lg)', marginBottom: '1.5rem', overflow: 'hidden', boxShadow: 'var(--shadow-md)', transition: 'all 0.2s ease' },
  cardHeader: { padding: '1.25rem 1.75rem', background: 'var(--background)', borderBottom: '1px solid var(--border)', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '1rem', color: 'var(--foreground-heavy)', letterSpacing: '-0.01em' },
  cardBody: { padding: '1.75rem' },
  cardFooter: { padding: '1.25rem 1.75rem', background: 'var(--background)', borderTop: '1px solid var(--border)', display: 'flex', alignItems: 'center', gap: '0.75rem', flexWrap: 'wrap' },

  label: { display: 'block', marginBottom: '0.6rem', fontWeight: '600', color: 'var(--foreground-heavy)', fontSize: '0.875rem', letterSpacing: '-0.01em' },
  input: { width: '100%', padding: '0.75rem 1rem', fontSize: '0.95rem', border: '1px solid var(--border)', borderRadius: 'var(--radius)', boxSizing: 'border-box', background: 'var(--card)', color: 'var(--foreground)', transition: 'all 0.2s ease', outline: 'none' },
  textarea: { width: '100%', minHeight: '200px', padding: '1rem', fontSize: '1rem', border: '1px solid var(--border)', borderRadius: 'var(--radius)', boxSizing: 'border-box', resize: 'vertical', background: 'var(--background)', color: 'var(--foreground)', transition: 'all 0.2s ease', lineHeight: 1.5, outline: 'none', fontFamily: 'monospace' },
  fileInput: { width: '100%', padding: '0.75rem', fontSize: '1rem', boxSizing: 'border-box', marginTop: '0.5rem', border: '1px solid var(--border)', borderRadius: 'var(--radius)', background: 'var(--background)' },
  formGroup: { marginBottom: '1.25rem' },
  formHelperText: { fontSize: '0.85rem', color: 'var(--muted-foreground)', marginTop: '0.5rem' },

  // --- DASHBOARD GRID ---
  dashboardNav: { display: 'flex', gap: '1rem', borderBottom: '1px solid var(--border)', marginBottom: '2rem', flexWrap: 'wrap', paddingBottom: '0.5rem' },
  dashboardNavButton: { padding: '0.75rem 1rem', color: 'var(--muted-foreground)', border: 'none', background: 'transparent', cursor: 'pointer', borderBottom: '3px solid transparent', display: 'flex', alignItems: 'center', gap: '0.5rem', transition: 'color 0.2s, border-color 0.2s', fontSize: '1rem', fontWeight: 500 },
  dashboardNavButtonActive: { padding: '0.75rem 1rem', border: 'none', background: 'transparent', cursor: 'pointer', borderBottom: '3px solid var(--primary)', display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--primary)', fontWeight: 600, fontSize: '1rem' },
  agentGrid: { display: 'flex', flexDirection: 'column', gap: '1.5rem', width: '100%', maxWidth: '900px', margin: '0 auto', paddingBottom: '2rem' },

  // --- CATEGORY CARDS ---
  categoryHeader: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '1.5rem 1.75rem', background: 'var(--background)', borderBottom: '1px solid var(--border)' },
  categoryTitle: { fontWeight: '600', display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '1.05rem', color: 'var(--foreground-heavy)', letterSpacing: '-0.02em' },
  indexStatus: { color: '#fff', padding: '0.35rem 0.85rem', borderRadius: '99px', fontSize: '0.7rem', marginLeft: 'auto', textTransform: 'uppercase', fontWeight: 600, letterSpacing: '0.8px', boxShadow: 'var(--shadow-sm)' },

  // --- COMPACT FILE LIST ---
  fileListContainer: { maxHeight: '400px', overflowY: 'auto', border: '1px solid var(--border)', borderRadius: 'var(--radius)', background: 'var(--secondary)' },
  fileList: { listStyle: 'none', padding: 0, margin: 0, display: 'flex', flexDirection: 'column' },
  fileListItemCompact: { display: 'flex', alignItems: 'center', justifyContent: 'space-between', padding: '0.75rem 1rem', background: 'var(--card)', borderBottom: '1px solid var(--border)', fontSize: '0.9rem', transition: 'all 0.2s', gap: '1rem' },
  fileInfoCompact: { display: 'flex', alignItems: 'center', gap: '0.75rem', fontWeight: 500, color: 'var(--foreground)', flex: 1, overflow: 'hidden' },
  fileNameText: { whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis', maxWidth: '350px' },
  fileMetaCompact: { fontSize: '0.75rem', color: 'var(--muted-foreground)', display: 'flex', alignItems: 'center', gap: '1rem', flexShrink: 0 },
  emptyStateSmall: { padding: '2rem', textAlign: 'center', color: 'var(--muted-foreground)', fontStyle: 'italic', fontSize: '0.9rem' },

  // --- BENCHMARK RESULTS ---
  benchmarkContainer: { display: 'flex', flexDirection: 'column', gap: '1.5rem' },

  scoreCard: {
    background: 'linear-gradient(135deg, var(--card) 0%, var(--secondary) 100%)',
    borderRadius: 'var(--radius)',
    padding: '2rem',
    display: 'flex',
    alignItems: 'center',
    justifyContent: 'center',
    flexDirection: 'column',
    border: '1px solid var(--border)',
    marginBottom: '1rem',
    boxShadow: 'inset 0 0 20px rgba(0,0,0,0.02)'
  },
  scoreBig: { fontSize: '3.5rem', fontWeight: 800, color: 'var(--primary)', lineHeight: 1 },
  scoreLabel: { fontSize: '0.9rem', color: 'var(--muted-foreground)', textTransform: 'uppercase', letterSpacing: '1px', marginTop: '0.5rem', fontWeight: 600 },

  benchmarkConfig: { padding: '2rem', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: '1.5rem', textAlign: 'center' },
  benchmarkResultItem: { background: 'var(--background)', border: '1px solid var(--border)', borderRadius: 'var(--radius)', marginBottom: '1rem', padding: '1.25rem', position: 'relative', overflow: 'hidden' },
  benchmarkScoreBadge: {
    position: 'absolute',
    top: '1rem',
    right: '1rem',
    background: 'var(--foreground-heavy)',
    color: 'var(--background)',
    padding: '0.25rem 0.75rem',
    borderRadius: '99px',
    fontWeight: 700,
    fontSize: '0.85rem',
    boxShadow: 'var(--shadow-sm)'
  },
  benchmarkQuestion: { display: 'flex', alignItems: 'flex-start', gap: '0.5rem', fontWeight: 600, color: 'var(--foreground-heavy)', marginBottom: '0.75rem', fontSize: '0.95rem', paddingRight: '3rem' },
  benchmarkAnswer: { background: 'var(--chat-bg)', padding: '1rem', borderRadius: 'var(--radius-sm)', borderLeft: '3px solid var(--primary)', fontSize: '0.9rem', color: 'var(--foreground)', lineHeight: 1.6 },
  benchmarkReason: { fontSize: '0.8rem', color: 'var(--muted-foreground)', marginTop: '0.75rem', fontStyle: 'italic', display: 'flex', alignItems: 'center', gap: '0.5rem' },
  benchmarkLabel: { fontSize: '0.75rem', textTransform: 'uppercase', color: 'var(--muted-foreground)', letterSpacing: '0.5px', marginBottom: '0.25rem', fontWeight: 600 },

  // --- CHAT INTERFACE ---
  chatContainer: { display: 'flex', flexDirection: 'column', height: '100%', background: 'var(--chat-bg)', overflow: 'hidden', position: 'relative' },
  chatHeader: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '1rem 2rem', borderBottom: '1px solid var(--border)', flexShrink: 0, gap: '1rem', background: 'var(--card)', zIndex: 10 },
  chatHeaderInfo: { display: 'flex', alignItems: 'center', gap: '0.75rem', flexGrow: 1, justifyContent: 'center' },
  chatHeaderInfoChip: { display: 'flex', alignItems: 'center', gap: '0.5rem', color: 'var(--muted-foreground)', background: 'var(--secondary)', padding: '6px 12px', borderRadius: '99px', fontSize: '0.9rem', fontWeight: 500 },
  chatHistory: { flexGrow: 1, overflowY: 'auto', padding: '2rem', scrollBehavior: 'smooth' },
  chatHistoryContent: { maxWidth: '800px', width: '100%', margin: '0 auto' },
  chatInputContainer: { padding: '2rem', background: 'var(--chat-bg)', flexShrink: 0, borderTop: '1px solid var(--border)' },
  chatInputArea: { width: '100%', maxWidth: '800px', margin: '0 auto' },
  inputWrapper: { position: 'relative', display: 'flex', alignItems: 'center', background: 'var(--card)', borderRadius: '12px', padding: '0.75rem', boxShadow: 'var(--shadow-lg)', border: '1px solid var(--border)' },
  chatInput: { flexGrow: 1, padding: '0.5rem 1rem', border: 'none', resize: 'none', outline: 'none', background: 'transparent', color: 'var(--foreground)', fontSize: '1rem', lineHeight: '1.5', maxHeight: '150px' },

  // --- MESSAGES ---
  userMessage: { display: 'flex', justifyContent: 'flex-end', marginBottom: '1.5rem', gap: '1rem', marginLeft: 'auto', maxWidth: '100%', alignItems: 'flex-start' },
  aiMessage: { display: 'flex', justifyContent: 'flex-start', marginBottom: '1.5rem', gap: '1rem', marginRight: 'auto', maxWidth: '100%', alignItems: 'flex-start' },
  messageAvatar: { flexShrink: 0, borderRadius: '50%', width: '32px', height: '32px', display: 'flex', alignItems: 'center', justifyContent: 'center', background: 'var(--card)', color: 'var(--muted-foreground)', border: '1px solid var(--border)', marginTop: '4px', boxShadow: 'var(--shadow-sm)' },
  aiMessageContent: { padding: '1rem 1.5rem', borderRadius: '0 16px 16px 16px', background: 'var(--card)', wordBreak: 'break-word', lineHeight: 1.6, color: 'var(--foreground)', border: '1px solid var(--border)', boxShadow: 'var(--shadow-sm)' },
  userMessageContent: { padding: '1rem 1.5rem', borderRadius: '16px 0 16px 16px', background: 'var(--primary)', color: 'var(--primary-foreground)', wordBreak: 'break-word', lineHeight: 1.6, boxShadow: 'var(--shadow-md)' },
  sourcesContainer: { fontSize: '0.8rem', color: 'var(--muted-foreground)', marginTop: '0.75rem', borderTop: '1px solid var(--border)', paddingTop: '0.75rem' },

  // --- MODALS ---
  modalOverlay: { position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'rgba(16, 24, 40, 0.7)', backdropFilter: 'blur(8px)', display: 'flex', alignItems: 'center', justifyContent: 'center', zIndex: 3000, padding: '1rem' },
  modalContent: {
    background: 'var(--card)',
    color: 'var(--foreground)',
    padding: '0',
    borderRadius: 'var(--radius-lg)',
    position: 'relative',
    width: '95%',
    maxWidth: '650px',
    maxHeight: '90vh',
    display: 'flex',
    flexDirection: 'column',
    boxShadow: 'var(--shadow-lg)',
    overflow: 'hidden',
    border: '1px solid var(--border)'
  },
  modalHeader: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '1.25rem 2rem', borderBottom: '1px solid var(--border)', flexShrink: 0 },
  modalBody: { padding: '2rem', overflowY: 'auto', flexGrow: 1 },
  modalCloseButton: { background: 'transparent', border: 'none', color: 'var(--muted-foreground)', cursor: 'pointer', borderRadius: '50%', padding: '0.5rem', display: 'flex', alignItems: 'center', justifyContent: 'center' },
  modalTitle: { margin: '0', fontSize: '1.25rem', fontWeight: 600, color: 'var(--foreground-heavy)' },
  modalFooter: { display: 'flex', justifyContent: 'flex-end', gap: '0.75rem', padding: '1.25rem 2rem', borderTop: '1px solid var(--border)', background: 'var(--secondary)', flexShrink: 0 },

  // --- SEARCH ---
  searchWrapper: { position: 'relative', margin: '0 0 1rem 0' },
  searchIcon: { position: 'absolute', top: '50%', left: '1rem', transform: 'translateY(-50%)', color: 'var(--muted-foreground)' },
  searchInput: { width: '100%', padding: '0.75rem 1rem 0.75rem 2.5rem', borderRadius: 'var(--radius)', border: '1px solid var(--border)', background: 'var(--background)', color: 'var(--foreground)', fontSize: '0.95rem', boxSizing: 'border-box', outline: 'none' },

  // --- MISC ---
  alert: { padding: '1rem', border: '1px solid transparent', borderRadius: 'var(--radius)', display: 'flex', alignItems: 'center', gap: '0.5rem' },
  alertDanger: { color: '#b91c1c', background: '#fef2f2', borderColor: '#fecaca' },
  alertSuccess: { color: '#15803d', background: '#dcfce7', borderColor: '#bbf7d0' },
  p: { padding: '1rem 1.25rem', color: 'var(--muted-foreground)', textAlign: 'center' },
  spinner: { animation: 'spin 1s linear infinite' },

  // --- LIST ITEMS ---
  personaItem: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '1.25rem', borderBottom: '1px solid var(--border)', gap: '1rem' },
  personaName: { margin: 0, fontWeight: 600, fontSize: '1rem' },
  personaPrompt: { margin: '0.25rem 0 0', color: 'var(--muted-foreground)', fontSize: '0.85rem' },
  apiKeyTypeChip: { background: 'var(--secondary)', color: 'var(--muted-foreground)', padding: '0.2rem 0.5rem', borderRadius: '6px', fontSize: '0.7rem', marginLeft: '0.5rem', display: 'inline-block', fontWeight: 600, border: '1px solid var(--border)' },
  statusIndicator: { width: '8px', height: '8px', borderRadius: '50%', flexShrink: 0 },
  personaActions: { display: 'flex', gap: '0.5rem', flexShrink: 0 },

  // --- SETTINGS PANELS ---
  permissionsContainer: { borderTop: '1px solid var(--border)', marginTop: '0', background: 'var(--chat-bg)' },
  permissionsHeader: { padding: '1.25rem 1.75rem', background: 'var(--chat-bg)', fontWeight: '600', display: 'flex', alignItems: 'center', gap: '0.6rem', fontSize: '0.8rem', color: 'var(--muted-foreground)', textTransform: 'uppercase', letterSpacing: '0.8px' },
  permissionsBody: { padding: '0.75rem 1.75rem 1.5rem' },
  permissionUserRow: { display: 'flex', justifyContent: 'space-between', alignItems: 'center', padding: '0.75rem 0', fontSize: '0.9rem', color: 'var(--foreground)' },

  personaSelectorContainer: { borderTop: '1px solid var(--border)', background: 'var(--chat-bg)' },
  personaSelectorBody: { padding: '0.75rem 1.75rem 1.5rem', display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' },

  complianceContentPreview: { margin: '0.5rem 0 0', color: 'var(--muted-foreground)', fontSize: '0.85rem', whiteSpace: 'pre-wrap', background: 'var(--secondary)', padding: '1rem', borderRadius: 'var(--radius)', border: '1px solid var(--border)', maxHeight: '100px', overflow: 'hidden', fontFamily: 'monospace' },

  voiceSettingsSection: { marginBottom: '2rem' },
  voiceSettingsHeader: { fontWeight: 600, color: 'var(--foreground-heavy)', marginBottom: '1rem', paddingBottom: '0.5rem', borderBottom: `1px solid var(--border)`, fontSize: '0.9rem', textTransform: 'uppercase', letterSpacing: '0.5px' },
  providerToggleContainer: { display: 'flex', gap: '1rem', flexWrap: 'wrap' },
  providerButton: { flex: 1, background: 'var(--background)', border: '1px solid var(--border)', padding: '0.75rem 1rem', borderRadius: 'var(--radius)', cursor: 'pointer', fontWeight: 500, transition: 'all 0.2s', color: 'var(--foreground)' },
  providerButtonActive: { flex: 1, background: 'var(--primary-light)', border: '1px solid var(--primary)', padding: '0.75rem 1rem', borderRadius: 'var(--radius)', cursor: 'pointer', fontWeight: 600, transition: 'all 0.2s', color: 'var(--primary)', boxShadow: '0 0 0 2px var(--primary-light)' },

  voiceModeContainer: { position: 'fixed', top: 0, left: 0, right: 0, bottom: 0, background: 'linear-gradient(180deg, #101828 0%, #000000 100%)', zIndex: 2000, display: 'flex', alignItems: 'center', justifyContent: 'center' },
  voiceFullScreen: { display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'flex-end', height: '100%', width: '100%', color: 'white', padding: '2rem', boxSizing: 'border-box' },
  liveTranscriptContainer: { width: '100%', flex: 1, overflowY: 'auto', padding: '1rem 0', marginBottom: 'auto', maskImage: 'linear-gradient(to bottom, transparent, black 10%, black 90%, transparent)' },
  voiceStatusText: { fontSize: '1.2rem', margin: '1.5rem 0', color: '#98a2b3', height: '40px', fontWeight: 300, letterSpacing: '1px', textTransform: 'uppercase' },
  voiceMicIcon: { background: 'rgba(255,255,255,0.05)', color: '#98a2b3', border: '1px solid rgba(255,255,255,0.1)', width: '90px', height: '90px', borderRadius: '50%', display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'all 0.2s', userSelect: 'none', backdropFilter: 'blur(10px)' },
  voiceMicIconListening: { background: 'var(--primary)', color: 'white', border: 'none', animation: 'pulse 2s infinite', boxShadow: '0 0 40px var(--primary)' },
  voiceStopButton: { background: 'var(--danger)', color: 'white', border: 'none', width: '70px', height: '70px', borderRadius: '50%', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', transition: 'opacity 0.3s' },
  voiceExitButton: { position: 'absolute', top: '2rem', right: '2rem', background: 'rgba(255,255,255,0.1)', color: 'white', border: 'none', width: '48px', height: '48px', borderRadius: '50%', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', backdropFilter: 'blur(4px)' },

  sendButton: { width: '40px', height: '40px', borderRadius: '8px', border: 'none', background: 'var(--primary)', color: 'var(--primary-foreground)', cursor: 'pointer', display: 'flex', alignItems: 'center', justifyContent: 'center', flexShrink: 0, transition: 'all 0.2s' },
  sendButtonDisabled: { background: 'var(--secondary)', color: 'var(--muted-foreground)', cursor: 'not-allowed' },
};

export default styles;