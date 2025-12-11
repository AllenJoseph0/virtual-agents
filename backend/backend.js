// backend.js
/* eslint-disable no-console */
const express = require('express');
const multer = require('multer');
const path = require('path');
const fsp = require('fs').promises;
const fs = require('fs');
const axios = require('axios');
const cors = require('cors');
const os = require('os');
const winston = require('winston');
const FormData = require('form-data');
const mysql = require('mysql2/promise');
require('dotenv').config();

// ============================================================================
// 1) LOGGER & DIRECTORIES
// ============================================================================
const logDir = path.join(__dirname, 'logs');
if (!fs.existsSync(logDir)) fs.mkdirSync(logDir, { recursive: true });

const logger = winston.createLogger({
  level: 'info',
  format: winston.format.combine(winston.format.timestamp(), winston.format.json()),
  transports: [
    new winston.transports.File({
      filename: path.join(logDir, 'backend.log'),
      maxsize: 5 * 1024 * 1024,
      maxFiles: 5,
    }),
    new winston.transports.Console({ format: winston.format.simple() }),
  ],
});

const CHAT_HISTORY_DIR = path.join(__dirname, 'chat_histories');
const DB_DIR = path.join(__dirname, 'db');
const USERS_DB_PATH = path.join(DB_DIR, 'users.json');
const PERMISSIONS_DB_PATH = path.join(DB_DIR, 'permissions.json');
const RULEBOOKS_DB_PATH = path.join(DB_DIR, 'rulebooks.json');
const COMPLIANCE_DB_PATH = path.join(DB_DIR, 'compliance.json');
const TEST_QUESTIONS_DB_PATH = path.join(DB_DIR, 'test_questions.json');
const SHARES_DB_PATH = path.join(DB_DIR, 'shares.json');
const UPLOAD_FOLDER = path.join(__dirname, 'data', 'uploads');

fs.mkdirSync(CHAT_HISTORY_DIR, { recursive: true });
fs.mkdirSync(DB_DIR, { recursive: true });
fs.mkdirSync(UPLOAD_FOLDER, { recursive: true });

// ============================================================================
// 2) EXPRESS, CORS & DB
// ============================================================================
const app = express();

const allowedOrigins = [
  'http://192.168.18.15:8251',
  'http://localhost:8250',
];

const corsOptions = {
  origin(origin, cb) {
    if (!origin) return cb(null, true);
    if (allowedOrigins.includes(origin)) return cb(null, true);
    // Allow for development convenience; remove in strict prod
    return cb(null, true); 
  },
  optionsSuccessStatus: 200,
};

app.use(cors(corsOptions));
app.use(express.json({ limit: '20mb' }));

const AI_SERVER_URL = process.env.AI_SERVER_URL;

const dbPool = mysql.createPool({
  host: process.env.DB_HOST,
  user: process.env.DB_USER,
  password: process.env.DB_PASSWORD,
  database: process.env.DB_DATABASE,
  port: process.env.DB_PORT || 3306,
  waitForConnections: true,
  connectionLimit: 10,
  queueLimit: 0,
}).on('error', (err) => {
  logger.error('MySQL Pool Error', { error: err.message });
});


// ============================================================================
// 3) FS HELPERS
// ============================================================================
const readJsonDb = (filePath) => async () => {
  try {
    if (!fs.existsSync(filePath)) {
      await fsp.writeFile(filePath, JSON.stringify(filePath.endsWith('users.json') ? [] : {}));
      return filePath.endsWith('users.json') ? [] : {};
    }
    const data = await fsp.readFile(filePath, 'utf8');
    return JSON.parse(data || (filePath.endsWith('users.json') ? '[]' : '{}'));
  } catch (e) {
    logger.error(`readJsonDb failed for ${filePath}`, { error: e.message });
    return filePath.endsWith('users.json') ? [] : {};
  }
};

const writeJsonDb = (filePath) => async (data) => {
  try {
    await fsp.writeFile(filePath, JSON.stringify(data, null, 2));
  } catch (e) {
    logger.error(`writeJsonDb failed for ${filePath}`, { error: e.message });
  }
};

const readUsers = readJsonDb(USERS_DB_PATH);
const writeUsers = writeJsonDb(USERS_DB_PATH);
const readPermissions = readJsonDb(PERMISSIONS_DB_PATH);
const writePermissions = writeJsonDb(PERMISSIONS_DB_PATH);
const readRulebooks = readJsonDb(RULEBOOKS_DB_PATH);
const writeRulebooks = writeJsonDb(RULEBOOKS_DB_PATH);
const readCompliance = readJsonDb(COMPLIANCE_DB_PATH);
const writeCompliance = writeJsonDb(COMPLIANCE_DB_PATH);
const readTestQuestions = readJsonDb(TEST_QUESTIONS_DB_PATH);
const writeTestQuestions = writeJsonDb(TEST_QUESTIONS_DB_PATH);
const readShares = readJsonDb(SHARES_DB_PATH);
const writeShares = writeJsonDb(SHARES_DB_PATH);


const clearAiServerCache = async () => {
  try {
    await axios.post(`${AI_SERVER_URL}/clear-cache`);
    logger.info('Cleared AI server retriever cache.');
  } catch (e) {
    logger.warn('Failed to clear AI server cache', { error: e.response?.data || e.message });
  }
};

// ============================================================================
// 4) GENERIC PROXY
// ============================================================================
const proxyToAiServer =
  (route, method = 'post', shouldClearCache = false) =>
    async (req, res) => {
      const finalRoute = Object.keys(req.params).reduce(
        (acc, key) => acc.replace(`:${key}`, encodeURIComponent(req.params[key])),
        route
      );
      const endpoint = `${AI_SERVER_URL}/${finalRoute}`;
      logger.info('Proxying request to AI server', { method: method.toUpperCase(), endpoint });

      try {
        const isStream = route.includes('tts') || route.includes('demo') || route.includes('greeting');
        const config = {
          method,
          url: endpoint,
          data: req.body,
          params: req.query,
          ...(isStream ? { responseType: 'stream' } : {}),
        };

        const resp = await axios(config);

        if (shouldClearCache) await clearAiServerCache();

        if (isStream) {
          res.setHeader('Content-Type', resp.headers['content-type'] || 'audio/wav');
          if (resp.headers['content-disposition']) {
            res.setHeader('Content-Disposition', resp.headers['content-disposition']);
          }
          if (resp.headers['content-length']) {
            res.setHeader('Content-Length', resp.headers['content-length']);
          }
          resp.data.pipe(res);
        } else {
          res.status(resp.status).json(resp.data);
        }
      } catch (e) {
        const status = e.response?.status || 500;
        const payload = e.response?.data || { error: 'Internal AI server error' };
        logger.error(`Proxy for '${route}' failed`, { status, error: payload });
        return res.status(status).json(payload);
      }
    };

// ============================================================================
// 5) GUARD: prevent concurrent jobs
// ============================================================================
const activeJobs = new Map();
const JOB_TIMEOUT_MS = 10 * 60 * 1000;

const keyFor = (u, c, action) => `${u}::${c}::${action}`;

const guardedHandler = (action) => (req, res, next) => {
  // Relaxed guard for browser tasks which might not have 'username/category' in standard format
  if (action === 'browser-task') {
      return next();
  }

  const { username, category } = req.body || {};
  if (!username || !category) {
    return res.status(400).json({ error: 'username and category are required' });
  }

  const key = keyFor(username, category, action);
  const existingJob = activeJobs.get(key);

  if (existingJob) {
    if (Date.now() - existingJob.timestamp > JOB_TIMEOUT_MS) {
      logger.warn('Stale job found and removed', { key });
      activeJobs.delete(key);
    } else {
      return res
        .status(409)
        .json({ error: `Another '${action}' is already running for ${username}/${category}. Try again shortly.` });
    }
  }

  activeJobs.set(key, { timestamp: Date.now() });
  res.once('finish', () => activeJobs.delete(key));
  next();
};

// ============================================================================
// 6) FILE UPLOAD
// ============================================================================
const storage = multer.diskStorage({
  destination: async (req, file, cb) => {
    const { username, category } = req.body || {};
    if (!username || !category) return cb(new Error('Username and category are required for upload.'));
    const dir = path.join(UPLOAD_FOLDER, username, category);
    try {
      await fsp.mkdir(dir, { recursive: true });
      cb(null, dir);
    } catch (err) {
      cb(err);
    }
  },
  filename: (_req, file, cb) => cb(null, Buffer.from(file.originalname, 'latin1').toString('utf8')),
});
const upload = multer({ storage });

app.post('/api/rag/upload', upload.array('files'), async (req, res) => {
  try {
    const { username, category } = req.body || {};
    if (!username || !category) return res.status(400).json({ error: 'username and category are required' });

    const permissions = await readPermissions();
    const categoryId = `${username}-${category}`;
    if (!permissions[categoryId]) {
      permissions[categoryId] = {
        owner: username,
        categoryName: category,
        business: false,
        basic: false,
        personaId: null,
        complianceProfileId: null
      };
      await writePermissions(permissions);
      logger.info('Created new default permissions for category', { categoryId });
    }

    logger.info('File upload successful', { user: username, category, files: req.files?.length || 0 });
    await clearAiServerCache();

    res.json({
      message: `Successfully uploaded ${req.files.length} files.`,
      files: (req.files || []).map((f) => f.originalname),
    });
  } catch (e) {
    logger.error('Upload failed', { error: e.message });
    res.status(500).json({ error: 'Upload failed' });
  }
});

// ============================================================================
// 7) VOICE STT / TTS / DEMO
// ============================================================================
const memoryUpload = multer({
  storage: multer.memoryStorage(),
  limits: { fileSize: 25 * 1024 * 1024 },
});

app.post('/api/voice/stt', memoryUpload.single('audio'), async (req, res) => {
  if (!req.file) return res.status(400).json({ error: 'No audio file provided' });
  try {
    const form = new FormData();
    form.append('audio', req.file.buffer, {
      filename: req.file.originalname || 'audio.webm',
      contentType: req.file.mimetype || 'audio/webm',
    });
    if (req.body.provider) {
      form.append('provider', req.body.provider);
    }
    if (req.body.firm_id) {
      form.append('firm_id', req.body.firm_id);
    }
    const resp = await axios.post(`${AI_SERVER_URL}/voice/stt`, form, { headers: form.getHeaders() });
    res.json(resp.data);
  } catch (e) {
    logger.error('STT proxy failed', { error: e.response?.data || e.message });
    res.status(e.response?.status || 500).json(e.response?.data || { error: 'Speech-to-text failed' });
  }
});

app.post('/api/voice/tts', proxyToAiServer('voice/tts'));
app.post('/api/voice/demo', proxyToAiServer('voice/demo'));
app.post('/api/voice/greeting', proxyToAiServer('voice/greeting'));
app.get('/api/voice/list-voices', proxyToAiServer('voice/list-voices', 'get'));
app.get('/api/voice/list-google-voices', proxyToAiServer('voice/list-google-voices', 'get'));
app.get('/api/voice/list-elevenlabs-voices', proxyToAiServer('voice/list-elevenlabs-voices', 'get'));
app.get('/api/voice/list-deepgram-voices', proxyToAiServer('voice/list-deepgram-voices', 'get'));


// ============================================================================
// 8) RAG MGMT & DATA
// ============================================================================
app.post('/api/rag/create-index', guardedHandler('create-index'), proxyToAiServer('create-index', 'post', true));
app.post('/api/rag/update-index', guardedHandler('update-index'), proxyToAiServer('update-index', 'post', true));
app.post('/api/rag/delete-index', guardedHandler('delete-index'), proxyToAiServer('delete-index', 'post', true));

app.post('/api/rag/delete-category', guardedHandler('delete-category'), async (req, res) => {
  const { username, category } = req.body;
  if (!username || !category) {
    return res.status(400).json({ error: 'username and category are required' });
  }
  try {
    const permissions = await readPermissions();
    const categoryId = `${username}-${category}`;
    const permEntry = permissions[categoryId];

    if (permEntry && permEntry.personaId) {
      try {
        await axios.delete(`${AI_SERVER_URL}/personas/${permEntry.personaId}`);
      } catch (err) {
        logger.warn('Failed to delete linked persona on AI server', { error: err.message });
      }
    }

    if (permissions[categoryId]) {
      delete permissions[categoryId];
      await writePermissions(permissions);
    }
    const rulebooks = await readRulebooks();
    if (rulebooks[categoryId]) {
      delete rulebooks[categoryId];
      await writeRulebooks(rulebooks);
    }
    const testQuestions = await readTestQuestions();
    if (testQuestions[categoryId]) {
      delete testQuestions[categoryId];
      await writeTestQuestions(testQuestions);
    }

    await axios.post(`${AI_SERVER_URL}/delete-category`, { username, category });

    const categoryDir = path.join(UPLOAD_FOLDER, username, category);
    if (fs.existsSync(categoryDir)) {
      await fsp.rm(categoryDir, { recursive: true, force: true });
    }

    await clearAiServerCache();
    logger.info(`Agent '${category}' deleted along with linked resources.`);
    res.json({ message: `Agent '${category}' and its linked persona have been permanently deleted.` });

  } catch (e) {
    logger.error('Delete category cascade failed', { error: e.message });
    res.status(500).json({ error: 'Failed to delete agent and resources.' });
  }
});

app.delete('/api/rag/file', guardedHandler('delete-file'), async (req, res) => {
  const { username, category, filename } = req.body;
  if (!username || !category || !filename) {
    return res.status(400).json({ error: 'username, category, and filename are required' });
  }
  try {
    const filePath = path.join(UPLOAD_FOLDER, username, category, filename);
    try { await fsp.access(filePath); } catch (err) { return res.status(404).json({ error: 'File not found.' }); }
    await fsp.unlink(filePath);
    await clearAiServerCache();
    logger.info('File deleted successfully', { username, category, filename });
    res.json({ message: `File '${filename}' deleted.` });
  } catch (e) {
    logger.error('Failed to delete file', { error: e.message });
    res.status(500).json({ error: 'Failed to delete file.' });
  }
});

app.get('/api/rag/structure', async (req, res) => {
  try {
    const { username } = req.query;
    if (!username) return res.status(400).json({ error: 'Username query parameter is required' });

    const resp = await axios.get(`${AI_SERVER_URL}/structure/${encodeURIComponent(username)}`);
    const structureData = resp.data;
    const allPermissions = await readPermissions();
    let permissionsModified = false;

    if (structureData && structureData[username]) {
      structureData[username] = structureData[username].map(category => {
        const categoryId = `${username}-${category.name}`;
        let categoryPermissions = allPermissions[categoryId];

        if (!categoryPermissions) {
          permissionsModified = true;
          allPermissions[categoryId] = {
            owner: username,
            categoryName: category.name,
            business: false,
            basic: false,
            personaId: null,
            complianceProfileId: null
          };
          categoryPermissions = allPermissions[categoryId];
        }

        return {
          ...category,
          permissions: {
            business: categoryPermissions.business,
            basic: categoryPermissions.basic
          },
          personaId: categoryPermissions.personaId || null,
          complianceProfileId: categoryPermissions.complianceProfileId || null
        };
      });

      if (permissionsModified) {
        await writePermissions(allPermissions);
      }
    }
    res.json(structureData);
  } catch (e) {
    logger.error('Structure endpoint failed', { error: e.response?.data || e.message });
    res.status(e.response?.status || 500).json({ error: 'Failed to fetch data structure' });
  }
});

// ============================================================================
// 9) CHAT HISTORY
// ============================================================================
const getHistoryFilePath = async (userId, userRole, category) => {
  const userHistoryDir = path.join(CHAT_HISTORY_DIR, encodeURIComponent(userRole), encodeURIComponent(userId));
  await fsp.mkdir(userHistoryDir, { recursive: true });
  const safeFilename = `${encodeURIComponent(category)}.json`;
  return path.join(userHistoryDir, safeFilename);
};

app.get('/api/chat/history/:username/:role/:category', async (req, res) => {
  const { username, role, category } = req.params;
  try {
    const filePath = await getHistoryFilePath(username, role, category);
    if (fs.existsSync(filePath)) {
      const data = await fsp.readFile(filePath, 'utf8');
      res.json(JSON.parse(data));
    } else {
      res.json([]);
    }
  } catch (e) {
    logger.error('Failed to read chat history', { username, role, category, error: e.message });
    res.status(500).json({ error: 'Failed to retrieve chat history.' });
  }
});

app.post('/api/chat/history/:username/:role/:category', async (req, res) => {
  const { username, role, category } = req.params;
  const chatHistory = req.body;
  if (!Array.isArray(chatHistory)) {
    return res.status(400).json({ error: 'Request body must be a chat history array.' });
  }
  try {
    const filePath = await getHistoryFilePath(username, role, category);
    await fsp.writeFile(filePath, JSON.stringify(chatHistory, null, 2));
    res.status(200).json({ message: 'History saved successfully.' });
  } catch (e) {
    logger.error('Failed to write chat history', { username, role, category, error: e.message });
    res.status(500).json({ error: 'Failed to save chat history.' });
  }
});

// ============================================================================
// 10) CORE RAG QUERY
// ============================================================================
app.post('/api/rag/query', async (req, res) => {
  const { owner_id, category, question, session_id, queried_by_id, queried_by_role, persona_id, firmId } = req.body || {};
  const finalSessionId = session_id || `${queried_by_id}-${category}-${persona_id}`;
  const permissions = await readPermissions();
  const categoryId = `${owner_id}-${category}`;
  const categorySettings = permissions[categoryId] || {};
  const complianceProfileId = categorySettings.complianceProfileId || null;

  let complianceRules = null;
  if (complianceProfileId) {
    const complianceProfiles = await readCompliance();
    const profile = complianceProfiles[complianceProfileId];
    if (profile) complianceRules = profile.content;
  }

  try {
    const rulebooks = await readRulebooks();
    const rulebookKey = `${owner_id}-${category}`;
    let rulebookPayload = null;
    const rawRulebook = rulebooks[rulebookKey];
    if (rawRulebook && typeof rawRulebook === 'string' && rawRulebook.trim() !== '') {
      try { rulebookPayload = JSON.parse(rawRulebook); } catch (e) { rulebookPayload = null; }
    }

    const resp = await axios.post(`${AI_SERVER_URL}/rag/chain`, {
      ...req.body,
      session_id: finalSessionId,
      rulebook: rulebookPayload,
      compliance_rules: complianceRules,
      firm_id: firmId,
    });
    res.json(resp.data);
  } catch (e) {
    logger.error('RAG query proxy failed', { error: e.response?.data || e.message });
    res.status(e.response?.status || 500).json(e.response?.data || { error: 'Failed to process query' });
  }
});


// ============================================================================
// 11) USERS & EMPLOYEES
// ============================================================================
app.post('/api/users/sync', async (req, res) => {
  const { id, name, role } = req.body || {};
  if (!id || !name || !role) return res.status(400).json({ error: 'id, name, role required' });

  const users = await readUsers();
  const found = users.find((u) => u.id === id);
  if (!found) {
    const newUser = { id, name, role };
    users.push(newUser);
    await writeUsers(users);
    return res.status(201).json(newUser);
  }
  res.json(found);
});

app.get('/api/employees', async (req, res) => {
  const { excludeId } = req.query;
  try {
    let sql = `SELECT EMPID, EMPNAME, STATUS, FIRM_ID, TYPE FROM EMPLOY_REGISTRATION WHERE STATUS = 'Active' AND TYPE = 'Employe'`;
    const params = [];
    if (excludeId) {
      sql += ' AND EMPID != ?';
      params.push(excludeId);
    }
    const [rows] = await dbPool.query(sql, params);
    res.json(rows);
  } catch (e) {
    logger.error('Failed to fetch employees', { error: e.message });
    res.status(500).json({ error: 'Failed to fetch employee list.' });
  }
});


// ============================================================================
// 12) PERSONA MGMT
// ============================================================================
app.get('/api/personas', proxyToAiServer('personas', 'get'));
app.post('/api/personas', proxyToAiServer('personas', 'post'));
app.put('/api/personas/:persona_id', proxyToAiServer('personas/:persona_id', 'put'));
app.delete('/api/personas/:persona_id', proxyToAiServer('personas/:persona_id', 'delete'));

// ============================================================================
// 13) COMPLIANCE MGMT
// ============================================================================
app.get('/api/compliance', async (req, res) => {
  try {
    const profiles = await readCompliance();
    res.json(Object.values(profiles));
  } catch (e) {
    res.status(500).json({ error: 'Failed to get compliance profiles' });
  }
});

app.post('/api/compliance', async (req, res) => {
  const { id, name, content } = req.body;
  if (!id || !name || typeof content === 'undefined') return res.status(400).json({ error: 'id, name, and content are required.' });
  try {
    const profiles = await readCompliance();
    profiles[id] = { id, name, content };
    await writeCompliance(profiles);
    res.status(201).json(profiles[id]);
  } catch (e) {
    res.status(500).json({ error: 'Failed to save compliance profile' });
  }
});

app.delete('/api/compliance/:profile_id', async (req, res) => {
  const { profile_id } = req.params;
  try {
    const profiles = await readCompliance();
    if (profile_id in profiles) {
      delete profiles[profile_id];
      await writeCompliance(profiles);
    }
    res.status(204).send();
  } catch (e) {
    res.status(500).json({ error: 'Failed to delete compliance profile' });
  }
});

// ============================================================================
// 14) PERMISSIONS, RULEBOOKS & SETTINGS
// ============================================================================
app.put('/api/permissions/category', async (req, res) => {
  const { adminId, category, roleToUpdate, hasAccess } = req.body;
  try {
    const categoryId = `${adminId}-${category}`;
    const permissions = await readPermissions();
    if (!permissions[categoryId]) {
      permissions[categoryId] = { owner: adminId, categoryName: category, business: false, basic: false };
    }
    permissions[categoryId][roleToUpdate] = hasAccess;
    await writePermissions(permissions);
    res.status(200).json({ message: 'Permissions updated successfully' });
  } catch (e) {
    res.status(500).json({ error: 'Failed to update permissions.' });
  }
});

app.put('/api/category/settings', async (req, res) => {
  const { adminId, categoryName, settings } = req.body;
  try {
    const categoryId = `${adminId}-${categoryName}`;
    const permissions = await readPermissions();
    if (!permissions[categoryId]) return res.status(404).json({ error: 'Category not found.' });
    permissions[categoryId] = { ...permissions[categoryId], ...settings };
    await writePermissions(permissions);
    res.status(200).json({ message: 'Settings updated successfully' });
  } catch (e) {
    res.status(500).json({ error: 'Failed to update settings.' });
  }
});

app.get('/api/rag/rulebook/:adminId/:categoryName', async (req, res) => {
  const { adminId, categoryName } = req.params;
  try {
    const rulebooks = await readRulebooks();
    const rulebookKey = `${adminId}-${categoryName}`;
    res.status(200).json({ content: rulebooks[rulebookKey] || '' });
  } catch (e) {
    res.status(500).json({ error: 'Failed to retrieve rulebook.' });
  }
});

app.post('/api/rag/rulebook', async (req, res) => {
  const { adminId, category, rulebookContent } = req.body;
  try {
    const rulebooks = await readRulebooks();
    const rulebookKey = `${adminId}-${category}`;
    rulebooks[rulebookKey] = rulebookContent;
    await writeRulebooks(rulebooks);
    res.status(200).json({ message: 'Rulebook saved successfully.' });
  } catch (e) {
    res.status(500).json({ error: 'Failed to save rulebook.' });
  }
});

app.get('/api/rag/test-questions/:adminId/:categoryName', async (req, res) => {
  const { adminId, categoryName } = req.params;
  try {
    const allQuestions = await readTestQuestions();
    const key = `${adminId}-${categoryName}`;
    res.status(200).json({ questions: allQuestions[key] || '' });
  } catch (e) {
    res.status(500).json({ error: 'Failed to retrieve test questions.' });
  }
});

app.post('/api/rag/test-questions', async (req, res) => {
  const { adminId, category, questions } = req.body;
  try {
    const allQuestions = await readTestQuestions();
    const key = `${adminId}-${category}`;
    allQuestions[key] = questions;
    await writeTestQuestions(allQuestions);
    res.status(200).json({ message: 'Test questions saved successfully.' });
  } catch (e) {
    res.status(500).json({ error: 'Failed to save test questions.' });
  }
});

app.post('/api/rag/run-test', async (req, res) => {
  const { adminId, category, personaId, complianceProfileId, num_questions, firmId } = req.body;
  try {
    let complianceRules = null;
    if (complianceProfileId) {
      const complianceProfiles = await readCompliance();
      if (complianceProfiles[complianceProfileId]) complianceRules = complianceProfiles[complianceProfileId].content;
    }

    const aiServerPayload = {
      owner_id: adminId, category, persona_id: personaId, compliance_rules: complianceRules, num_questions: num_questions || 10, firmId
    };

    const resp = await axios.post(`${AI_SERVER_URL}/rag/run-test`, aiServerPayload, { timeout: 300000 });
    res.status(resp.status).json(resp.data);
  } catch (e) {
    logger.error('RAG test run proxy failed', { error: e.response?.data || e.message });
    res.status(e.response?.status || 500).json(e.response?.data || { error: 'Failed to run test' });
  }
});


app.get('/api/rag/viewable', async (req, res) => {
  const { userId, userRole } = req.query;
  try {
    const allPermissions = await readPermissions();
    let categoriesToStatusCheck = new Map();

    if (userRole === 'admin') {
      try {
        const ownResp = await axios.get(`${AI_SERVER_URL}/structure/${encodeURIComponent(userId)}`);
        const userCats = ownResp.data?.[userId] || [];
        userCats.forEach(cat => categoriesToStatusCheck.set(`${userId}-${cat.name}`, { name: cat.name, owner: userId }));
      } catch (e) { }
    }

    if (['business', 'basic'].includes(userRole)) {
      Object.values(allPermissions)
        .filter(perm => perm && perm[userRole] === true)
        .forEach(perm => {
          categoriesToStatusCheck.set(`${perm.owner}-${perm.categoryName}`, { name: perm.categoryName, owner: perm.owner });
        });
    }

    const shares = await readShares();
    const userShares = shares[userId] || [];
    userShares.forEach(share => {
      categoriesToStatusCheck.set(`${share.ownerId}-${share.categoryName}`, { name: share.categoryName, owner: String(share.ownerId) });
    });

    const categoryList = Array.from(categoriesToStatusCheck.values());
    if (categoryList.length === 0) return res.status(200).json([]);

    const statusResp = await axios.post(`${AI_SERVER_URL}/batch-status-check`, { categories: categoryList });
    const activeRagsWithStatus = (statusResp.data || []).filter(c => c.indexStatus === 'ACTIVE');

    const finalActiveRags = activeRagsWithStatus.map(rag => {
      const categoryId = `${rag.owner}-${rag.name}`;
      const permissions = allPermissions[categoryId];
      return {
        ...rag,
        personaId: permissions ? (permissions.personaId || null) : null,
        complianceProfileId: permissions ? (permissions.complianceProfileId || null) : null
      };
    });

    res.status(200).json(finalActiveRags);
  } catch (e) {
    res.status(500).json({ error: 'Failed to retrieve categories.' });
  }
});

// ============================================================================
// 15) API KEY MANAGER ENDPOINTS
// ============================================================================
const parseEnum = (colType) => {
  const match = colType.match(/^enum\((.*)\)$/);
  if (!match) return [];
  return match[1].split(',').map(item => item.replace(/'/g, ''));
};

app.get('/api/llm/options', async (req, res) => {
  try {
    const sql = `
            SELECT COLUMN_NAME, COLUMN_TYPE 
            FROM INFORMATION_SCHEMA.COLUMNS
            WHERE TABLE_SCHEMA = ? 
            AND TABLE_NAME = 'LLM_DETAILS' 
            AND COLUMN_NAME IN ('LLM_PROVIDER', 'LLM_PROVIDER_TYPE')
        `;
    const [rows] = await dbPool.query(sql, [process.env.DB_DATABASE]);
    const options = { providers: [], types: [] };
    rows.forEach(row => {
      if (row.COLUMN_NAME === 'LLM_PROVIDER') {
        const allProviders = parseEnum(row.COLUMN_TYPE);
        options.providers = allProviders.filter(p => ['GROQ', 'GEMINI', 'GOOGLE_TTS', 'ELEVENLABS', 'DEEPGRAM'].includes(p));
      } else if (row.COLUMN_NAME === 'LLM_PROVIDER_TYPE') {
        options.types = parseEnum(row.COLUMN_TYPE);
      }
    });
    res.json(options);
  } catch (e) {
    res.status(500).json({ error: 'Failed to retrieve LLM options' });
  }
});

app.get('/api/llm/keys', async (req, res) => {
  const { userId, firmId } = req.query;
  try {
    const sql = `SELECT ID, LLM_PROVIDER, LLM_PROVIDER_TYPE, API_KEY, STATUS FROM LLM_DETAILS WHERE USERID = ? AND FIRMID = ?`;
    const [rows] = await dbPool.query(sql, [userId, firmId]);
    res.json(rows);
  } catch (e) {
    res.status(500).json({ error: 'Failed to retrieve API keys.' });
  }
});

app.post('/api/llm/keys', async (req, res) => {
  const { userId, firmId, llmProvider, llmProviderType, apiKey } = req.body;
  try {
    const sql = `INSERT INTO LLM_DETAILS (USERID, FIRMID, LLM_PROVIDER, LLM_PROVIDER_TYPE, API_KEY) VALUES (?, ?, ?, ?, ?)`;
    await dbPool.query(sql, [userId, firmId, llmProvider, llmProviderType, apiKey]);
    res.status(201).json({ message: 'API key saved successfully.' });
  } catch (e) {
    res.status(500).json({ error: 'Failed to save API key.' });
  }
});

app.put('/api/llm/keys/:id', async (req, res) => {
  const { id } = req.params;
  const { userId, firmId, API_KEY, STATUS } = req.body;
  try {
    const sql = 'UPDATE LLM_DETAILS SET API_KEY = ?, STATUS = ? WHERE ID = ? AND USERID = ? AND FIRMID = ?';
    const [result] = await dbPool.query(sql, [API_KEY, STATUS, id, userId, firmId]);
    if (result.affectedRows > 0) res.status(200).json({ message: 'API key updated.' });
    else res.status(404).json({ error: 'API key not found or you do not have permission to edit it.' });
  } catch (e) {
    res.status(500).json({ error: 'Failed to update API key.' });
  }
});

app.delete('/api/llm/keys/:id', async (req, res) => {
  const { id } = req.params;
  const { userId, firmId } = req.body;
  try {
    const sql = 'DELETE FROM LLM_DETAILS WHERE ID = ? AND USERID = ? AND FIRMID = ?';
    const [result] = await dbPool.query(sql, [id, userId, firmId]);
    if (result.affectedRows > 0) res.status(200).json({ message: 'API key deleted successfully.' });
    else res.status(404).json({ error: 'API key not found or you do not have permission to delete it.' });
  } catch (e) {
    res.status(500).json({ error: 'Failed to delete API key.' });
  }
});

// ============================================================================
// 16) SHARING ENDPOINTS
// ============================================================================
app.get('/api/rag/shares/:ownerId', async (req, res) => {
  const { ownerId } = req.params;
  try {
    const allShares = await readShares();
    const ownerShares = {};
    for (const granteeId in allShares) {
      const sharesFromOwner = allShares[granteeId].filter(share => String(share.ownerId) === ownerId);
      if (sharesFromOwner.length > 0) ownerShares[granteeId] = sharesFromOwner;
    }
    res.json(ownerShares);
  } catch (e) {
    res.status(500).json({ error: 'Could not retrieve sharing information.' });
  }
});

app.post('/api/rag/share', async (req, res) => {
  const { ownerId, categoryName, granteeId } = req.body;
  if (String(ownerId) === String(granteeId)) return res.status(400).json({ error: 'You cannot share a knowledge base with yourself.' });
  try {
    const shares = await readShares();
    if (!shares[granteeId]) shares[granteeId] = [];
    const numOwnerId = Number(ownerId);
    const alreadyExists = shares[granteeId].some(share => share.ownerId === numOwnerId && share.categoryName === categoryName);
    if (alreadyExists) return res.status(409).json({ error: 'This knowledge base is already shared with this user.' });
    shares[granteeId].push({ ownerId: numOwnerId, categoryName });
    await writeShares(shares);
    res.status(201).json({ message: 'Knowledge base shared successfully.' });
  } catch (e) {
    res.status(500).json({ error: 'Failed to share the knowledge base.' });
  }
});

app.delete('/api/rag/share', async (req, res) => {
  const { ownerId, categoryName, granteeId } = req.body;
  try {
    const shares = await readShares();
    if (shares[granteeId]) {
      const numOwnerId = Number(ownerId);
      shares[granteeId] = shares[granteeId].filter(share => !(share.ownerId === numOwnerId && share.categoryName === categoryName));
      if (shares[granteeId].length === 0) delete shares[granteeId];
    }
    await writeShares(shares);
    res.status(200).json({ message: 'Access revoked.' });
  } catch (e) {
    res.status(500).json({ error: 'Failed to revoke access.' });
  }
});

// ============================================================================
// 17) BROWSER AGENT PROXY
// ============================================================================
app.post('/api/agent/browser-task', guardedHandler('browser-task'), proxyToAiServer('agent/browser-task', 'post'));

// ============================================================================
// 18) SERVER
// ============================================================================
app.get('/', (_req, res) => {
  res.status(200).send(`<h1>RAG System Backend is running.</h1><p>AI Server Target: ${AI_SERVER_URL}</p>`);
});

const PORT = process.env.PORT || 8251;
const HOST = '0.0.0.0';

const getLocalIps = () => {
  const nets = os.networkInterfaces();
  const results = [];
  for (const name of Object.keys(nets)) {
    for (const net of nets[name]) {
      if (net.family === 'IPv4' && !net.internal) {
        results.push(net.address);
      }
    }
  }
  return results.length > 0 ? results : ['not found'];
};

app.listen(PORT, HOST, () => {
  const localIps = getLocalIps();
  logger.info('RAG backend server started.', { port: PORT, host: HOST, localIps, aiServer: AI_SERVER_URL });
  console.log(`RAG Backend Server listening on port ${PORT}`);
});