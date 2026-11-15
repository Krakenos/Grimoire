/* Grimoire Management Panel — vanilla JS, no framework */

const BASE = '/panel/api';

// -------------------------------------------------------------------------
// State
// -------------------------------------------------------------------------
let currentUser = null;
let currentChat = null;
let network = null;   // vis.Network instance

// -------------------------------------------------------------------------
// Toast notifications
// -------------------------------------------------------------------------
const toast = document.getElementById('toast');
let toastTimer = null;

function showToast(msg, type = 'ok') {
  clearTimeout(toastTimer);
  toast.textContent = msg;
  toast.className = `visible ${type}`;
  toastTimer = setTimeout(() => { toast.className = ''; }, 3000);
}

// -------------------------------------------------------------------------
// API helpers
// -------------------------------------------------------------------------
const PANEL_KEY_STORAGE = 'grimoire_panel_key';

function getPanelKey() {
  return localStorage.getItem(PANEL_KEY_STORAGE) || '';
}

function authHeaders() {
  const key = getPanelKey();
  return key ? { Authorization: `Bearer ${key}` } : {};
}

// Prompts once for the panel key and retries `run`. If the retry still 401s, clears the
// stored key so the next request prompts again rather than looping silently.
async function withKeyPrompt(run) {
  try {
    return await run();
  } catch (e) {
    if (e.status !== 401) throw e;
    const key = window.prompt('Panel key required (Authorization: Bearer <key>):', getPanelKey());
    if (key === null) throw e;
    localStorage.setItem(PANEL_KEY_STORAGE, key);
    try {
      return await run();
    } catch (e2) {
      if (e2.status === 401) localStorage.removeItem(PANEL_KEY_STORAGE);
      throw e2;
    }
  }
}

async function api(method, path, body) {
  const opts = {
    method,
    headers: { ...authHeaders(), ...(body ? { 'Content-Type': 'application/json' } : {}) },
  };
  if (body) opts.body = JSON.stringify(body);
  return withKeyPrompt(async () => {
    const res = await fetch(BASE + path, opts);
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      const error = new Error(err.detail || res.statusText);
      error.status = res.status;
      throw error;
    }
    if (res.status === 204) return null;
    return res.json();
  });
}

const get   = (path)        => api('GET',    path);
const del   = (path)        => api('DELETE', path);
const patch = (path, body)  => api('PATCH',  path, body);
const post  = (path, body)  => api('POST',   path, body);
const put   = (path, body)  => api('PUT',    path, body);

async function uploadFile(path, file) {
  const form = new FormData();
  form.append('file', file);
  return withKeyPrompt(async () => {
    const res = await fetch(BASE + path, { method: 'POST', headers: authHeaders(), body: form });
    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      const error = new Error(err.detail || res.statusText);
      error.status = res.status;
      throw error;
    }
    return res.json();
  });
}

// -------------------------------------------------------------------------
// Sidebar: users + chats
// -------------------------------------------------------------------------
async function loadSidebar() {
  const tree = document.getElementById('sidebar-tree');
  tree.innerHTML = '';

  let users;
  try {
    users = await get('/users');
  } catch (e) {
    tree.innerHTML = `<div style="padding:12px;color:var(--danger);font-size:0.8rem;">Error: ${e.message}</div>`;
    return;
  }

  if (!users.length) {
    tree.innerHTML = '<div style="padding:12px;color:var(--text-muted);font-size:0.8rem;">No users found.</div>';
    return;
  }

  for (const user of users) {
    const uDiv = document.createElement('div');
    uDiv.className = 'tree-user';
    uDiv.textContent = `User ${user.id} — ${user.external_id}`;
    tree.appendChild(uDiv);

    let chats;
    try {
      chats = await get(`/users/${user.id}/chats`);
    } catch {
      chats = [];
    }

    for (const chat of chats) {
      const cDiv = document.createElement('div');
      cDiv.className = 'tree-chat';
      cDiv.textContent = chat.external_id;
      cDiv.title = `Chat ID ${chat.id}`;
      cDiv.dataset.userId = user.id;
      cDiv.dataset.chatId = chat.id;
      cDiv.addEventListener('click', () => selectChat(user, chat, cDiv));
      tree.appendChild(cDiv);
    }
  }
}

function selectChat(user, chat, el) {
  document.querySelectorAll('.tree-chat').forEach(c => c.classList.remove('active'));
  el.classList.add('active');
  currentUser = user;
  currentChat = chat;

  const noMsg   = document.getElementById('no-chat-msg');
  const panelV  = document.getElementById('panel-view');
  const lbView  = document.getElementById('lorebook-view');
  const settV   = document.getElementById('settings-view');
  if (noMsg) noMsg.style.display = 'none';
  lbView.style.display = 'none';
  settV.style.display = 'none';
  panelV.style.display = 'flex';
  panelV.style.flexDirection = 'column';

  // Reset to messages tab
  switchTab('messages');
}

// -------------------------------------------------------------------------
// Tabs
// -------------------------------------------------------------------------
document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => switchTab(btn.dataset.tab));
});

function switchTab(name) {
  document.querySelectorAll('.tab-btn').forEach(b => b.classList.toggle('active', b.dataset.tab === name));
  document.querySelectorAll('.tab-pane').forEach(p => {
    const matches = p.id === `tab-${name}`;
    p.classList.toggle('active', matches);
  });

  if (name === 'messages')  loadMessages();
  if (name === 'knowledge') loadKnowledge();
  if (name === 'memories')  loadMemories();
  if (name === 'graph')     loadGraph();
}

// -------------------------------------------------------------------------
// Helpers
// -------------------------------------------------------------------------
function fmtDate(iso) {
  if (!iso) return '—';
  return new Date(iso).toLocaleString();
}

function escHtml(str) {
  return String(str ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;');
}

function badge(val) {
  return val
    ? '<span class="badge badge-on">Yes</span>'
    : '<span class="badge badge-off">No</span>';
}

// ---- Inline editable textarea ----
function makeEditableCell(currentText, onSave) {
  const wrapper = document.createElement('div');

  const span = document.createElement('span');
  span.className = 'cell-text';
  span.textContent = currentText;

  const textarea = document.createElement('textarea');
  textarea.className = 'cell-editor';
  textarea.style.display = 'none';
  textarea.value = currentText;

  const actions = document.createElement('div');
  actions.className = 'edit-actions';
  actions.style.display = 'none';

  const saveBtn = document.createElement('button');
  saveBtn.className = 'btn btn-primary btn-sm';
  saveBtn.textContent = 'Save';

  const cancelBtn = document.createElement('button');
  cancelBtn.className = 'btn btn-ghost btn-sm';
  cancelBtn.textContent = 'Cancel';

  actions.append(saveBtn, cancelBtn);
  wrapper.append(span, textarea, actions);

  function startEdit() {
    span.style.display = 'none';
    textarea.style.display = 'block';
    actions.style.display = 'flex';
    textarea.focus();
    textarea.selectionStart = textarea.value.length;
  }

  function cancelEdit() {
    textarea.value = span.textContent;
    textarea.style.display = 'none';
    actions.style.display = 'none';
    span.style.display = '';
  }

  span.addEventListener('click', startEdit);
  cancelBtn.addEventListener('click', cancelEdit);
  saveBtn.addEventListener('click', async () => {
    const newVal = textarea.value;
    if (newVal === span.textContent) { cancelEdit(); return; }
    saveBtn.disabled = true;
    saveBtn.textContent = 'Saving…';
    try {
      await onSave(newVal);
      span.textContent = newVal;
      showToast('Saved', 'ok');
    } catch (e) {
      showToast(`Error: ${e.message}`, 'error');
    } finally {
      saveBtn.disabled = false;
      saveBtn.textContent = 'Save';
      cancelEdit();
    }
  });

  return wrapper;
}

// ---- Delete button ----
function deleteBtn(label, onConfirm) {
  const btn = document.createElement('button');
  btn.className = 'btn btn-danger btn-sm';
  btn.textContent = 'Delete';
  btn.addEventListener('click', async () => {
    if (!confirm(`Delete this ${label}? This cannot be undone.`)) return;
    btn.disabled = true;
    try {
      await onConfirm();
      showToast('Deleted', 'ok');
    } catch (e) {
      showToast(`Error: ${e.message}`, 'error');
      btn.disabled = false;
    }
  });
  return btn;
}

// -------------------------------------------------------------------------
// Messages
// -------------------------------------------------------------------------
async function loadMessages() {
  const loading = document.getElementById('messages-loading');
  const table   = document.getElementById('messages-table');
  const tbody   = document.getElementById('messages-body');

  loading.style.display = 'block';
  table.style.display = 'none';
  tbody.innerHTML = '';

  let messages;
  try {
    messages = await get(`/users/${currentUser.id}/chats/${currentChat.id}/messages?limit=500`);
  } catch (e) {
    loading.textContent = `Error: ${e.message}`;
    return;
  }

  loading.style.display = 'none';
  if (!messages.length) {
    loading.textContent = 'No messages.';
    loading.style.display = 'block';
    return;
  }

  for (const msg of messages) {
    const tr = document.createElement('tr');

    const idx = msg.message_index;
    const uid = currentUser.id;
    const cid = currentChat.id;

    const editCell = makeEditableCell(msg.message, async (newText) => {
      await patch(`/users/${uid}/chats/${cid}/messages/${idx}`, { message: newText, created_date: msg.created_date });
    });

    const del_btn = deleteBtn('message', async () => {
      await del(`/users/${uid}/chats/${cid}/messages/${idx}`);
      tr.remove();
    });

    tr.innerHTML = `<td>${escHtml(idx)}</td><td>${escHtml(fmtDate(msg.created_date))}</td><td></td><td></td>`;
    tr.cells[2].appendChild(editCell);
    tr.cells[3].appendChild(del_btn);
    tbody.appendChild(tr);
  }

  table.style.display = 'table';
}

// -------------------------------------------------------------------------
// Knowledge
// -------------------------------------------------------------------------
async function loadKnowledge() {
  const loading = document.getElementById('knowledge-loading');
  const table   = document.getElementById('knowledge-table');
  const tbody   = document.getElementById('knowledge-body');

  loading.style.display = 'block';
  table.style.display = 'none';
  tbody.innerHTML = '';

  let entries;
  try {
    entries = await get(`/users/${currentUser.id}/chats/${currentChat.id}/knowledge?limit=500`);
  } catch (e) {
    loading.textContent = `Error: ${e.message}`;
    return;
  }

  loading.style.display = 'none';
  if (!entries.length) {
    loading.textContent = 'No knowledge entries.';
    loading.style.display = 'block';
    return;
  }

  for (const kn of entries) {
    const tr = document.createElement('tr');
    const uid = currentUser.id;
    const cid = currentChat.id;
    const kid = kn.id;

    const summaryCell = makeEditableCell(kn.summary, async (newText) => {
      await patch(`/users/${uid}/chats/${cid}/knowledge/${kid}`, { summary: newText });
    });

    const enabledCell = document.createElement('span');
    enabledCell.innerHTML = badge(kn.enabled);
    enabledCell.style.cursor = 'pointer';
    enabledCell.title = 'Click to toggle';
    let enabledState = kn.enabled;
    enabledCell.addEventListener('click', async () => {
      try {
        enabledState = !enabledState;
        await patch(`/users/${uid}/chats/${cid}/knowledge/${kid}`, { enabled: enabledState });
        enabledCell.innerHTML = badge(enabledState);
        showToast('Updated', 'ok');
      } catch (e) {
        enabledState = !enabledState;
        showToast(`Error: ${e.message}`, 'error');
      }
    });

    const frozenCell = document.createElement('span');
    frozenCell.innerHTML = badge(kn.frozen);
    frozenCell.style.cursor = 'pointer';
    frozenCell.title = 'Click to toggle';
    let frozenState = kn.frozen;
    frozenCell.addEventListener('click', async () => {
      try {
        frozenState = !frozenState;
        await patch(`/users/${uid}/chats/${cid}/knowledge/${kid}`, { frozen: frozenState });
        frozenCell.innerHTML = badge(frozenState);
        showToast('Updated', 'ok');
      } catch (e) {
        frozenState = !frozenState;
        showToast(`Error: ${e.message}`, 'error');
      }
    });

    const del_btn = deleteBtn('knowledge entry', async () => {
      await del(`/users/${uid}/chats/${cid}/knowledge/${kid}`);
      tr.remove();
    });

    tr.innerHTML = `
      <td>${escHtml(kn.entity)}</td>
      <td>${escHtml(kn.entity_label ?? '—')}</td>
      <td></td>
      <td></td>
      <td></td>
      <td>${escHtml(fmtDate(kn.updated_date))}</td>
      <td></td>
    `;
    tr.cells[2].appendChild(summaryCell);
    tr.cells[3].appendChild(enabledCell);
    tr.cells[4].appendChild(frozenCell);
    tr.cells[6].appendChild(del_btn);
    tbody.appendChild(tr);
  }

  table.style.display = 'table';
}

// -------------------------------------------------------------------------
// Memories
// -------------------------------------------------------------------------
async function loadMemories() {
  const loading = document.getElementById('memories-loading');
  const table   = document.getElementById('memories-table');
  const tbody   = document.getElementById('memories-body');

  loading.style.display = 'block';
  table.style.display = 'none';
  tbody.innerHTML = '';

  let memories;
  try {
    memories = await get(`/users/${currentUser.id}/chats/${currentChat.id}/memories?limit=500`);
  } catch (e) {
    loading.textContent = `Error: ${e.message}`;
    return;
  }

  loading.style.display = 'none';
  if (!memories.length) {
    loading.textContent = 'No segmented memories.';
    loading.style.display = 'block';
    return;
  }

  for (const mem of memories) {
    const tr = document.createElement('tr');
    const uid = currentUser.id;
    const cid = currentChat.id;
    const mid = mem.id;

    const summaryCell = makeEditableCell(mem.summary, async (newText) => {
      await patch(`/users/${uid}/chats/${cid}/memories/${mid}`, { summary: newText });
    });

    const del_btn = deleteBtn('memory', async () => {
      await del(`/users/${uid}/chats/${cid}/memories/${mid}`);
      tr.remove();
    });

    tr.innerHTML = `<td>${escHtml(mid)}</td><td>${escHtml(fmtDate(mem.created_date))}</td><td>${escHtml(mem.token_count)}</td><td></td><td></td>`;
    tr.cells[3].appendChild(summaryCell);
    tr.cells[4].appendChild(del_btn);
    tbody.appendChild(tr);
  }

  table.style.display = 'table';
}

// -------------------------------------------------------------------------
// Graph
// -------------------------------------------------------------------------
async function loadGraph() {
  const canvas = document.getElementById('graph-canvas');
  canvas.innerHTML = '<div style="padding:20px;color:var(--text-muted)">Loading graph…</div>';

  if (network) { network.destroy(); network = null; }

  let data;
  try {
    data = await get(`/users/${currentUser.id}/chats/${currentChat.id}/memory_graph`);
  } catch (e) {
    canvas.innerHTML = `<div style="padding:20px;color:var(--danger)">Error: ${e.message}</div>`;
    return;
  }

  if (!data.nodes || !data.nodes.length) {
    canvas.innerHTML = '<div style="padding:20px;color:var(--text-muted)">No graph data yet — knowledge must have summaries and shared memories before edges appear.</div>';
    return;
  }

  canvas.innerHTML = '';

  const nodes = new vis.DataSet(data.nodes.map(n => ({
    id: n.id,
    label: n.label,
    title: `Memories: ${(n.memory_list || []).join(', ') || 'none'}`,
  })));

  const edges = new vis.DataSet(data.edges.map((e, i) => ({
    id: i,
    from: e.source,
    to: e.target,
    title: `${e.weight} shared ${e.weight === 1 ? 'memory' : 'memories'}`,
    width: Math.min(1 + e.weight, 6),
    value: e.weight,
  })));

  const options = {
    nodes: {
      shape: 'dot',
      size: 14,
      color: { background: '#e94560', border: '#c2314a', highlight: { background: '#ff6b88', border: '#e94560' } },
      font: { color: '#eaeaea', size: 13, strokeWidth: 3, strokeColor: '#1a1a2e' },
    },
    edges: {
      color: { color: '#4a4a6a', highlight: '#9a9ab0', opacity: 0.8 },
      // labels are shown in hover tooltips only — inline text clutters the graph
      smooth: { type: 'continuous' },
    },
    physics: {
      stabilization: { iterations: 250, fit: true },
      barnesHut: {
        gravitationalConstant: -10000,
        springLength: 220,
        springConstant: 0.04,
        damping: 0.12,
        avoidOverlap: 0.5,
      },
    },
    interaction: { hover: true, tooltipDelay: 100, navigationButtons: true, keyboard: true },
    layout: { improvedLayout: true },
  };

  network = new vis.Network(canvas, { nodes, edges }, options);

  // After the physics engine settles, freeze the layout and zoom to fit
  network.on('stabilizationIterationsDone', () => {
    network.setOptions({ physics: { enabled: false } });
    network.fit({ animation: { duration: 600, easingFunction: 'easeInOutQuad' } });
  });
}

// -------------------------------------------------------------------------
// Lorebook Generator
// -------------------------------------------------------------------------
let _lbPollTimer   = null;   // setInterval handle
let _lbLorebook    = null;   // last completed Lorebook object
let _lbInFlight    = false;  // guard against overlapping runs

function showLorebook() {
  // Deselect any active chat
  document.querySelectorAll('.tree-chat').forEach(c => c.classList.remove('active'));
  currentUser = null;
  currentChat = null;

  const noMsg     = document.getElementById('no-chat-msg');
  const panelV    = document.getElementById('panel-view');
  const lbView    = document.getElementById('lorebook-view');
  const settV     = document.getElementById('settings-view');
  if (noMsg)   noMsg.style.display   = 'none';
  panelV.style.display = 'none';
  settV.style.display  = 'none';
  lbView.style.display = 'flex';
}

function _lbSetButtons(disabled) {
  document.getElementById('lb-gen-text').disabled = disabled;
  document.getElementById('lb-gen-file').disabled = disabled;
}

function _lbSetStatus(msg, type = '') {
  const el = document.getElementById('lb-status');
  el.textContent = msg;
  el.className   = type ? `lb-status-${type}` : '';
  el.style.display = msg ? 'block' : 'none';
}

function _lbRenderResults(lorebook) {
  _lbLorebook = lorebook;
  const entries = Object.values(lorebook.entries || {});

  document.getElementById('lb-entry-count').textContent =
    `${entries.length} entr${entries.length === 1 ? 'y' : 'ies'} generated`;

  const tbody = document.getElementById('lb-body');
  tbody.innerHTML = '';
  for (const entry of entries) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${escHtml(entry.comment)}</td>
      <td>${escHtml((entry.key || []).join(', '))}</td>
      <td>${escHtml(entry.content)}</td>
    `;
    tbody.appendChild(tr);
  }

  document.getElementById('lb-results').style.display = 'block';
}

function _lbStartPolling(requestId) {
  if (_lbPollTimer) clearInterval(_lbPollTimer);

  _lbPollTimer = setInterval(async () => {
    let resp;
    try {
      resp = await post('/autolorebook/status', { request_id: requestId });
    } catch (e) {
      clearInterval(_lbPollTimer);
      _lbPollTimer = null;
      _lbInFlight  = false;
      _lbSetStatus(`Error polling status: ${e.message}`, 'error');
      _lbSetButtons(false);
      return;
    }

    if (resp.status === 'processing') {
      const done = Object.values(resp.lorebook?.entries || {}).length;
      _lbSetStatus(`Processing… ${done} entr${done === 1 ? 'y' : 'ies'} ready so far`);
      return;
    }

    // Terminal states
    clearInterval(_lbPollTimer);
    _lbPollTimer = null;
    _lbInFlight  = false;
    _lbSetButtons(false);

    if (resp.status === 'done') {
      _lbSetStatus('');
      _lbRenderResults(resp.lorebook);
      showToast('Lorebook generated!', 'ok');
    } else {
      _lbSetStatus('Generation failed — check that the Celery worker is running.', 'error');
      showToast('Lorebook generation failed', 'error');
    }
  }, 2000);
}

async function generateLorebook(requestPromise) {
  if (_lbInFlight) return;
  _lbInFlight = true;
  _lbSetButtons(true);
  document.getElementById('lb-results').style.display = 'none';
  _lbSetStatus('Submitting…');

  let resp;
  try {
    resp = await requestPromise;
  } catch (e) {
    _lbInFlight = false;
    _lbSetButtons(false);
    _lbSetStatus(`Error: ${e.message}`, 'error');
    showToast(`Error: ${e.message}`, 'error');
    return;
  }

  _lbSetStatus('Processing… waiting for Celery workers');
  _lbStartPolling(resp.request_id);
}

document.getElementById('lorebook-btn').addEventListener('click', showLorebook);

document.getElementById('lb-gen-text').addEventListener('click', () => {
  const text = document.getElementById('lb-text').value.trim();
  if (!text) { showToast('Please paste some text first.', 'error'); return; }
  generateLorebook(post('/autolorebook/create', { text }));
});

document.getElementById('lb-gen-file').addEventListener('click', () => {
  const fileInput = document.getElementById('lb-file');
  if (!fileInput.files.length) { showToast('Please select a file first.', 'error'); return; }
  generateLorebook(uploadFile('/autolorebook/upload_file', fileInput.files[0]));
});

document.getElementById('lb-download').addEventListener('click', () => {
  if (!_lbLorebook) return;
  const blob = new Blob([JSON.stringify(_lbLorebook, null, 2)], { type: 'application/json' });
  const url  = URL.createObjectURL(blob);
  const a    = document.createElement('a');
  a.href     = url;
  a.download = `${_lbLorebook.name || 'lorebook'}.json`;
  a.click();
  URL.revokeObjectURL(url);
});

// -------------------------------------------------------------------------
// Settings
// -------------------------------------------------------------------------
const SETTINGS_SECTIONS = ['summarization_api', 'summarization', 'tokenization'];

function showSettings() {
  document.querySelectorAll('.tree-chat').forEach(c => c.classList.remove('active'));
  currentUser = null;
  currentChat = null;

  const noMsg  = document.getElementById('no-chat-msg');
  const panelV = document.getElementById('panel-view');
  const lbView = document.getElementById('lorebook-view');
  const settV  = document.getElementById('settings-view');
  if (noMsg) noMsg.style.display = 'none';
  panelV.style.display = 'none';
  lbView.style.display = 'none';
  settV.style.display  = 'block';

  loadSettings();
}

function _settingsField(section, name) {
  return document.querySelector(`fieldset[data-section="${section}"] [data-field="${name}"]`);
}

function _fillSettingsForm(data) {
  for (const section of SETTINGS_SECTIONS) {
    for (const [name, value] of Object.entries(data[section])) {
      if (name === 'auth_key_set') continue;
      const el = _settingsField(section, name);
      if (!el) continue;
      if (el.type === 'checkbox') el.checked = value;
      else if ('json' in el.dataset) el.value = JSON.stringify(value, null, 2);
      else el.value = value;
    }
  }

  _settingsField('summarization_api', 'auth_key').placeholder =
    data.summarization_api.auth_key_set ? 'unchanged (key is set)' : 'unchanged (no key set)';

  for (const section of SETTINGS_SECTIONS) {
    const badge = document.querySelector(`[data-badge="${section}"]`);
    const isOverridden = data.overridden_sections.includes(section);
    badge.textContent = isOverridden ? 'overridden' : 'default';
    badge.className = `section-badge ${isOverridden ? 'section-badge-on' : ''}`;
  }
}

async function loadSettings() {
  const loading = document.getElementById('settings-loading');
  const form    = document.getElementById('settings-form');
  loading.style.display = 'block';
  form.style.display = 'none';

  let data;
  try {
    data = await get('/settings');
  } catch (e) {
    loading.textContent = `Error: ${e.message}`;
    return;
  }

  _fillSettingsForm(data);
  loading.style.display = 'none';
  form.style.display = 'flex';
}

function _readSection(section) {
  const fieldset = document.querySelector(`fieldset[data-section="${section}"]`);
  const out = {};
  fieldset.querySelectorAll('[data-field]').forEach(el => {
    const name = el.dataset.field;
    if (el.type === 'checkbox') {
      out[name] = el.checked;
    } else if (el.type === 'number') {
      if (el.value !== '') out[name] = Number(el.value);
    } else if (name === 'auth_key') {
      if (el.value) out[name] = el.value;  // blank = leave unchanged
    } else if ('json' in el.dataset) {
      out[name] = JSON.parse(el.value);    // let JSON errors surface to the caller
    } else {
      out[name] = el.value;
    }
  });
  return out;
}

function _readSectionOrThrow(section, label) {
  try {
    return _readSection(section);
  } catch (parseErr) {
    throw new Error(`${label} must be valid JSON: ${parseErr.message}`);
  }
}

async function saveSettings(e) {
  e.preventDefault();
  const saveBtn = document.getElementById('settings-save');
  saveBtn.disabled = true;
  saveBtn.textContent = 'Saving…';

  try {
    const body = {
      summarization_api: _readSectionOrThrow('summarization_api', 'Chat template kwargs'),
      summarization: _readSectionOrThrow('summarization', 'Sampler params'),
      tokenization: _readSection('tokenization'),
    };
    const data = await put('/settings', body);
    _fillSettingsForm(data);
    showToast('Settings saved', 'ok');
  } catch (e2) {
    showToast(`Error: ${e2.message}`, 'error');
  } finally {
    saveBtn.disabled = false;
    saveBtn.textContent = 'Save changes';
  }
}

async function resetSection(section) {
  if (!confirm(`Reset "${section}" settings to the file default?`)) return;
  try {
    const data = await post('/settings/reset', { section });
    _fillSettingsForm(data);
    showToast('Reset to default', 'ok');
  } catch (e) {
    showToast(`Error: ${e.message}`, 'error');
  }
}

document.getElementById('settings-btn').addEventListener('click', showSettings);
document.getElementById('settings-form').addEventListener('submit', saveSettings);
document.querySelectorAll('.settings-reset').forEach(btn => {
  btn.addEventListener('click', () => resetSection(btn.dataset.reset));
});

// -------------------------------------------------------------------------
// Boot
// -------------------------------------------------------------------------
loadSidebar();
