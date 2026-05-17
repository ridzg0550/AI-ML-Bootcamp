/* ═══════════════════════════════════════════════════════════════
   DungeonBrain++ — Frontend Application
   ═══════════════════════════════════════════════════════════════ */

const API = '';  // Same-origin, no prefix needed

// ── State ──────────────────────────────────────────────────────
let sessionId = null;
let turnCount = 0;
let isWaiting = false;

// ── DOM References ─────────────────────────────────────────────
const $ = (sel) => document.querySelector(sel);
const welcomeScreen   = $('#welcome-screen');
const messagesContainer = $('#messages-container');
const typingIndicator = $('#typing-indicator');
const playerInput     = $('#player-input');
const btnSend         = $('#btn-send');
const btnNew          = $('#btn-new');
const btnSave         = $('#btn-save');
const btnToggleStats  = $('#btn-toggle-stats');
const btnStartWelcome = $('#btn-start-welcome');
const sessionLabel    = $('#session-label');
const statsPanel      = $('#stats-panel');
const storyPanel      = $('#story-panel');

// ── Event Listeners ────────────────────────────────────────────
btnNew.addEventListener('click', startNewSession);
btnStartWelcome.addEventListener('click', startNewSession);
btnSave.addEventListener('click', saveSession);
btnToggleStats.addEventListener('click', toggleStats);
btnSend.addEventListener('click', sendMessage);

playerInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !isWaiting) {
        e.preventDefault();
        sendMessage();
    }
});

document.addEventListener('keydown', (e) => {
    if (e.ctrlKey && e.key === 's') {
        e.preventDefault();
        saveSession();
    }
});

// ── Session Management ─────────────────────────────────────────
async function startNewSession() {
    if (isWaiting) return;
    setLoading(true);

    try {
        const res = await fetch(`${API}/api/session/new`, { method: 'POST' });
        if (!res.ok) throw new Error(`HTTP ${res.status}`);
        const data = await res.json();

        sessionId = data.session_id;
        turnCount = data.turn_count;

        // Update UI state
        sessionLabel.textContent = `Session: ${sessionId}`;
        btnSave.disabled = false;
        playerInput.disabled = false;
        btnSend.disabled = false;
        welcomeScreen.style.display = 'none';
        messagesContainer.style.display = 'flex';
        messagesContainer.innerHTML = '';

        // Add opening message
        addMessage('dm', data.opening_message, 0);
        updateTurnBadge(0);

        // Fetch initial stats
        await refreshStats();

        playerInput.focus();
    } catch (err) {
        showError('Failed to start session: ' + err.message);
    } finally {
        setLoading(false);
    }
}

async function saveSession() {
    if (!sessionId) return;
    try {
        const res = await fetch(`${API}/api/session/${sessionId}/save`, { method: 'POST' });
        if (!res.ok) throw new Error('Save failed');
        showToast('💾 Campaign saved!');
    } catch (err) {
        showError('Save failed: ' + err.message);
    }
}

// ── Messaging ──────────────────────────────────────────────────
async function sendMessage() {
    const text = playerInput.value.trim();
    if (!text || !sessionId || isWaiting) return;

    playerInput.value = '';
    addMessage('player', text, turnCount + 1);
    setLoading(true);
    showTyping(true);

    try {
        const res = await fetch(`${API}/api/session/${sessionId}/message`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: text }),
        });

        if (!res.ok) {
            const errData = await res.json().catch(() => ({}));
            throw new Error(errData.detail || `HTTP ${res.status}`);
        }

        const data = await res.json();
        turnCount = data.turn_count;

        showTyping(false);
        addMessage('dm', data.response, turnCount);

        // Update stats from response
        if (data.stats) {
            updateDashboard(data.stats);
        }
    } catch (err) {
        showTyping(false);
        addMessage('dm', '⚠️ ' + err.message, turnCount);
    } finally {
        setLoading(false);
        playerInput.focus();
    }
}

// ── Message Rendering ──────────────────────────────────────────
function addMessage(role, text, turn) {
    const msg = document.createElement('div');
    msg.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = role === 'dm' ? '🐉' : '⚔️';

    const content = document.createElement('div');
    content.className = 'message-content';

    const body = document.createElement('div');
    body.className = 'message-body';

    // Typewriter effect for DM messages
    if (role === 'dm') {
        typewriterEffect(body, text);
    } else {
        body.textContent = text;
    }

    const meta = document.createElement('div');
    meta.className = 'message-meta';
    meta.textContent = `Turn ${turn}`;

    content.appendChild(body);
    content.appendChild(meta);
    msg.appendChild(avatar);
    msg.appendChild(content);
    messagesContainer.appendChild(msg);

    // Scroll to bottom
    requestAnimationFrame(() => {
        storyPanel.scrollTop = storyPanel.scrollHeight;
    });
}

function typewriterEffect(element, text, speed = 12) {
    let i = 0;
    element.textContent = '';
    const timer = setInterval(() => {
        if (i < text.length) {
            element.textContent += text[i];
            i++;
            // Keep scrolled during typing
            if (i % 10 === 0) storyPanel.scrollTop = storyPanel.scrollHeight;
        } else {
            clearInterval(timer);
            storyPanel.scrollTop = storyPanel.scrollHeight;
        }
    }, speed);
}

// ── Dashboard Updates ──────────────────────────────────────────
async function refreshStats() {
    if (!sessionId) return;
    try {
        const res = await fetch(`${API}/api/session/${sessionId}/stats`);
        if (res.ok) {
            const stats = await res.json();
            updateDashboard(stats);
        }
    } catch (e) { /* silent */ }
}

function updateDashboard(s) {
    // Turn badge
    updateTurnBadge(s.turn_count);

    // Context amplification
    animateNumber('amp-value', s.context_amplification, 'x', 2);
    $('#native-tokens').textContent = s.native_context_tokens.toLocaleString();
    $('#effective-tokens').textContent = s.effective_context_tokens.toLocaleString();
    const ampPercent = Math.min(100, (s.effective_context_tokens / Math.max(1, s.native_context_tokens)) * 100);
    $('#effective-bar').style.width = ampPercent + '%';

    // Memory bank
    animateCounter('stat-total-memories', s.total_memories);
    animateCounter('stat-permanent', s.permanent_memories);
    animateCounter('stat-transient', s.transient_memories);
    $('#stat-salience').textContent = s.avg_salience.toFixed(2);

    // Era bar
    const totalMem = s.total_memories || 1;
    const era = s.memory_by_era || { early: 0, mid: 0, late: 0 };
    $('#era-early').style.width = ((era.early / totalMem) * 100) + '%';
    $('#era-mid').style.width = ((era.mid / totalMem) * 100) + '%';
    $('#era-late').style.width = ((era.late / totalMem) * 100) + '%';

    // Neural links
    animateCounter('stat-links', s.total_links);
    $('#stat-link-strength').textContent = s.avg_link_strength.toFixed(3);

    // Retrieval stats
    const rs = s.retrieval_stats || {};
    animateCounter('stat-retrievals', rs.total_retrievals || 0);
    $('#stat-early-recall').textContent = ((rs.early_event_recall_rate || 0) * 100).toFixed(0) + '%';
    $('#stat-avg-retrieved').textContent = (rs.avg_retrieved_per_turn || 0).toFixed(1);

    // NPCs
    updateNPCList(s.npc_list || []);
    $('#npc-count').textContent = s.npc_count || 0;

    // Quests
    updateQuestList(s.quest_list || []);
    $('#quest-count').textContent = s.total_quests || 0;

    // World state
    const slots = s.slot_memory || {};
    $('#slot-location').textContent = slots.location || 'Unknown';
    $('#slot-time').textContent = slots.time_of_day || '—';
    $('#slot-goal').textContent = slots.current_goal || '—';

    // Engine
    $('#stat-emb-dim').textContent = s.embedding_dimension || 384;
    $('#stat-faiss').textContent = s.faiss_index_size || 0;
}

function updateNPCList(npcs) {
    const container = $('#npc-list');
    if (!npcs || npcs.length === 0) {
        container.innerHTML = '<div class="empty-state">No characters encountered yet</div>';
        return;
    }
    container.innerHTML = npcs.map(npc => `
        <div class="npc-item">
            <span class="npc-rel-dot ${npc.relationship || 'neutral'}"></span>
            <span class="npc-name">${escapeHtml(npc.name)}</span>
            <span class="npc-info">${npc.interactions || 0}× · T${npc.first_met || '?'}</span>
        </div>
    `).join('');
}

function updateQuestList(quests) {
    const container = $('#quest-list');
    if (!quests || quests.length === 0) {
        container.innerHTML = '<div class="empty-state">No quests discovered yet</div>';
        return;
    }
    container.innerHTML = quests.map(q => `
        <div class="quest-item ${q.status === 'completed' ? 'completed' : ''}">
            <div class="quest-name">${q.status === 'completed' ? '✓' : '→'} ${escapeHtml(q.name)}</div>
            <div class="quest-status">Started Turn ${q.started_turn} · ${q.updates || 0} updates</div>
        </div>
    `).join('');
}

function updateTurnBadge(turn) {
    $('#turn-badge').textContent = `Turn ${turn}`;
}

// ── Animation Helpers ──────────────────────────────────────────
function animateCounter(elId, target) {
    const el = document.getElementById(elId);
    if (!el) return;
    const current = parseInt(el.textContent) || 0;
    if (current === target) return;

    const duration = 400;
    const start = performance.now();

    function step(now) {
        const progress = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // ease-out cubic
        el.textContent = Math.round(current + (target - current) * eased);
        if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
}

function animateNumber(elId, target, suffix = '', decimals = 0) {
    const el = document.getElementById(elId);
    if (!el) return;
    const currentText = el.textContent.replace(/[^0-9.]/g, '');
    const current = parseFloat(currentText) || 0;

    const duration = 500;
    const start = performance.now();

    function step(now) {
        const progress = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const value = current + (target - current) * eased;
        el.textContent = value.toFixed(decimals) + suffix;
        if (progress < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
}

// ── UI Helpers ─────────────────────────────────────────────────
function setLoading(loading) {
    isWaiting = loading;
    playerInput.disabled = loading || !sessionId;
    btnSend.disabled = loading || !sessionId;
}

function showTyping(show) {
    typingIndicator.style.display = show ? 'flex' : 'none';
    if (show) storyPanel.scrollTop = storyPanel.scrollHeight;
}

function toggleStats() {
    statsPanel.classList.toggle('collapsed');
}

function showToast(msg) {
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed; bottom: 100px; left: 50%; transform: translateX(-50%);
        padding: 10px 24px; border-radius: 8px; z-index: 999;
        background: rgba(74, 222, 128, 0.15); border: 1px solid rgba(74, 222, 128, 0.3);
        color: #4ade80; font-size: 13px; font-weight: 500;
        animation: fadeIn 0.3s ease;
    `;
    toast.textContent = msg;
    document.body.appendChild(toast);
    setTimeout(() => { toast.style.opacity = '0'; toast.style.transition = 'opacity 0.4s'; }, 2000);
    setTimeout(() => toast.remove(), 2500);
}

function showError(msg) {
    const toast = document.createElement('div');
    toast.style.cssText = `
        position: fixed; bottom: 100px; left: 50%; transform: translateX(-50%);
        padding: 10px 24px; border-radius: 8px; z-index: 999;
        background: rgba(251, 113, 133, 0.15); border: 1px solid rgba(251, 113, 133, 0.3);
        color: #fb7185; font-size: 13px; font-weight: 500;
        animation: fadeIn 0.3s ease;
    `;
    toast.textContent = msg;
    document.body.appendChild(toast);
    setTimeout(() => { toast.style.opacity = '0'; toast.style.transition = 'opacity 0.4s'; }, 3000);
    setTimeout(() => toast.remove(), 3500);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}
