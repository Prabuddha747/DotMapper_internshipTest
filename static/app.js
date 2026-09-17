// Support Intelligence UI: fetches /api/* and renders results.
// Plain DOM APIs + CSS conic-gradient/flexbox for charts — no framework,
// no charting library, no build step needed at this scale.

const STATUS_COLORS = { Resolved: 'var(--status-resolved)', Open: 'var(--status-open)', Escalated: 'var(--status-escalated)' };
const STATUS_ORDER = ['Resolved', 'Open', 'Escalated'];
const PRIORITY_COLORS = { Critical: 'var(--priority-critical)', High: 'var(--priority-high)', Medium: 'var(--priority-medium)', Low: 'var(--priority-low)' };
const PRIORITY_ORDER = ['Critical', 'High', 'Medium', 'Low'];

async function fetchJSON(url, options) {
  const response = await fetch(url, options);
  return response.json();
}

// Health badge in the header, from GET /health.
async function loadHealth() {
  const badge = document.getElementById('health-badge');
  try {
    const health = await fetchJSON('/health');
    if (health.status !== 'ok') {
      badge.textContent = 'Data not loaded';
      badge.className = 'health-badge down';
    } else if (!health.groq_configured) {
      badge.textContent = 'Ready (no Groq key — Q&A disabled)';
      badge.className = 'health-badge degraded';
    } else {
      badge.textContent = `System ready · ${health.tickets_loaded} tickets`;
      badge.className = 'health-badge ok';
    }
  } catch {
    badge.textContent = 'Server unreachable';
    badge.className = 'health-badge down';
  }
}

// Four header stat cards + the two charts, all from GET /api/stats.
async function loadStats() {
  const stats = await fetchJSON('/api/stats');
  document.getElementById('stat-total').textContent = stats.total_tickets ?? '—';
  document.getElementById('stat-open').textContent = stats.by_status?.Open ?? 0;
  document.getElementById('stat-escalated').textContent = stats.by_status?.Escalated ?? 0;
  document.getElementById('stat-anomalies').textContent = stats.anomaly_count ?? 0;
  renderStatusDonut(stats.by_status ?? {}, stats.total_tickets ?? 0);
  renderPriorityBars(stats.by_priority ?? {});
}

// CSS conic-gradient donut (no SVG/library needed) + a text legend.
function renderStatusDonut(byStatus, total) {
  const donut = document.getElementById('status-donut');
  const legend = document.getElementById('status-legend');
  legend.innerHTML = '';

  let cursor = 0;
  const stops = STATUS_ORDER.filter((s) => byStatus[s]).map((status) => {
    const pct = total ? (byStatus[status] / total) * 100 : 0;
    const stop = `${STATUS_COLORS[status]} ${cursor}% ${cursor + pct}%`;
    cursor += pct;
    return stop;
  });
  donut.style.background = `conic-gradient(${stops.join(', ')})`;
  donut.setAttribute('data-total', total);

  STATUS_ORDER.filter((s) => byStatus[s]).forEach((status) => {
    const li = document.createElement('li');
    const pct = total ? Math.round((byStatus[status] / total) * 100) : 0;
    li.innerHTML = `<span class="dot" style="background:${STATUS_COLORS[status]}"></span>${status}`;
    const count = document.createElement('span');
    count.className = 'count';
    count.textContent = `${byStatus[status]} (${pct}%)`;
    li.appendChild(count);
    legend.appendChild(li);
  });
}

// Simple flexbox bar chart, bar height proportional to the largest value.
function renderPriorityBars(byPriority) {
  const container = document.getElementById('priority-bars');
  container.innerHTML = '';
  const max = Math.max(...Object.values(byPriority), 1);

  PRIORITY_ORDER.filter((p) => byPriority[p] !== undefined).forEach((priority) => {
    const value = byPriority[priority];
    const col = document.createElement('div');
    col.className = 'bar-col';

    const valueLabel = document.createElement('span');
    valueLabel.className = 'bar-value';
    valueLabel.textContent = value;

    const fill = document.createElement('div');
    fill.className = 'bar-fill';
    fill.style.height = `${Math.max((value / max) * 100, 4)}%`;
    fill.style.background = PRIORITY_COLORS[priority];

    const label = document.createElement('span');
    label.className = 'bar-label';
    label.textContent = priority;

    col.append(valueLabel, fill, label);
    container.appendChild(col);
  });
}

// Human-readable one-liner per anomaly rule shape (see app/anomaly.py).
function describeAnomaly(item) {
  if (item.rule === 'overdue_unresolved') {
    return `${item.priority}/${item.status}, ${item.age_hours}h old (${item.over_by_hours}h over the ${item.threshold_hours}h threshold)`;
  }
  if (item.rule === 'resolution_outlier') {
    return `resolved in ${item.resolution_time_hrs}h, ${item.deviation_hrs}h over the ${item.threshold_hrs}h threshold`;
  }
  return item.issue;
}

// Build one anomaly row via DOM nodes (never innerHTML) so free-text ticket
// data from the CSV can never be interpreted as markup.
function anomalyRowEl(item) {
  const row = document.createElement('div');
  row.className = 'anomaly-row';

  const top = document.createElement('div');
  const badge = document.createElement('span');
  badge.className = `rule-badge rule-${item.rule}`;
  badge.textContent = item.rule.replace('_', ' ');
  top.appendChild(badge);

  const detail = document.createElement('span');
  detail.textContent = `${item.ticket_id}: ${describeAnomaly(item)}`;

  row.append(top, detail);
  return row;
}

// Populate the anomalies panel from GET /api/anomalies.
async function loadAnomalies() {
  const { anomalies } = await fetchJSON('/api/anomalies');
  const list = document.getElementById('anomalies-list');
  list.innerHTML = '';
  if (anomalies.length === 0) {
    list.textContent = 'No anomalies found.';
    return;
  }
  anomalies.forEach((item) => list.appendChild(anomalyRowEl(item)));
}

// Ticket-rows table, appended under an AI message (filter_rows data.tickets).
function ticketTableEl(tickets) {
  if (!tickets || tickets.length === 0) return null;
  const columns = Object.keys(tickets[0]);
  const table = document.createElement('table');

  const headerRow = document.createElement('tr');
  columns.forEach((col) => {
    const th = document.createElement('th');
    th.textContent = col;
    headerRow.appendChild(th);
  });
  table.appendChild(headerRow);

  tickets.forEach((ticket) => {
    const row = document.createElement('tr');
    columns.forEach((col) => {
      const td = document.createElement('td');
      td.textContent = ticket[col] ?? '';
      row.appendChild(td);
    });
    table.appendChild(row);
  });
  return table;
}

// Append a chat bubble (user question or AI answer) and scroll it into view.
function appendChatMessage(role, text, table) {
  const history = document.getElementById('chat-history');
  const msg = document.createElement('div');
  msg.className = `chat-msg ${role}`;

  const p = document.createElement('p');
  p.textContent = text;
  msg.appendChild(p);
  if (table) msg.appendChild(table);

  history.appendChild(msg);
  history.scrollTop = history.scrollHeight;
  return msg;
}

let questionInFlight = false; // one request at a time — rapid re-clicks burn the free-tier Groq quota fast

// One id per browser tab session, so a follow-up ("...and by agent?") only
// ever inherits context from this tab's own previous question, never
// another visitor's. Persisted so a page refresh keeps the same session.
function getSessionId() {
  try {
    let id = localStorage.getItem('sessionId');
    if (!id) {
      id = crypto.randomUUID();
      localStorage.setItem('sessionId', id);
    }
    return id;
  } catch {
    return crypto.randomUUID(); // storage blocked (private mode etc.) — fresh id per page load
  }
}
const sessionId = getSessionId();

// Ask a question: POST /api/query, render the answer as a new AI chat bubble.
async function askQuestion(question) {
  if (questionInFlight) return;
  questionInFlight = true;
  setAskControlsEnabled(false);

  appendChatMessage('user', question);
  const placeholder = appendChatMessage('ai', 'Thinking…');

  try {
    const result = await fetchJSON('/api/query', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ question, session_id: sessionId }),
    });
    placeholder.querySelector('p').textContent = result.answer ?? 'No answer returned.';
    if (result.operation === 'error') placeholder.classList.add('error');
    const table = ticketTableEl(result.data?.tickets);
    if (table) placeholder.appendChild(table);
  } catch {
    placeholder.querySelector('p').textContent = 'Request failed — is the server running?';
    placeholder.classList.add('error');
  } finally {
    questionInFlight = false;
    setAskControlsEnabled(true);
  }
}

// Disable Ask + example buttons while a question is in flight.
function setAskControlsEnabled(enabled) {
  document.querySelectorAll('#ask-form button, .example-btn').forEach((btn) => { btn.disabled = !enabled; });
}

document.addEventListener('DOMContentLoaded', () => {
  loadHealth();
  loadStats();
  loadAnomalies();

  document.getElementById('ask-form').addEventListener('submit', (event) => {
    event.preventDefault();
    const input = document.getElementById('question-input');
    if (!input.value.trim()) return;
    askQuestion(input.value);
    input.value = '';
  });

  document.querySelectorAll('.example-btn').forEach((button) => {
    button.addEventListener('click', () => askQuestion(button.textContent));
  });
});
