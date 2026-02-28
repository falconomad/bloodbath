const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000'

async function request(path, options = {}) {
  const res = await fetch(`${API_BASE}${path}`, {
    headers: { 'Content-Type': 'application/json' },
    ...options
  })
  if (!res.ok) {
    const text = await res.text()
    throw new Error(`${res.status} ${res.statusText}: ${text}`)
  }
  return res.json()
}

export const api = {
  getPortfolio: (limit = 500) => request(`/api/portfolio?limit=${limit}`),
  getPositions: () => request('/api/positions'),
  getTransactions: (limit = 100) => request(`/api/transactions?limit=${limit}`),
  getSignals: (limit = 50) => request(`/api/signals?limit=${limit}`),
  getGoal: () => request('/api/goal/latest'),
  getManualChecks: (limit = 20) => request(`/api/manual-checks?limit=${limit}`),
  addManualCheck: (ticker) => request('/api/manual-checks', { method: 'POST', body: JSON.stringify({ ticker }) }),
  deleteManualCheck: (ticker) => request(`/api/manual-checks/${ticker}`, { method: 'DELETE' }),
  runCycle: () => request('/api/cycle/run', { method: 'POST' })
}
