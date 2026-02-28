import { useEffect, useMemo, useState } from 'react'
import { api } from './api'
import MetricCard from './components/MetricCard'

function fmtUsd(v) {
  return `$${Number(v || 0).toLocaleString(undefined, { maximumFractionDigits: 2 })}`
}

function fmtPct(v) {
  return `${(Number(v || 0) * 100).toFixed(2)}%`
}

export default function App() {
  const [portfolio, setPortfolio] = useState([])
  const [positions, setPositions] = useState([])
  const [signals, setSignals] = useState([])
  const [transactions, setTransactions] = useState([])
  const [goal, setGoal] = useState(null)
  const [manualChecks, setManualChecks] = useState([])
  const [tickerInput, setTickerInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const refresh = async () => {
    setLoading(true)
    setError('')
    try {
      const [p, pos, s, t, g, m] = await Promise.all([
        api.getPortfolio(),
        api.getPositions(),
        api.getSignals(),
        api.getTransactions(),
        api.getGoal(),
        api.getManualChecks()
      ])
      setPortfolio(p.rows || [])
      setPositions(pos.rows || [])
      setSignals(s.rows || [])
      setTransactions(t.rows || [])
      setGoal(g.row || null)
      setManualChecks(m.rows || [])
    } catch (e) {
      setError(e.message)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, 60000)
    return () => clearInterval(id)
  }, [])

  const latestPortfolioValue = useMemo(() => {
    if (!portfolio.length) return 0
    return Number(portfolio[0].value || 0)
  }, [portfolio])

  const goalTarget = Number(goal?.target_capital || 2100)
  const goalRemaining = Math.max(Number(goal?.remaining_capital || goalTarget - latestPortfolioValue), 0)
  const daysRemaining = Number(goal?.days_remaining || 0)
  const requiredDaily = Number(goal?.required_daily_return || 0)

  const runCycle = async () => {
    setLoading(true)
    setError('')
    try {
      await api.runCycle()
      await refresh()
    } catch (e) {
      setError(e.message)
      setLoading(false)
    }
  }

  const addCheck = async (e) => {
    e.preventDefault()
    const ticker = tickerInput.trim().toUpperCase()
    if (!ticker) return
    setLoading(true)
    try {
      await api.addManualCheck(ticker)
      setTickerInput('')
      await refresh()
    } catch (e2) {
      setError(e2.message)
      setLoading(false)
    }
  }

  const removeCheck = async (ticker) => {
    setLoading(true)
    try {
      await api.deleteManualCheck(ticker)
      await refresh()
    } catch (e) {
      setError(e.message)
      setLoading(false)
    }
  }

  return (
    <div className="page">
      <header className="topbar">
        <h1>Kaibot Control</h1>
        <div className="actions">
          <button onClick={refresh} disabled={loading}>Refresh</button>
          <button onClick={runCycle} disabled={loading}>Run Cycle</button>
        </div>
      </header>

      {error ? <div className="error">{error}</div> : null}

      <section className="metrics-grid">
        <MetricCard label="Portfolio Value" value={fmtUsd(latestPortfolioValue)} />
        <MetricCard label="Goal Target" value={fmtUsd(goalTarget)} />
        <MetricCard label="Remaining" value={fmtUsd(goalRemaining)} />
        <MetricCard label="Required Daily Return" value={fmtPct(requiredDaily)} sub={`${daysRemaining.toFixed(2)} days left`} />
      </section>

      <section className="panel">
        <h2>Manual Ticker Checks</h2>
        <form onSubmit={addCheck} className="inline-form">
          <input value={tickerInput} onChange={(e) => setTickerInput(e.target.value)} placeholder="AAPL" />
          <button type="submit" disabled={loading}>Add Check</button>
        </form>
        <table>
          <thead><tr><th>Ticker</th><th>Decision</th><th>Score</th><th>Price</th><th></th></tr></thead>
          <tbody>
            {manualChecks.map((row) => (
              <tr key={row.id || row.ticker}>
                <td>{row.ticker}</td>
                <td>{row.decision}</td>
                <td>{Number(row.score || 0).toFixed(3)}</td>
                <td>{Number(row.price || 0).toFixed(2)}</td>
                <td><button onClick={() => removeCheck(row.ticker)} disabled={loading}>Remove</button></td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>

      <section className="split">
        <div className="panel">
          <h2>Open Positions</h2>
          <table>
            <thead><tr><th>Ticker</th><th>Shares</th><th>Value</th><th>P/L %</th></tr></thead>
            <tbody>
              {positions.map((row, idx) => (
                <tr key={`${row.ticker}-${idx}`}>
                  <td>{row.ticker}</td>
                  <td>{Number(row.shares || 0).toFixed(4)}</td>
                  <td>{fmtUsd(row.market_value)}</td>
                  <td>{(Number(row.pnl_pct || 0) * 100).toFixed(2)}%</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        <div className="panel">
          <h2>Top Signals</h2>
          <table>
            <thead><tr><th>Ticker</th><th>Decision</th><th>Score</th><th>Price</th></tr></thead>
            <tbody>
              {signals.map((row, idx) => (
                <tr key={`${row.ticker}-${idx}`}>
                  <td>{row.ticker}</td>
                  <td>{row.decision}</td>
                  <td>{Number(row.score || 0).toFixed(3)}</td>
                  <td>{Number(row.price || 0).toFixed(2)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </section>

      <section className="panel">
        <h2>Recent Transactions</h2>
        <table>
          <thead><tr><th>Time</th><th>Ticker</th><th>Action</th><th>Shares</th><th>Price</th></tr></thead>
          <tbody>
            {transactions.map((row, idx) => (
              <tr key={`${row.time}-${row.ticker}-${idx}`}>
                <td>{row.time}</td>
                <td>{row.ticker}</td>
                <td>{row.action}</td>
                <td>{Number(row.shares || 0).toFixed(4)}</td>
                <td>{Number(row.price || 0).toFixed(2)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  )
}
