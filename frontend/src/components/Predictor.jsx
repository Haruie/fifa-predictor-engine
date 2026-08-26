import { useEffect, useMemo, useState } from 'react'
import { getTeams, predict } from '../api'
import { MODEL_LABEL } from '../constants'

const DEFAULT_MATCHUP = ['Brazil', 'Germany']

export default function Predictor() {
  const [teams, setTeams] = useState([])
  const [teamA, setTeamA] = useState('')
  const [teamB, setTeamB] = useState('')
  const [year, setYear] = useState('')
  const [result, setResult] = useState(null)
  const [error, setError] = useState('')
  const [loading, setLoading] = useState(false)

  useEffect(() => {
    getTeams().then((data) => {
      setTeams(data)
      const names = new Set(data.map((t) => t.team))
      const [a, b] = DEFAULT_MATCHUP.every((n) => names.has(n))
        ? DEFAULT_MATCHUP
        : data.slice(0, 2).map((t) => t.team)
      if (a && b) {
        setTeamA(a)
        setTeamB(b)
      }
    }).catch((err) => setError(err.message))
  }, [])

  const years = useMemo(() => {
    const a = teams.find((t) => t.team === teamA)
    const b = teams.find((t) => t.team === teamB)
    if (!a || !b) return []
    return a.years.filter((y) => b.years.includes(y))
  }, [teams, teamA, teamB])

  useEffect(() => {
    if (years.length && !years.includes(Number(year))) setYear(years[years.length - 1])
  }, [years]) // eslint-disable-line react-hooks/exhaustive-deps

  async function handlePredict(e) {
    e.preventDefault()
    setError('')
    setResult(null)
    setLoading(true)
    try {
      const data = await predict(teamA, teamB, Number(year))
      setResult(data)
    } catch (err) {
      setError(err.message)
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card">
      <form className="predict-form" onSubmit={handlePredict}>
        <div className="field">
          <label>Team A</label>
          <select value={teamA} onChange={(e) => setTeamA(e.target.value)}>
            {teams.map((t) => <option key={t.team} value={t.team}>{t.team}</option>)}
          </select>
        </div>
        <div className="vs">vs</div>
        <div className="field">
          <label>Team B</label>
          <select value={teamB} onChange={(e) => setTeamB(e.target.value)}>
            {teams.map((t) => <option key={t.team} value={t.team}>{t.team}</option>)}
          </select>
        </div>
        <div className="field">
          <label>FIFA edition year</label>
          <select value={year} onChange={(e) => setYear(e.target.value)} disabled={!years.length}>
            {years.map((y) => <option key={y} value={y}>{y}</option>)}
          </select>
        </div>
        <button type="submit" disabled={loading || !years.length || teamA === teamB}>
          {loading ? 'Predicting…' : 'Predict'}
        </button>
      </form>

      {teamA && teamB && years.length > 0 && (
        <div className="request-preview">
          <span className="prompt">&gt;</span> predict(team_a=<span className="arg">"{teamA}"</span>, team_b=<span className="arg">"{teamB}"</span>, year=<span className="arg">{year}</span>)
        </div>
      )}

      {!years.length && teamA && teamB && teamA !== teamB && (
        <p className="hint">No overlapping FIFA edition year for these two teams.</p>
      )}
      {error && <p className="error">{error}</p>}

      {result && (
        <div className="result">
          <div className="result-head">
            <span className="result-status">predicted</span>
            <span className="result-matchup">{result.team_a} vs {result.team_b} · {result.year}</span>
          </div>
          <div className="winner">
            <span className="trophy">&gt;</span>{result.winner} wins
          </div>
          {(() => {
            const votes = Object.values(result.model_votes)
            const total = votes.length
            const agree = votes.filter((v) => v === result.winner).length
            const loser = result.winner === result.team_a ? result.team_b : result.team_a
            const winnerPct = Math.round(result.confidence * 100)
            const loserPct = 100 - winnerPct
            return (
              <div className="consensus">
                <div className="consensus-value">
                  {winnerPct}<span className="unit">%</span>
                  <span className="sub">predicted probability · {agree}/{total} models agree</span>
                </div>
                <div className="consensus-bar">
                  <div className="consensus-bar-fill" style={{ transform: `scaleX(${result.confidence})` }} />
                </div>
                <div className="consensus-legend">
                  <span><strong>{winnerPct}%</strong> {result.winner}</span>
                  <span><strong>{loserPct}%</strong> {loser}</span>
                </div>
              </div>
            )
          })()}
          <div className="votes">
            <h4>Model votes</h4>
            <ul>
              {Object.entries(result.model_votes).map(([model, pick]) => {
                const pct = Math.round(result.model_confidence[model] * 100)
                return (
                  <li key={model}>
                    <span className="model-name">{MODEL_LABEL[model] ?? model}</span>
                    <span className="model-pick">{pick} <span className="model-pct">{pct}%</span></span>
                  </li>
                )
              })}
            </ul>
          </div>
        </div>
      )}
    </div>
  )
}
