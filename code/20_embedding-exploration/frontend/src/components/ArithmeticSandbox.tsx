import { useState } from 'react'
import { api } from '../api'
import type { CacheItem, ArithmeticResult } from '../types'

interface Props {
  allItems: CacheItem[]
}

function Picker({
  label,
  color,
  value,
  items,
  onChange,
}: {
  label: string
  color: string
  value: number | null
  items: CacheItem[]
  onChange: (id: number) => void
}) {
  return (
    <div style={{ flex: 1, minWidth: 180 }}>
      <div className="section-label" style={{ color, marginBottom: 6 }}>{label}</div>
      <select
        style={{ width: '100%' }}
        value={value ?? ''}
        onChange={e => onChange(Number(e.target.value))}
      >
        <option value="">— select —</option>
        {items.map(it => (
          <option key={it.id} value={it.id}>
            {it.text.slice(0, 70)}
          </option>
        ))}
      </select>
    </div>
  )
}

export default function ArithmeticSandbox({ allItems }: Props) {
  const [idA, setIdA] = useState<number | null>(null)
  const [idB, setIdB] = useState<number | null>(null)
  const [idC, setIdC] = useState<number | null>(null)
  const [result, setResult] = useState<ArithmeticResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const ready = idA !== null && idB !== null && idC !== null

  const handleCompute = async () => {
    if (!ready) return
    setLoading(true)
    setError('')
    setResult(null)
    try {
      const r = await api.arithmetic(idA!, idB!, idC!)
      setResult(r)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Error')
    } finally {
      setLoading(false)
    }
  }

  if (allItems.length < 3) {
    return (
      <div className="panel">
        <div className="panel-title">Arithmetic Sandbox</div>
        <div className="empty">Add at least 3 texts to the cache to use this view.</div>
      </div>
    )
  }

  return (
    <div className="panel">
      <div style={{ marginBottom: 20 }}>
        <span className="panel-title">Arithmetic Sandbox</span>
      </div>

      <div className="info-box" style={{ marginBottom: 20 }}>
        Computes <strong>A − B + C</strong> in embedding space and finds the nearest cached texts.
        Classic analogy test: if A=king, B=man, C=woman → result ≈ queen?
      </div>

      {/* Formula display */}
      <div style={{
        display: 'flex',
        alignItems: 'center',
        gap: 8,
        marginBottom: 20,
        fontSize: 18,
        fontWeight: 700,
        color: '#64748b',
        flexWrap: 'wrap',
      }}>
        <span style={{ color: '#6366f1' }}>A</span>
        <span>−</span>
        <span style={{ color: '#f43f5e' }}>B</span>
        <span>+</span>
        <span style={{ color: '#22c55e' }}>C</span>
        <span>=</span>
        <span style={{ color: '#818cf8' }}>?</span>
      </div>

      {/* Pickers */}
      <div className="flex-row" style={{ marginBottom: 16, alignItems: 'flex-start', flexWrap: 'wrap' }}>
        <Picker label="A" color="#6366f1" value={idA} items={allItems} onChange={setIdA} />
        <Picker label="B" color="#f43f5e" value={idB} items={allItems} onChange={setIdB} />
        <Picker label="C" color="#22c55e" value={idC} items={allItems} onChange={setIdC} />
      </div>

      <div className="flex-row" style={{ marginBottom: 20 }}>
        <button className="btn btn-primary" onClick={handleCompute} disabled={!ready || loading}>
          {loading ? 'Computing…' : 'Compute A − B + C'}
        </button>
      </div>

      {loading && <div className="loading"><div className="spinner" />Computing…</div>}
      {error && <div className="error-box">{error}</div>}

      {!loading && result && (
        <div>
          {/* Equation recap */}
          <div style={{
            background: '#131622',
            border: '1px solid #1e2235',
            borderRadius: 8,
            padding: '14px 16px',
            marginBottom: 20,
            fontSize: 13,
          }}>
            <span style={{ color: '#6366f1' }}>{result.text_a}</span>
            <span style={{ color: '#64748b', margin: '0 10px' }}>−</span>
            <span style={{ color: '#f43f5e' }}>{result.text_b}</span>
            <span style={{ color: '#64748b', margin: '0 10px' }}>+</span>
            <span style={{ color: '#22c55e' }}>{result.text_c}</span>
            <span style={{ color: '#64748b', margin: '0 10px' }}>=</span>
            <span style={{ color: '#818cf8', fontWeight: 600 }}>?</span>
          </div>

          {result.nearest.length === 0 ? (
            <div className="info-box">
              No other texts in cache to compare against. Add more texts first.
            </div>
          ) : (
            <>
              <div className="section-label" style={{ marginBottom: 10 }}>Nearest neighbors in cache</div>
              {result.nearest.map((n, i) => (
                <div key={n.id} className="result-row">
                  <span style={{ fontSize: 12, color: '#374151', width: 20, flexShrink: 0 }}>#{i + 1}</span>
                  <span className="result-text">{n.text}</span>
                  <span className="sim-badge">{n.similarity.toFixed(4)}</span>
                </div>
              ))}
            </>
          )}
        </div>
      )}
    </div>
  )
}
