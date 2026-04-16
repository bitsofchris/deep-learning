import { useState, useEffect } from 'react'
import { api } from '../api'
import type { SimilarityResult } from '../types'

interface Props {
  selectedIds: number[]
}

function simColor(v: number): { backgroundColor: string; color: string } {
  // Indigo cells that deepen with similarity
  const alpha = 0.08 + v * 0.88
  return {
    backgroundColor: `rgba(99,102,241,${alpha})`,
    color: v > 0.6 ? '#e2e8f0' : '#818cf8',
  }
}

function truncate(text: string, n = 22): string {
  return text.length > n ? text.slice(0, n) + '…' : text
}

export default function SimilarityMatrix({ selectedIds }: Props) {
  const [result, setResult] = useState<SimilarityResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (selectedIds.length < 2) { setResult(null); return }
    setLoading(true)
    setError('')
    api.similarity(selectedIds)
      .then(setResult)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [selectedIds.join(',')])  // eslint-disable-line react-hooks/exhaustive-deps

  if (selectedIds.length < 2) {
    return (
      <div className="panel">
        <div className="panel-title">Similarity Matrix</div>
        <div className="empty">Select 2 or more texts in the Cache tab.</div>
      </div>
    )
  }

  return (
    <div className="panel">
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20 }}>
        <span className="panel-title" style={{ marginBottom: 0 }}>Similarity Matrix</span>
        <span style={{ fontSize: 11, color: '#374151' }}>cosine similarity · deeper = more similar</span>
      </div>

      {loading && <div className="loading"><div className="spinner" />Computing…</div>}
      {error && <div className="error-box">{error}</div>}

      {!loading && result && (
        <div className="sim-matrix">
          <table>
            <thead>
              <tr>
                <th />
                {result.texts.map((t, j) => (
                  <th key={j} title={t}>{truncate(t)}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.matrix.map((row, i) => (
                <tr key={i}>
                  <th title={result.texts[i]}>{truncate(result.texts[i])}</th>
                  {row.map((val, j) => (
                    <td key={j} style={simColor(val)} title={`${result.texts[i]} × ${result.texts[j]}`}>
                      {val.toFixed(3)}
                    </td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>

          {/* Sorted pairs */}
          <div style={{ marginTop: 28 }}>
            <div className="section-label" style={{ marginBottom: 10 }}>All pairs ranked by similarity</div>
            {result.matrix
              .flatMap((row, i) =>
                row.slice(i + 1).map((val, jOffset) => ({
                  val,
                  a: result.texts[i],
                  b: result.texts[i + 1 + jOffset],
                }))
              )
              .sort((a, b) => b.val - a.val)
              .map(({ val, a, b }, idx) => (
                <div key={idx} className="result-row">
                  <span className="result-text">
                    <span style={{ color: '#6366f1' }}>{truncate(a, 40)}</span>
                    <span style={{ color: '#374151', margin: '0 8px' }}>×</span>
                    <span style={{ color: '#f43f5e' }}>{truncate(b, 40)}</span>
                  </span>
                  <span className="sim-badge">{val.toFixed(4)}</span>
                </div>
              ))}
          </div>
        </div>
      )}
    </div>
  )
}
