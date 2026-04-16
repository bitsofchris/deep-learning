import { useState, useEffect } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ReferenceLine,
  ResponsiveContainer, Label,
} from 'recharts'
import { api } from '../api'
import type { MatryoshkaResult } from '../types'

interface Props {
  selectedIds: number[]
}

export default function MatryoshkaCurve({ selectedIds }: Props) {
  const [result, setResult] = useState<MatryoshkaResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const needsTwo = selectedIds.length !== 2

  useEffect(() => {
    if (selectedIds.length !== 2) { setResult(null); return }
    setLoading(true)
    setError('')
    api.matryoshka(selectedIds[0], selectedIds[1])
      .then(setResult)
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [selectedIds.join(',')])  // eslint-disable-line react-hooks/exhaustive-deps

  if (needsTwo) {
    return (
      <div className="panel">
        <div className="panel-title">Matryoshka Decay Curve</div>
        <div className="empty">
          {selectedIds.length === 0
            ? 'Select exactly 2 texts in the Cache tab.'
            : `${selectedIds.length} selected — select exactly 2.`}
        </div>
      </div>
    )
  }

  const data = result
    ? result.dims.map((d, i) => ({ dims: d, similarity: result.similarities[i] }))
    : []

  // Find the elbow: biggest drop between consecutive dims
  let maxDrop = 0
  let elbowDim = 0
  if (data.length > 1) {
    for (let i = 1; i < data.length; i++) {
      const drop = data[i - 1].similarity - data[i].similarity
      if (drop > maxDrop) { maxDrop = drop; elbowDim = data[i].dims }
    }
  }

  return (
    <div className="panel">
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16 }}>
        <span className="panel-title" style={{ marginBottom: 0 }}>Matryoshka Decay Curve</span>
        <span style={{ fontSize: 11, color: '#374151' }}>
          cosine similarity as dims truncate from 256 → 3072
        </span>
      </div>

      {loading && <div className="loading"><div className="spinner" />Computing…</div>}
      {error && <div className="error-box">{error}</div>}

      {!loading && result && (
        <>
          <div className="flex-row" style={{ marginBottom: 16 }}>
            <span style={{ fontSize: 13, color: '#6366f1', background: '#131622', padding: '3px 10px', borderRadius: 99 }}>
              A: {result.text_a.slice(0, 60)}
            </span>
            <span style={{ fontSize: 13, color: '#f43f5e', background: '#131622', padding: '3px 10px', borderRadius: 99 }}>
              B: {result.text_b.slice(0, 60)}
            </span>
          </div>

          <ResponsiveContainer width="100%" height={320}>
            <LineChart data={data} margin={{ top: 10, right: 30, left: 0, bottom: 30 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2235" />
              <XAxis
                dataKey="dims"
                type="number"
                scale="log"
                domain={['dataMin', 'dataMax']}
                ticks={[256, 512, 1024, 2048, 3072]}
                tick={{ fill: '#374151', fontSize: 11 }}
              >
                <Label value="Dimensions (log scale)" position="insideBottom" offset={-15} fill="#64748b" fontSize={12} />
              </XAxis>
              <YAxis
                domain={['auto', 'auto']}
                tick={{ fill: '#374151', fontSize: 11 }}
              >
                <Label value="Cosine similarity" angle={-90} position="insideLeft" offset={15} fill="#64748b" fontSize={12} />
              </YAxis>
              <Tooltip
                contentStyle={{ background: '#0f1120', border: '1px solid #1e2235', borderRadius: 8, fontSize: 12 }}
                labelFormatter={v => `${v} dims`}
                formatter={(v: number) => [v.toFixed(4), 'Cosine sim']}
              />
              {maxDrop > 0.005 && (
                <ReferenceLine
                  x={elbowDim}
                  stroke="#f59e0b"
                  strokeDasharray="4 4"
                  label={{ value: `elbow @ ${elbowDim}d`, fill: '#f59e0b', fontSize: 11, position: 'insideTopRight' }}
                />
              )}
              <Line
                type="monotone"
                dataKey="similarity"
                stroke="#6366f1"
                strokeWidth={2.5}
                dot={{ fill: '#6366f1', r: 5 }}
                isAnimationActive={false}
              />
            </LineChart>
          </ResponsiveContainer>

          {/* Scores table */}
          <div style={{ marginTop: 20 }}>
            <div className="section-label" style={{ marginBottom: 8 }}>Scores by dimensionality</div>
            <div className="flex-row" style={{ flexWrap: 'wrap' }}>
              {data.map(({ dims, similarity }) => (
                <div
                  key={dims}
                  style={{
                    background: '#131622',
                    border: '1px solid #1e2235',
                    borderRadius: 8,
                    padding: '8px 14px',
                    textAlign: 'center',
                    minWidth: 80,
                  }}
                >
                  <div style={{ fontSize: 11, color: '#64748b', marginBottom: 4 }}>{dims}d</div>
                  <div style={{ fontSize: 16, fontWeight: 700, color: '#818cf8' }}>{similarity.toFixed(3)}</div>
                </div>
              ))}
            </div>
          </div>
        </>
      )}
    </div>
  )
}
