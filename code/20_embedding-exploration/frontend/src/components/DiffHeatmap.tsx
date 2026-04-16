import { useState, useEffect } from 'react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts'
import { api } from '../api'
import type { VectorItem } from '../types'

const SAMPLE_STEP = 12
const TOP_N = 15

interface Props {
  selectedIds: number[]
}

export default function DiffHeatmap({ selectedIds }: Props) {
  const [vectors, setVectors] = useState<VectorItem[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const needsTwo = selectedIds.length !== 2

  useEffect(() => {
    if (selectedIds.length !== 2) { setVectors([]); return }
    setLoading(true)
    setError('')
    api.getVectors(selectedIds)
      .then(r => setVectors(r.items))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [selectedIds.join(',')])  // eslint-disable-line react-hooks/exhaustive-deps

  if (needsTwo) {
    return (
      <div className="panel">
        <div className="panel-title">Difference Heatmap</div>
        <div className="empty">
          {selectedIds.length === 0
            ? 'Select exactly 2 texts in the Cache tab.'
            : `${selectedIds.length} selected — select exactly 2.`}
        </div>
      </div>
    )
  }

  const [vecA, vecB] = vectors
  if (!vecA || !vecB) {
    return (
      <div className="panel">
        <div className="panel-title">Difference Heatmap</div>
        {loading && <div className="loading"><div className="spinner" />Loading vectors…</div>}
        {error && <div className="error-box">{error}</div>}
      </div>
    )
  }

  // Compute |A − B| per dimension
  const diff = vecA.embedding.map((v, i) => Math.abs(v - (vecB.embedding[i] ?? 0)))
  const maxDiff = Math.max(...diff, 1e-9)

  // Sampled curve data
  const sampleDims = Array.from({ length: Math.ceil(3072 / SAMPLE_STEP) }, (_, i) => i * SAMPLE_STEP)
  const curveData = sampleDims.map(dim => ({ dim, diff: diff[dim] ?? 0 }))

  // Top-N most discriminating dimensions
  const topDims = diff
    .map((d, i) => ({ dim: i, diff: d }))
    .sort((a, b) => b.diff - a.diff)
    .slice(0, TOP_N)

  // 1D heatmap strip (sampled)
  const stripCells = sampleDims.map(dim => ({
    dim,
    intensity: (diff[dim] ?? 0) / maxDiff,
  }))

  return (
    <div className="panel">
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
        <span className="panel-title" style={{ marginBottom: 0 }}>Difference Heatmap</span>
        <span style={{ fontSize: 11, color: '#374151' }}>|A − B| per dimension</span>
      </div>

      {loading && <div className="loading"><div className="spinner" />Loading vectors…</div>}
      {error && <div className="error-box">{error}</div>}

      {!loading && vecA && vecB && (
        <>
          {/* Labels */}
          <div className="flex-row" style={{ marginBottom: 16 }}>
            <span style={{ fontSize: 13, color: '#6366f1', background: '#131622', padding: '3px 10px', borderRadius: 99 }}>
              A: {vecA.text.slice(0, 60)}
            </span>
            <span style={{ fontSize: 13, color: '#f43f5e', background: '#131622', padding: '3px 10px', borderRadius: 99 }}>
              B: {vecB.text.slice(0, 60)}
            </span>
          </div>

          {/* 1D heatmap strip */}
          <div style={{ marginBottom: 16 }}>
            <div className="section-label">Heatmap strip — red = high divergence</div>
            <div style={{ display: 'flex', height: 28, gap: 1, overflowX: 'auto', borderRadius: 6, overflow: 'hidden' }}>
              {stripCells.map(({ dim, intensity }) => (
                <div
                  key={dim}
                  title={`Dim ${dim}: ${(diff[dim] ?? 0).toFixed(5)}`}
                  style={{
                    width: 4,
                    height: '100%',
                    backgroundColor: `rgba(239,68,68,${0.1 + intensity * 0.9})`,
                    flexShrink: 0,
                  }}
                />
              ))}
            </div>
          </div>

          {/* Area chart of diff curve */}
          <div style={{ marginBottom: 24 }}>
            <div className="section-label">Difference curve</div>
            <ResponsiveContainer width="100%" height={220}>
              <AreaChart data={curveData} margin={{ top: 5, right: 20, left: 0, bottom: 20 }}>
                <defs>
                  <linearGradient id="diffGrad" x1="0" y1="0" x2="0" y2="1">
                    <stop offset="5%" stopColor="#f43f5e" stopOpacity={0.6} />
                    <stop offset="95%" stopColor="#f43f5e" stopOpacity={0.05} />
                  </linearGradient>
                </defs>
                <CartesianGrid strokeDasharray="3 3" stroke="#1e2235" />
                <XAxis
                  dataKey="dim"
                  tick={{ fill: '#374151', fontSize: 11 }}
                  label={{ value: 'Dimension index', position: 'insideBottom', offset: -10, fill: '#64748b', fontSize: 12 }}
                />
                <YAxis tick={{ fill: '#374151', fontSize: 11 }} />
                <Tooltip
                  contentStyle={{ background: '#0f1120', border: '1px solid #1e2235', borderRadius: 8, fontSize: 12 }}
                  labelFormatter={v => `Dim ${v}`}
                  formatter={(v: number) => [v.toFixed(5), '|A − B|']}
                />
                <Area
                  type="monotone"
                  dataKey="diff"
                  stroke="#f43f5e"
                  fill="url(#diffGrad)"
                  dot={false}
                  isAnimationActive={false}
                  strokeWidth={1.5}
                />
              </AreaChart>
            </ResponsiveContainer>
          </div>

          {/* Top-N table */}
          <div>
            <div className="section-label">Top {TOP_N} most discriminating dimensions</div>
            <table className="diff-table">
              <thead>
                <tr>
                  <th>Rank</th>
                  <th>Dim index</th>
                  <th>A value</th>
                  <th>B value</th>
                  <th>|A − B|</th>
                </tr>
              </thead>
              <tbody>
                {topDims.map(({ dim, diff: d }, rank) => (
                  <tr key={dim}>
                    <td style={{ color: '#64748b' }}>{rank + 1}</td>
                    <td style={{ color: '#818cf8' }}>{dim}</td>
                    <td style={{ color: '#6366f1' }}>{(vecA.embedding[dim] ?? 0).toFixed(5)}</td>
                    <td style={{ color: '#f43f5e' }}>{(vecB.embedding[dim] ?? 0).toFixed(5)}</td>
                    <td>{d.toFixed(5)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </>
      )}
    </div>
  )
}
