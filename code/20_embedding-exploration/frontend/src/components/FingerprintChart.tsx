import { useState, useEffect } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer,
} from 'recharts'
import { api } from '../api'
import type { VectorItem } from '../types'

const COLORS = ['#6366f1', '#f43f5e', '#22c55e', '#f59e0b', '#06b6d4', '#a855f7', '#ec4899', '#10b981']
const SAMPLE_STEP = 12  // sample every 12th dim → 256 points per line

interface Props {
  selectedIds: number[]
}

export default function FingerprintChart({ selectedIds }: Props) {
  const [vectors, setVectors] = useState<VectorItem[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  useEffect(() => {
    if (selectedIds.length === 0) { setVectors([]); return }
    setLoading(true)
    setError('')
    api.getVectors(selectedIds)
      .then(r => setVectors(r.items))
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [selectedIds.join(',')])  // eslint-disable-line react-hooks/exhaustive-deps

  if (selectedIds.length === 0) {
    return (
      <div className="panel">
        <div className="panel-title">Fingerprint Chart</div>
        <div className="empty">Select one or more texts in the Cache tab to compare their fingerprints.</div>
      </div>
    )
  }

  const sampleDims = Array.from(
    { length: Math.ceil(3072 / SAMPLE_STEP) },
    (_, i) => i * SAMPLE_STEP,
  )

  const data = sampleDims.map(dim => {
    const point: Record<string, number> = { dim }
    vectors.forEach(v => { point[`id_${v.id}`] = v.embedding[dim] ?? 0 })
    return point
  })

  return (
    <div className="panel">
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 20 }}>
        <span className="panel-title" style={{ marginBottom: 0 }}>Fingerprint Chart</span>
        <span style={{ fontSize: 11, color: '#374151' }}>
          dimension index (0–3071, sampled every {SAMPLE_STEP}) × value
        </span>
      </div>

      {loading && <div className="loading"><div className="spinner" /> Loading vectors…</div>}
      {error && <div className="error-box">{error}</div>}

      {!loading && vectors.length > 0 && (
        <>
          <div className="info-box" style={{ marginBottom: 16 }}>
            Hover to inspect individual dimensions. Each line is one text's 3,072-dim vector.
          </div>
          <ResponsiveContainer width="100%" height={380}>
            <LineChart data={data} margin={{ top: 5, right: 20, left: 0, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2235" />
              <XAxis
                dataKey="dim"
                label={{ value: 'Dimension index', position: 'insideBottom', offset: -10, fill: '#64748b', fontSize: 12 }}
                tick={{ fill: '#374151', fontSize: 11 }}
              />
              <YAxis
                tick={{ fill: '#374151', fontSize: 11 }}
                label={{ value: 'Value', angle: -90, position: 'insideLeft', offset: 10, fill: '#64748b', fontSize: 12 }}
              />
              <Tooltip
                contentStyle={{ background: '#0f1120', border: '1px solid #1e2235', borderRadius: 8, fontSize: 12 }}
                labelFormatter={v => `Dim ${v}`}
                formatter={(value: number, name: string) => {
                  const item = vectors.find(x => `id_${x.id}` === name)
                  return [value.toFixed(5), item?.text.slice(0, 50) ?? name]
                }}
              />
              <Legend
                formatter={value => {
                  const item = vectors.find(x => `id_${x.id}` === value)
                  return <span style={{ fontSize: 12, color: '#cbd5e1' }}>{item?.text.slice(0, 60) ?? value}</span>
                }}
              />
              {vectors.map((v, i) => (
                <Line
                  key={v.id}
                  type="monotone"
                  dataKey={`id_${v.id}`}
                  name={`id_${v.id}`}
                  stroke={COLORS[i % COLORS.length]}
                  dot={false}
                  isAnimationActive={false}
                  strokeWidth={1.2}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </>
      )}

      {/* Selected text labels below chart */}
      {vectors.length > 0 && (
        <div style={{ marginTop: 16, display: 'flex', flexWrap: 'wrap', gap: 8 }}>
          {vectors.map((v, i) => (
            <span key={v.id} style={{ fontSize: 12, color: COLORS[i % COLORS.length], background: '#131622', padding: '3px 10px', borderRadius: 99 }}>
              {v.text.slice(0, 60)}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}
