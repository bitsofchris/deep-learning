import { useState, useEffect, useMemo } from 'react'
import {
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
} from 'recharts'
import { api } from '../api'
import type { VectorItem } from '../types'

const FALLBACK_COLORS = ['#6366f1', '#f43f5e', '#22c55e', '#f59e0b', '#06b6d4', '#a855f7', '#ec4899', '#10b981']
const SAMPLE_STEP = 12  // sample every 12th dim → 256 points per line

// Stable colors for known tag values (matches UMAP/Cache)
const TAG_COLORS: Record<string, string> = {
  male: '#60a5fa',
  female: '#f472b6',
  royal: '#facc15',
  common: '#94a3b8',
  short: '#34d399',
  long: '#a78bfa',
}

function colorForValue(value: string | undefined, index: number): string {
  if (value && TAG_COLORS[value]) return TAG_COLORS[value]
  return FALLBACK_COLORS[index % FALLBACK_COLORS.length]
}

interface Props {
  selectedIds: number[]
  tagColumns: string[]
  tagValues: Record<string, string[]>
}

export default function FingerprintChart({ selectedIds, tagColumns, tagValues }: Props) {
  const [vectors, setVectors] = useState<VectorItem[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [colorBy, setColorByState] = useState<string>('')
  const [userPickedColor, setUserPickedColor] = useState(false)
  const [showLabels, setShowLabels] = useState(false)

  // Wrapper so every UI change counts as an explicit user pick
  const setColorBy = (val: string) => {
    setUserPickedColor(true)
    setColorByState(val)
  }

  useEffect(() => {
    if (selectedIds.length === 0) { setVectors([]); return }
    setLoading(true)
    setError('')
    api.getVectors(selectedIds)
      .then(r => {
        setVectors(r.items)
        // Auto-pick a color tag on big fresh selections — only if user hasn't chosen yet
        if (!userPickedColor && r.items.length > 8) {
          const firstWithValues = tagColumns.find(c => (tagValues[c]?.length ?? 0) > 1)
          if (firstWithValues) setColorByState(firstWithValues)
        }
      })
      .catch(e => setError(e.message))
      .finally(() => setLoading(false))
  }, [selectedIds.join(',')])  // eslint-disable-line react-hooks/exhaustive-deps

  // Group info per tag value (stable order: first-seen)
  const legend = useMemo(() => {
    if (!colorBy) return [] as { value: string; color: string; count: number }[]
    const seen: string[] = []
    const counts = new Map<string, number>()
    for (const v of vectors) {
      const val = v.tags[colorBy] ?? '(untagged)'
      if (!counts.has(val)) seen.push(val)
      counts.set(val, (counts.get(val) ?? 0) + 1)
    }
    return seen.map((value, i) => ({
      value, count: counts.get(value) ?? 0, color: colorForValue(value, i),
    }))
  }, [colorBy, vectors])

  const colorForVector = (v: VectorItem, idx: number): string => {
    if (colorBy) {
      const val = v.tags[colorBy] ?? '(untagged)'
      const entry = legend.find(l => l.value === val)
      return entry?.color ?? FALLBACK_COLORS[idx % FALLBACK_COLORS.length]
    }
    return FALLBACK_COLORS[idx % FALLBACK_COLORS.length]
  }

  if (selectedIds.length === 0) {
    return (
      <div className="panel">
        <div className="panel-title">Fingerprint Chart</div>
        <div className="empty">Select one or more texts in the Data tab to compare their fingerprints.</div>
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

  // Fade lines heavily when many are overlaid — reveals the envelope of each group
  const lineOpacity = showLabels ? 1 : (vectors.length > 20 ? 0.3 : vectors.length > 8 ? 0.55 : 0.9)

  return (
    <div className="panel">
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16, flexWrap: 'wrap' }}>
        <span className="panel-title" style={{ marginBottom: 0 }}>Fingerprint Chart</span>
        <span style={{ fontSize: 11, color: '#374151' }}>
          dim index (0–3071, every {SAMPLE_STEP}) × value · {vectors.length} line{vectors.length === 1 ? '' : 's'}
        </span>
      </div>

      {/* Controls */}
      <div className="flex-row" style={{ marginBottom: 16, flexWrap: 'wrap' }}>
        {tagColumns.length > 0 && (
          <div className="flex-row">
            <span className="section-label" style={{ margin: 0 }}>Color by</span>
            <button
              className={`chip${colorBy === '' ? ' active' : ''}`}
              onClick={() => setColorBy('')}
            >none</button>
            {tagColumns.filter(c => (tagValues[c]?.length ?? 0) > 0).map(c => (
              <button
                key={c}
                className={`chip${colorBy === c ? ' active' : ''}`}
                onClick={() => setColorBy(c)}
              >{c}</button>
            ))}
          </div>
        )}

        <label style={{
          display: 'inline-flex', alignItems: 'center', gap: 6,
          fontSize: 12, color: '#cbd5e1', cursor: 'pointer', userSelect: 'none',
        }}>
          <input
            type="checkbox"
            checked={showLabels}
            onChange={e => setShowLabels(e.target.checked)}
            style={{ accentColor: '#6366f1' }}
          />
          Show text labels
        </label>
      </div>

      {loading && <div className="loading"><div className="spinner" /> Loading vectors…</div>}
      {error && <div className="error-box">{error}</div>}

      {!loading && vectors.length > 0 && (
        <>
          {/* Legend — tag-value counts (compact) OR per-text pills (when showLabels) */}
          {legend.length > 0 && !showLabels && (
            <div className="flex-row" style={{ marginBottom: 12, flexWrap: 'wrap' }}>
              {legend.map(l => (
                <span key={l.value} style={{
                  display: 'inline-flex', alignItems: 'center', gap: 6,
                  fontSize: 12, color: '#cbd5e1',
                  background: '#131622', padding: '3px 10px', borderRadius: 99,
                }}>
                  <span style={{ width: 10, height: 10, borderRadius: '50%', background: l.color }} />
                  {l.value} · {l.count}
                </span>
              ))}
            </div>
          )}

          <div className="info-box" style={{ marginBottom: 16 }}>
            {showLabels
              ? 'Hover any line to see which text it belongs to.'
              : vectors.length > 8
                ? 'Lines faded to reveal group envelopes. Toggle "Show text labels" for per-line identity.'
                : 'Hover to inspect individual dimensions.'}
          </div>

          <ResponsiveContainer width="100%" height={420}>
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
                cursor={{ strokeDasharray: '3 3', stroke: '#374151' }}
                content={({ active, payload, label }) => {
                  if (!active || !payload?.length) return null
                  const sorted = [...payload].sort(
                    (a, b) => (Number(b.value) ?? 0) - (Number(a.value) ?? 0),
                  )
                  return (
                    <div style={{
                      background: '#0f1120', border: '1px solid #1e2235', borderRadius: 8,
                      padding: '10px 12px', fontSize: 12, maxHeight: 360, overflowY: 'auto',
                      minWidth: 280, maxWidth: 480,
                    }}>
                      <div style={{ color: '#64748b', marginBottom: 6, fontSize: 11 }}>
                        Dim {label} · {sorted.length} line{sorted.length === 1 ? '' : 's'} · sorted ↓
                      </div>
                      {sorted.map((p, i) => {
                        const item = vectors.find(x => `id_${x.id}` === p.dataKey)
                        const text = item?.text.slice(0, 60) ?? String(p.dataKey)
                        return (
                          <div key={i} style={{
                            display: 'flex', gap: 10, alignItems: 'baseline',
                            padding: '2px 0',
                          }}>
                            <span style={{
                              color: p.color as string, fontFamily: "'JetBrains Mono', monospace",
                              minWidth: 72, textAlign: 'right', fontWeight: 600,
                            }}>
                              {Number(p.value).toFixed(4)}
                            </span>
                            <span style={{ color: '#cbd5e1' }}>{text}</span>
                          </div>
                        )
                      })}
                    </div>
                  )
                }}
              />
              {vectors.map((v, i) => (
                <Line
                  key={v.id}
                  type="monotone"
                  dataKey={`id_${v.id}`}
                  name={`id_${v.id}`}
                  stroke={colorForVector(v, i)}
                  strokeOpacity={lineOpacity}
                  dot={false}
                  isAnimationActive={false}
                  strokeWidth={1.2}
                />
              ))}
            </LineChart>
          </ResponsiveContainer>

          {/* Per-text pills — only when "Show text labels" is on */}
          {showLabels && (
            <div style={{ marginTop: 16, display: 'flex', flexWrap: 'wrap', gap: 8 }}>
              {vectors.map((v, i) => (
                <span key={v.id} style={{
                  fontSize: 12, color: colorForVector(v, i),
                  background: '#131622', padding: '3px 10px', borderRadius: 99,
                }}>
                  {v.text.slice(0, 60)}
                </span>
              ))}
            </div>
          )}
        </>
      )}
    </div>
  )
}
