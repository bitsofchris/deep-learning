import { useState, useEffect, useMemo } from 'react'
import {
  ScatterChart, Scatter, XAxis, YAxis, CartesianGrid, Tooltip,
  ResponsiveContainer,
} from 'recharts'
import { api } from '../api'
import type { UmapPoint, TagFilters } from '../types'

const DIM_OPTIONS = [32, 128, 256, 512, 1024, 3072]
const FALLBACK_COLORS = ['#6366f1', '#f43f5e', '#22c55e', '#f59e0b', '#06b6d4', '#a855f7', '#ec4899', '#10b981']

// Stable named colors for known tag values
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
  filters: TagFilters
  tagColumns: string[]
  tagValues: Record<string, string[]>
}

export default function UmapScatter({ selectedIds, filters, tagColumns, tagValues }: Props) {
  const [points, setPoints] = useState<UmapPoint[]>([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [computed, setComputed] = useState(false)
  const [dims, setDims] = useState<number>(3072)
  const [colorBy, setColorBy] = useState<string>('')
  const [computedDims, setComputedDims] = useState<number>(0)

  const handleCompute = async () => {
    setLoading(true)
    setError('')
    try {
      const r = await api.umap({ filters, dims })
      setPoints(r.points)
      setComputed(true)
      setComputedDims(r.dims)
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Error')
    } finally {
      setLoading(false)
    }
  }

  // Auto-recompute when dims or filters change AFTER first compute
  useEffect(() => {
    if (!computed) return
    handleCompute()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dims, JSON.stringify(filters)])

  // Group points by selected tag for coloring
  const legend = useMemo(() => {
    if (!colorBy) return [] as { value: string; color: string; count: number }[]
    const counts = new Map<string, number>()
    for (const p of points) {
      const v = p.tags[colorBy] ?? '(untagged)'
      counts.set(v, (counts.get(v) ?? 0) + 1)
    }
    return Array.from(counts.entries()).map(([value, count], i) => ({
      value, count, color: colorForValue(value, i),
    }))
  }, [colorBy, points])

  return (
    <div className="panel">
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16, flexWrap: 'wrap' }}>
        <span className="panel-title" style={{ marginBottom: 0 }}>UMAP Projection</span>
        <span style={{ fontSize: 11, color: '#374151' }}>
          2D projection · cosine metric · {points.length} points shown
        </span>
      </div>

      {/* Controls */}
      <div className="flex-row" style={{ marginBottom: 18, flexWrap: 'wrap' }}>
        <button className="btn btn-primary" onClick={handleCompute} disabled={loading}>
          {loading ? 'Computing…' : computed ? 'Recompute' : 'Compute UMAP'}
        </button>

        <div className="dim-toggle">
          <span className="section-label" style={{ margin: 0, marginRight: 6 }}>Dims</span>
          {DIM_OPTIONS.map(d => (
            <button
              key={d}
              className={`chip${dims === d ? ' active' : ''}`}
              onClick={() => setDims(d)}
            >
              {d}
            </button>
          ))}
        </div>

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
      </div>

      {loading && <div className="loading"><div className="spinner" />Running UMAP at {dims}d…</div>}
      {error && <div className="error-box">{error}</div>}

      {!loading && points.length > 0 && (
        <>
          <div className="info-box" style={{ marginBottom: 16 }}>
            {computedDims !== 3072 && <>Truncated to first <b>{computedDims}</b> dims (Matryoshka). </>}
            {selectedIds.length > 0 && <>Selected items ringed in white. </>}
            Hover for full text.
          </div>

          {/* Legend */}
          {legend.length > 0 && (
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

          <ResponsiveContainer width="100%" height={560}>
            <ScatterChart margin={{ top: 20, right: 40, left: 0, bottom: 20 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e2235" />
              <XAxis
                type="number" dataKey="x" domain={[-1.1, 1.1]}
                tick={{ fill: '#374151', fontSize: 11 }}
                tickFormatter={v => v.toFixed(1)}
              />
              <YAxis
                type="number" dataKey="y" domain={[-1.1, 1.1]}
                tick={{ fill: '#374151', fontSize: 11 }}
                tickFormatter={v => v.toFixed(1)}
              />
              <Tooltip
                cursor={{ strokeDasharray: '3 3', stroke: '#374151' }}
                content={({ payload }) => {
                  if (!payload?.length) return null
                  const p = payload[0].payload as UmapPoint
                  return (
                    <div style={{ background: '#0f1120', border: '1px solid #1e2235', borderRadius: 8, padding: '10px 12px', fontSize: 12, maxWidth: 320 }}>
                      <div style={{ color: '#818cf8', marginBottom: 4, fontWeight: 600 }}>{p.text}</div>
                      {Object.keys(p.tags).length > 0 && (
                        <div style={{ color: '#64748b', fontSize: 11, marginBottom: 4 }}>
                          {Object.entries(p.tags).map(([k, v]) => `${k}:${v}`).join(' · ')}
                        </div>
                      )}
                      <div style={{ color: '#374151', fontSize: 11 }}>({p.x.toFixed(3)}, {p.y.toFixed(3)})</div>
                    </div>
                  )
                }}
              />
              <Scatter
                data={points}
                shape={(props: object) => {
                  const { cx = 0, cy = 0, payload } = props as { cx?: number; cy?: number; payload?: UmapPoint }
                  if (!payload) return <g />
                  const isSelected = selectedIds.includes(payload.id)
                  const colorValue = colorBy ? payload.tags[colorBy] : undefined
                  const legendIdx = legend.findIndex(l => l.value === (colorValue ?? '(untagged)'))
                  const color = colorBy
                    ? colorForValue(colorValue, legendIdx >= 0 ? legendIdx : 0)
                    : '#6366f1'
                  const r = isSelected ? 7 : 4
                  return (
                    <g>
                      <circle cx={cx} cy={cy} r={r} fill={color}
                        fillOpacity={isSelected ? 1 : 0.75}
                        stroke={isSelected ? '#fff' : 'none'}
                        strokeWidth={1.5} />
                    </g>
                  )
                }}
              />
            </ScatterChart>
          </ResponsiveContainer>
        </>
      )}
    </div>
  )
}
