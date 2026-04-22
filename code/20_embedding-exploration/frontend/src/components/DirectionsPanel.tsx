import { useState, useEffect, useMemo, useCallback } from 'react'
import {
  BarChart, Bar, LineChart, Line, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ReferenceLine, ResponsiveContainer,
} from 'recharts'
import { api } from '../api'
import type {
  Direction,
  ProjectionResult,
  AucScanResult,
  DirectionMatryoshkaResult,
  TagFilters,
} from '../types'

const DIM_OPTIONS = [32, 64, 128, 256, 512, 1024, 3072]
const HIST_BINS = 24
const FALLBACK_COLORS = ['#60a5fa', '#f472b6', '#facc15', '#94a3b8', '#34d399', '#a78bfa']

const TAG_COLORS: Record<string, string> = {
  male: '#60a5fa',
  female: '#f472b6',
  royal: '#facc15',
  common: '#94a3b8',
  short: '#34d399',
  long: '#a78bfa',
  '(none)': '#475569',
}

function colorFor(value: string, i: number): string {
  return TAG_COLORS[value] ?? FALLBACK_COLORS[i % FALLBACK_COLORS.length]
}

interface Props {
  tagColumns: string[]
  tagValues: Record<string, string[]>
  filters: TagFilters
}

export default function DirectionsPanel({ tagColumns, tagValues, filters }: Props) {
  const [directions, setDirections] = useState<Direction[]>([])
  const [cosines, setCosines] = useState<number[][]>([])
  const [activeId, setActiveId] = useState<number | null>(null)
  const [projection, setProjection] = useState<ProjectionResult | null>(null)
  const [aucResult, setAucResult] = useState<AucScanResult | null>(null)
  const [matryoshka, setMatryoshka] = useState<DirectionMatryoshkaResult | null>(null)
  const [projDims, setProjDims] = useState(3072)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  // Create-form state
  const [formName, setFormName] = useState('')
  const [formTag, setFormTag] = useState('')
  const [formA, setFormA] = useState('')
  const [formB, setFormB] = useState('')
  const [randomName, setRandomName] = useState('')

  const refresh = useCallback(async () => {
    try {
      const r = await api.listDirections()
      setDirections(r.directions)
      setCosines(r.cosines)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Error')
    }
  }, [])

  useEffect(() => { refresh() }, [refresh])

  // Default the tag/values of the create form once tagValues arrive
  useEffect(() => {
    if (!formTag && tagColumns.length) {
      const firstWithValues = tagColumns.find(c => (tagValues[c]?.length ?? 0) >= 2)
      if (firstWithValues) {
        setFormTag(firstWithValues)
        setFormA(tagValues[firstWithValues][0])
        setFormB(tagValues[firstWithValues][1])
      }
    }
  }, [tagColumns, tagValues, formTag])

  const active = useMemo(
    () => directions.find(d => d.id === activeId) ?? null,
    [directions, activeId],
  )

  // Load projection / auc / matryoshka when active direction changes
  useEffect(() => {
    setProjection(null)
    setAucResult(null)
    setMatryoshka(null)
    if (!active) return
    setLoading(true)
    setError('')
    const tasks: Promise<unknown>[] = [
      api.projectDirection(active.id, { filters, dims: projDims })
        .then(setProjection),
    ]
    if (active.kind === 'mean_diff' && active.tag && active.value_a && active.value_b) {
      tasks.push(
        api.aucScan({
          tag: active.tag, value_a: active.value_a, value_b: active.value_b,
        }).then(setAucResult),
      )
      tasks.push(
        api.directionMatryoshka({
          tag: active.tag, value_a: active.value_a, value_b: active.value_b,
        }).then(setMatryoshka),
      )
    }
    Promise.all(tasks)
      .catch(e => setError(e instanceof Error ? e.message : 'Error'))
      .finally(() => setLoading(false))
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [activeId, projDims, JSON.stringify(filters)])

  const handleCreate = async () => {
    if (!formName.trim() || !formTag || !formA || !formB) return
    setError('')
    try {
      const d = await api.createDirection({
        name: formName.trim(),
        tag: formTag,
        value_a: formA,
        value_b: formB,
      })
      setFormName('')
      await refresh()
      setActiveId(d.id)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Error')
    }
  }

  const handleCreateRandom = async () => {
    if (!randomName.trim()) return
    setError('')
    try {
      const d = await api.createRandomDirection({ name: randomName.trim() })
      setRandomName('')
      await refresh()
      setActiveId(d.id)
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Error')
    }
  }

  const handleDelete = async (id: number) => {
    await api.deleteDirection(id)
    if (activeId === id) setActiveId(null)
    await refresh()
  }

  const availableTagsWithValues = tagColumns.filter(c => (tagValues[c]?.length ?? 0) >= 2)

  // Build histogram from projection.points, stacked by active direction's label tag
  const histogram = useMemo(() => {
    if (!projection || projection.points.length === 0) return null
    const scores = projection.points.map(p => p.score)
    const min = Math.min(...scores)
    const max = Math.max(...scores)
    const pad = (max - min) * 0.02 || 0.01
    const lo = min - pad
    const hi = max + pad
    const step = (hi - lo) / HIST_BINS
    const colorTag = projection.direction.tag
    const values = colorTag
      ? Array.from(new Set(projection.points.map(p => p.tags[colorTag] ?? '(none)')))
      : ['all']
    const bins = Array.from({ length: HIST_BINS }, (_, i) => {
      const center = lo + step * (i + 0.5)
      const entry: Record<string, number | string> = { bin: center.toFixed(3), center }
      for (const v of values) entry[v] = 0
      return entry
    })
    for (const p of projection.points) {
      const key = colorTag ? (p.tags[colorTag] ?? '(none)') : 'all'
      const idx = Math.min(HIST_BINS - 1, Math.max(0, Math.floor((p.score - lo) / step)))
      bins[idx][key] = (bins[idx][key] as number) + 1
    }
    return { bins, values }
  }, [projection])

  // AUC plot: downsample to ~200 points (3072/16)
  const aucCurve = useMemo(() => {
    if (!aucResult) return []
    const stride = 16
    return aucResult.auc
      .map((a, i) => ({ dim: i, auc: a, signal: Math.abs(a - 0.5) }))
      .filter((_, i) => i % stride === 0)
  }, [aucResult])

  return (
    <div className="panel">
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16, flexWrap: 'wrap' }}>
        <span className="panel-title" style={{ marginBottom: 0 }}>Directions</span>
        <span style={{ fontSize: 11, color: '#374151' }}>
          mean(A) − mean(B) unit vectors · project any embedding onto a concept axis
        </span>
      </div>

      {error && <div className="error-box" style={{ marginBottom: 12 }}>{error}</div>}

      {/* Create section */}
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 20 }}>
        <div style={{ background: '#0f1120', padding: 12, borderRadius: 8 }}>
          <div className="section-label">New mean-diff direction</div>
          <div className="flex-row" style={{ flexWrap: 'wrap', gap: 6, marginTop: 6 }}>
            <input
              placeholder="name, e.g. gender"
              value={formName}
              onChange={e => setFormName(e.target.value)}
              style={{ flex: '1 1 140px', padding: '6px 8px', fontSize: 12 }}
            />
            <select
              value={formTag}
              onChange={e => {
                const t = e.target.value
                setFormTag(t)
                setFormA(tagValues[t]?.[0] ?? '')
                setFormB(tagValues[t]?.[1] ?? '')
              }}
              style={{ padding: '6px 8px', fontSize: 12 }}
            >
              <option value="">tag…</option>
              {availableTagsWithValues.map(t => <option key={t} value={t}>{t}</option>)}
            </select>
            <select
              value={formA}
              onChange={e => setFormA(e.target.value)}
              style={{ padding: '6px 8px', fontSize: 12 }}
            >
              <option value="">A (positive)…</option>
              {(tagValues[formTag] ?? []).map(v => <option key={v} value={v}>{v}</option>)}
            </select>
            <span style={{ fontSize: 12, color: '#64748b' }}>−</span>
            <select
              value={formB}
              onChange={e => setFormB(e.target.value)}
              style={{ padding: '6px 8px', fontSize: 12 }}
            >
              <option value="">B (negative)…</option>
              {(tagValues[formTag] ?? []).map(v => <option key={v} value={v}>{v}</option>)}
            </select>
            <button className="btn btn-primary" onClick={handleCreate} style={{ padding: '6px 12px', fontSize: 12 }}>
              Create
            </button>
          </div>
        </div>

        <div style={{ background: '#0f1120', padding: 12, borderRadius: 8 }}>
          <div className="section-label">Random direction (control)</div>
          <div className="flex-row" style={{ gap: 6, marginTop: 6 }}>
            <input
              placeholder="name, e.g. random-1"
              value={randomName}
              onChange={e => setRandomName(e.target.value)}
              style={{ flex: 1, padding: '6px 8px', fontSize: 12 }}
            />
            <button className="btn btn-ghost" onClick={handleCreateRandom} style={{ padding: '6px 12px', fontSize: 12 }}>
              Create random
            </button>
          </div>
          <div style={{ fontSize: 11, color: '#64748b', marginTop: 6 }}>
            Sanity check: a random unit vector should give overlapping, uninformative projections.
          </div>
        </div>
      </div>

      {/* Directions list */}
      <div style={{ marginBottom: 20 }}>
        <div className="section-label">Saved directions ({directions.length})</div>
        {directions.length === 0 ? (
          <div className="empty" style={{ padding: 12 }}>No directions yet. Create one above.</div>
        ) : (
          <table className="diff-table" style={{ marginTop: 6 }}>
            <thead>
              <tr>
                <th></th>
                <th>Name</th>
                <th>Kind</th>
                <th>Tag · A − B</th>
                <th>n_A</th>
                <th>n_B</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {directions.map(d => (
                <tr
                  key={d.id}
                  style={{
                    background: d.id === activeId ? '#131a2e' : 'transparent',
                    cursor: 'pointer',
                  }}
                  onClick={() => setActiveId(d.id)}
                >
                  <td style={{ color: '#818cf8' }}>{d.id === activeId ? '●' : '○'}</td>
                  <td style={{ color: '#e5e7eb', fontWeight: 600 }}>{d.name}</td>
                  <td style={{ color: d.kind === 'random' ? '#94a3b8' : '#22c55e' }}>{d.kind}</td>
                  <td style={{ color: '#cbd5e1' }}>
                    {d.kind === 'mean_diff'
                      ? `${d.tag} · ${d.value_a} − ${d.value_b}`
                      : '—'}
                  </td>
                  <td style={{ color: '#64748b' }}>{d.n_a ?? '—'}</td>
                  <td style={{ color: '#64748b' }}>{d.n_b ?? '—'}</td>
                  <td>
                    <button
                      className="btn btn-ghost"
                      onClick={e => { e.stopPropagation(); handleDelete(d.id) }}
                      style={{ padding: '2px 8px', fontSize: 11 }}
                    >✕</button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>

      {/* Cosine matrix */}
      {directions.length >= 2 && (
        <div style={{ marginBottom: 24 }}>
          <div className="section-label">Pairwise cosine between directions</div>
          <div style={{ overflowX: 'auto' }}>
            <table className="diff-table" style={{ marginTop: 6 }}>
              <thead>
                <tr>
                  <th></th>
                  {directions.map(d => <th key={d.id} style={{ fontSize: 10 }}>{d.name}</th>)}
                </tr>
              </thead>
              <tbody>
                {directions.map((d, i) => (
                  <tr key={d.id}>
                    <td style={{ fontSize: 10, color: '#94a3b8' }}>{d.name}</td>
                    {directions.map((_, j) => {
                      const v = cosines[i]?.[j] ?? 0
                      const abs = Math.abs(v)
                      const bg = i === j
                        ? '#1e293b'
                        : `rgba(99,102,241,${0.08 + abs * 0.6})`
                      return (
                        <td key={j} style={{ background: bg, textAlign: 'center', fontVariantNumeric: 'tabular-nums' }}>
                          {v.toFixed(3)}
                        </td>
                      )
                    })}
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <div style={{ fontSize: 11, color: '#64748b', marginTop: 6 }}>
            Near-zero = orthogonal concepts. Large |cos| = entangled (surgery on one will also move the other).
          </div>
        </div>
      )}

      {/* Active direction details */}
      {active && (
        <div style={{ borderTop: '1px solid #1e2235', paddingTop: 16 }}>
          <div style={{ display: 'flex', alignItems: 'baseline', gap: 12, marginBottom: 8 }}>
            <span style={{ fontWeight: 600, color: '#e5e7eb' }}>{active.name}</span>
            {active.kind === 'mean_diff' && (
              <span style={{ fontSize: 12, color: '#64748b' }}>
                projection score = e · v̂ (positive → {active.value_a}, negative → {active.value_b})
              </span>
            )}
          </div>

          {/* Dims selector for projection */}
          <div className="flex-row" style={{ marginBottom: 12 }}>
            <span className="section-label" style={{ margin: 0 }}>Project at</span>
            {DIM_OPTIONS.map(d => (
              <button
                key={d}
                className={`chip${projDims === d ? ' active' : ''}`}
                onClick={() => setProjDims(d)}
              >{d}d</button>
            ))}
            {loading && <span style={{ fontSize: 11, color: '#64748b' }}>loading…</span>}
          </div>

          {/* Projection histogram */}
          {histogram && (
            <div style={{ marginBottom: 24 }}>
              <div className="section-label">
                Projection histogram — {projection?.points.length} points at {projection?.dims}d
              </div>
              <ResponsiveContainer width="100%" height={260}>
                <BarChart data={histogram.bins} margin={{ top: 5, right: 20, left: 0, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e2235" />
                  <XAxis
                    dataKey="bin"
                    tick={{ fill: '#374151', fontSize: 10 }}
                    label={{ value: 'e · v̂', position: 'insideBottom', offset: -8, fill: '#64748b', fontSize: 12 }}
                  />
                  <YAxis tick={{ fill: '#374151', fontSize: 11 }} />
                  <Tooltip
                    contentStyle={{ background: '#0f1120', border: '1px solid #1e2235', borderRadius: 8, fontSize: 12 }}
                  />
                  <Legend wrapperStyle={{ fontSize: 11 }} />
                  <ReferenceLine x={histogram.bins[Math.floor(HIST_BINS / 2)].bin} stroke="#64748b" strokeDasharray="2 2" />
                  {histogram.values.map((v, i) => (
                    <Bar key={v} dataKey={v} stackId="s" fill={colorFor(v, i)} />
                  ))}
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Per-dim AUC scan */}
          {aucResult && (
            <div style={{ marginBottom: 24 }}>
              <div className="section-label">
                Per-dim AUC scan — best single dim signal |auc−0.5| = {aucResult.max_abs_signal.toFixed(3)}
              </div>
              <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6 }}>
                No single of 3072 dims cleanly separates — this is what motivates directions.
              </div>
              <ResponsiveContainer width="100%" height={180}>
                <LineChart data={aucCurve} margin={{ top: 5, right: 20, left: 0, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e2235" />
                  <XAxis
                    dataKey="dim"
                    tick={{ fill: '#374151', fontSize: 10 }}
                    label={{ value: 'Dimension index', position: 'insideBottom', offset: -8, fill: '#64748b', fontSize: 12 }}
                  />
                  <YAxis
                    domain={[0, 1]}
                    tick={{ fill: '#374151', fontSize: 11 }}
                    label={{ value: 'AUC', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }}
                  />
                  <Tooltip
                    contentStyle={{ background: '#0f1120', border: '1px solid #1e2235', borderRadius: 8, fontSize: 12 }}
                    formatter={(v: number) => v.toFixed(3)}
                  />
                  <ReferenceLine y={0.5} stroke="#64748b" strokeDasharray="2 2" />
                  <Line type="monotone" dataKey="auc" stroke="#60a5fa" dot={false} strokeWidth={1} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
              <div
                style={{
                  marginTop: 14,
                  padding: '10px 12px',
                  background: '#0f1120',
                  borderLeft: '3px solid #6366f1',
                  borderRadius: 4,
                  fontSize: 12,
                  color: '#cbd5e1',
                  lineHeight: 1.55,
                }}
              >
                <div style={{ color: '#e5e7eb', fontWeight: 600, marginBottom: 4 }}>
                  Distributed representation
                </div>
                No single dim cleanly classifies the concept — the AUC curve
                hovers near 0.5, and the best single dim peaks well below 1.
                But the projection histogram above (the dot product with a
                learned direction) separates the classes near-perfectly. The
                concept isn&apos;t stored in a dimension; it&apos;s stored on a
                direction that recruits thousands of dimensions, each
                contributing a small, coherent piece. This is the central trick
                of neural representations — and arguably the core insight of
                modern ML: meaning lives on directions, not coordinates.
              </div>
            </div>
          )}

          {/* Matryoshka sweep */}
          {matryoshka && (
            <div style={{ marginBottom: 12 }}>
              <div className="section-label">
                Direction holds at lower Matryoshka dims?
              </div>
              <div style={{ fontSize: 11, color: '#64748b', marginBottom: 6 }}>
                AUC of e·v̂ when both the direction and embeddings are truncated to the first K dims and re-normalised.
              </div>
              <ResponsiveContainer width="100%" height={200}>
                <LineChart data={matryoshka.points} margin={{ top: 5, right: 20, left: 0, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#1e2235" />
                  <XAxis
                    dataKey="dim"
                    scale="log"
                    domain={['auto', 'auto']}
                    type="number"
                    tick={{ fill: '#374151', fontSize: 10 }}
                    label={{ value: 'Truncation dim (log)', position: 'insideBottom', offset: -8, fill: '#64748b', fontSize: 12 }}
                  />
                  <YAxis
                    domain={[0.5, 1]}
                    tick={{ fill: '#374151', fontSize: 11 }}
                    label={{ value: 'AUC', angle: -90, position: 'insideLeft', fill: '#64748b', fontSize: 11 }}
                  />
                  <Tooltip
                    contentStyle={{ background: '#0f1120', border: '1px solid #1e2235', borderRadius: 8, fontSize: 12 }}
                    formatter={(v: number) => v.toFixed(3)}
                  />
                  <Line type="monotone" dataKey="auc" stroke="#22c55e" strokeWidth={2} dot={{ r: 3 }} isAnimationActive={false} />
                </LineChart>
              </ResponsiveContainer>
            </div>
          )}
        </div>
      )}
    </div>
  )
}
