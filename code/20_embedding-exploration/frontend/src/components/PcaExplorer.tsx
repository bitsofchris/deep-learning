import { useState, useMemo, useEffect } from 'react'
import { api } from '../api'
import type { PcaItem, PcaResult, TagFilters } from '../types'

const DIM_OPTIONS = [32, 128, 256, 512, 1024, 3072]

const TAG_COLORS: Record<string, string> = {
  male: '#60a5fa',
  female: '#f472b6',
  royal: '#facc15',
  common: '#94a3b8',
  short: '#34d399',
  long: '#a78bfa',
  openaugi: '#818cf8',
  tsai: '#f59e0b',
  trading: '#22c55e',
  health: '#f43f5e',
}

interface Props {
  selectedIds: number[]
  filters: TagFilters
  tagColumns: string[]
  tagValues: Record<string, string[]>
}

type SortDir = 'desc' | 'asc'

export default function PcaExplorer({ selectedIds, filters, tagColumns, tagValues }: Props) {
  const [dims, setDims] = useState<number>(3072)
  const [nComponents, setNComponents] = useState<number>(10)
  const [selectedOnly, setSelectedOnly] = useState(false)
  const [result, setResult] = useState<PcaResult | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const [activePc, setActivePc] = useState<number>(0)
  const [sortDir, setSortDir] = useState<SortDir>('desc')
  const [showCount, setShowCount] = useState<number>(20)
  const [colorBy, setColorBy] = useState<string>('')

  const handleCompute = async () => {
    setLoading(true)
    setError('')
    try {
      const opts: { ids?: number[]; filters?: TagFilters; dims: number; n_components: number } = {
        dims, n_components: nComponents,
      }
      if (selectedOnly && selectedIds.length > 0) opts.ids = selectedIds
      else opts.filters = filters
      const r = await api.pca(opts)
      setResult(r)
      setActivePc(p => Math.min(p, r.n_components - 1))
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Error')
    } finally {
      setLoading(false)
    }
  }

  // Auto-recompute after first compute when knobs change
  useEffect(() => {
    if (!result) return
    handleCompute()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [dims, nComponents, selectedOnly, selectedOnly ? selectedIds.join(',') : JSON.stringify(filters)])

  const sorted = useMemo(() => {
    if (!result) return [] as PcaItem[]
    const items = [...result.items]
    items.sort((a, b) => {
      const av = a.scores[activePc]
      const bv = b.scores[activePc]
      return sortDir === 'desc' ? bv - av : av - bv
    })
    return items
  }, [result, activePc, sortDir])

  const maxAbs = useMemo(() => {
    if (!result) return 1
    let m = 0
    for (const it of result.items) {
      const v = Math.abs(it.scores[activePc])
      if (v > m) m = v
    }
    return m || 1
  }, [result, activePc])

  const visible = showCount >= sorted.length ? sorted : sorted.slice(0, showCount)

  return (
    <div className="panel">
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 16, flexWrap: 'wrap' }}>
        <span className="panel-title" style={{ marginBottom: 0 }}>PCA Explorer</span>
        <span style={{ fontSize: 11, color: '#374151' }}>
          Find interpretable axes of variation · inspect extremes to name each PC
        </span>
      </div>

      {/* Controls */}
      <div className="flex-row" style={{ marginBottom: 18, flexWrap: 'wrap' }}>
        <button className="btn btn-primary" onClick={handleCompute} disabled={loading}>
          {loading ? 'Computing…' : result ? 'Recompute' : 'Compute PCA'}
        </button>

        <div className="flex-row">
          <span className="section-label" style={{ margin: 0 }}>Scope</span>
          <button
            className={`chip${!selectedOnly ? ' active' : ''}`}
            onClick={() => setSelectedOnly(false)}
          >All</button>
          <button
            className={`chip${selectedOnly ? ' active' : ''}`}
            onClick={() => setSelectedOnly(true)}
            disabled={selectedIds.length < 3}
            title={selectedIds.length < 3 ? 'Select ≥3 items' : `Use ${selectedIds.length} selected`}
          >Selected ({selectedIds.length})</button>
        </div>

        <div className="dim-toggle">
          <span className="section-label" style={{ margin: 0, marginRight: 6 }}>Dims</span>
          {DIM_OPTIONS.map(d => (
            <button
              key={d}
              className={`chip${dims === d ? ' active' : ''}`}
              onClick={() => setDims(d)}
            >{d}</button>
          ))}
        </div>

        <div className="flex-row">
          <span className="section-label" style={{ margin: 0 }}>Components</span>
          <input
            type="number"
            min={1}
            max={50}
            value={nComponents}
            onChange={e => setNComponents(Math.max(1, Math.min(50, Number(e.target.value) || 1)))}
            style={{
              width: 60, padding: '3px 6px', fontSize: 12,
              background: '#131622', color: '#e2e8f0',
              border: '1px solid #1e2235', borderRadius: 4,
            }}
          />
        </div>
      </div>

      {loading && <div className="loading"><div className="spinner" />Running PCA at {dims}d…</div>}
      {error && <div className="error-box">{error}</div>}

      {result && !loading && (
        <>
          {/* Scree: PC selector + variance bars */}
          <div style={{ marginBottom: 16 }}>
            <div className="section-label" style={{ marginBottom: 6 }}>
              Principal components · {result.count} points, {result.dims}d input
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
              {result.explained_variance_ratio.map((v, i) => {
                const pct = v * 100
                const maxV = result.explained_variance_ratio[0] || 1
                const barPct = (v / maxV) * 100
                const active = i === activePc
                return (
                  <button
                    key={i}
                    onClick={() => setActivePc(i)}
                    style={{
                      display: 'grid',
                      gridTemplateColumns: '50px 1fr 60px',
                      alignItems: 'center',
                      gap: 8,
                      padding: '4px 8px',
                      background: active ? '#1e2235' : 'transparent',
                      border: active ? '1px solid #6366f1' : '1px solid transparent',
                      borderRadius: 4,
                      cursor: 'pointer',
                      color: active ? '#e2e8f0' : '#94a3b8',
                      fontSize: 12,
                      textAlign: 'left',
                    }}
                  >
                    <span style={{ fontWeight: active ? 700 : 500 }}>PC{i + 1}</span>
                    <div style={{ height: 10, background: '#131622', borderRadius: 2, position: 'relative' }}>
                      <div style={{
                        width: `${barPct}%`,
                        height: '100%',
                        background: active ? '#6366f1' : '#374151',
                        borderRadius: 2,
                      }} />
                    </div>
                    <span style={{ textAlign: 'right', fontVariantNumeric: 'tabular-nums' }}>{pct.toFixed(2)}%</span>
                  </button>
                )
              })}
            </div>
          </div>

          {/* Sorted list controls */}
          <div className="flex-row" style={{ marginBottom: 10, flexWrap: 'wrap' }}>
            <span className="section-label" style={{ margin: 0 }}>Sort</span>
            <button
              className={`chip${sortDir === 'desc' ? ' active' : ''}`}
              onClick={() => setSortDir('desc')}
              title="Highest score first"
            >high → low</button>
            <button
              className={`chip${sortDir === 'asc' ? ' active' : ''}`}
              onClick={() => setSortDir('asc')}
              title="Lowest score first"
            >low → high</button>

            <span className="section-label" style={{ margin: 0 }}>Show</span>
            {[10, 20, 50, 100, result.count].map((n, i) => (
              <button
                key={i}
                className={`chip${showCount === n ? ' active' : ''}`}
                onClick={() => setShowCount(n)}
              >{n === result.count ? `all (${n})` : n}</button>
            ))}

            {tagColumns.length > 0 && (
              <>
                <span className="section-label" style={{ margin: 0 }}>Tag pills</span>
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
              </>
            )}
          </div>

          {/* Ranked list */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {visible.map(it => {
              const score = it.scores[activePc]
              const widthPct = (Math.abs(score) / maxAbs) * 50
              const positive = score >= 0
              const pill = colorBy ? it.tags[colorBy] : undefined
              return (
                <div key={it.id} style={{
                  display: 'grid',
                  gridTemplateColumns: '80px 120px 1fr auto',
                  alignItems: 'center',
                  gap: 10,
                  padding: '5px 8px',
                  borderBottom: '1px solid #131622',
                  fontSize: 12,
                }}>
                  <span style={{
                    fontVariantNumeric: 'tabular-nums',
                    color: positive ? '#4ade80' : '#f87171',
                    fontWeight: 600,
                  }}>
                    {score.toFixed(3)}
                  </span>
                  <div style={{ position: 'relative', height: 10, background: '#0f1120', borderRadius: 2 }}>
                    <div style={{ position: 'absolute', top: 0, bottom: 0, left: '50%', width: 1, background: '#1e2235' }} />
                    <div style={{
                      position: 'absolute',
                      top: 0, bottom: 0,
                      left: positive ? '50%' : `${50 - widthPct}%`,
                      width: `${widthPct}%`,
                      background: positive ? '#4ade80' : '#f87171',
                      opacity: 0.7,
                      borderRadius: 2,
                    }} />
                  </div>
                  <span style={{ color: '#e2e8f0', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                    {it.text}
                  </span>
                  {pill && (
                    <span style={{
                      background: TAG_COLORS[pill] ?? '#374151',
                      color: '#0f1120',
                      padding: '1px 8px',
                      borderRadius: 99,
                      fontSize: 10,
                      fontWeight: 600,
                    }}>{pill}</span>
                  )}
                </div>
              )
            })}
          </div>
          {showCount < sorted.length && (
            <div style={{ marginTop: 8, fontSize: 11, color: '#64748b', textAlign: 'center' }}>
              Showing {showCount} of {sorted.length}
            </div>
          )}
        </>
      )}
    </div>
  )
}
