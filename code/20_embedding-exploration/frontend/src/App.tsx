import { useState, useEffect, useCallback } from 'react'
import { api } from './api'
import type { CacheItem, TagFilters, Tags } from './types'
import CachePanel from './components/CachePanel'
import FilterBar from './components/FilterBar'
import FingerprintChart from './components/FingerprintChart'
import DiffHeatmap from './components/DiffHeatmap'
import SimilarityMatrix from './components/SimilarityMatrix'
import MatryoshkaCurve from './components/MatryoshkaCurve'
import UmapScatter from './components/UmapScatter'

const TABS = ['Data', 'Fingerprint', 'Diff', 'Matrix', 'Matryoshka', 'UMAP'] as const
type Tab = typeof TABS[number]

export default function App() {
  const [tab, setTab] = useState<Tab>('Data')
  const [items, setItems] = useState<CacheItem[]>([])
  const [total, setTotal] = useState(0)
  const [selected, setSelected] = useState<Set<number>>(new Set())
  const [tagColumns, setTagColumns] = useState<string[]>([])
  const [tagValues, setTagValues] = useState<Record<string, string[]>>({})
  const [filters, setFilters] = useState<TagFilters>({})

  const refreshCache = useCallback(async () => {
    try {
      const r = await api.listCache(filters)
      setItems(r.items)
      setTotal(r.total)
    } catch {
      // backend not ready yet — ignore
    }
  }, [filters])

  const refreshTagValues = useCallback(async () => {
    try {
      const r = await api.tagValues()
      setTagColumns(r.tag_columns)
      setTagValues(r.tag_values)
    } catch {
      // ignore
    }
  }, [])

  useEffect(() => { refreshCache() }, [refreshCache])
  useEffect(() => { refreshTagValues() }, [refreshTagValues])

  const handleEmbed = async (text: string, tags?: Tags) => {
    await api.embed(text, tags)
    await Promise.all([refreshCache(), refreshTagValues()])
  }

  const handleDelete = async (id: number) => {
    await api.deleteCache(id)
    setSelected(prev => { const s = new Set(prev); s.delete(id); return s })
    await Promise.all([refreshCache(), refreshTagValues()])
  }

  const toggleSelect = (id: number) => {
    // Decide outside updater (Strict-Mode safe).
    const alreadySelected = selected.has(id)
    if (alreadySelected) {
      setSelected(prev => { const s = new Set(prev); s.delete(id); return s })
    } else {
      setSelected(prev => { const s = new Set(prev); s.add(id); return s })
    }
  }

  const clearSelection = () => setSelected(new Set())

  const selectAllVisible = () => setSelected(new Set(items.map(it => it.id)))

  const toggleFilter = (col: string, value: string) => {
    // Decide add-vs-remove outside the updater (Strict-Mode safe).
    const isActive = (filters[col] ?? []).includes(value)
    if (isActive) {
      setFilters(prev => {
        const cur = prev[col] ?? []
        const nextVals = cur.filter(v => v !== value)
        const result = { ...prev, [col]: nextVals }
        if (nextVals.length === 0) delete result[col]
        return result
      })
    } else {
      setFilters(prev => ({ ...prev, [col]: [...(prev[col] ?? []), value] }))
    }
  }

  const clearFilters = () => setFilters({})

  const selectedItems = items.filter(it => selected.has(it.id))
  const selectedIds = selectedItems.map(it => it.id)
  const hasFilters = Object.keys(filters).length > 0
  const hasTags = tagColumns.some(c => (tagValues[c]?.length ?? 0) > 0)

  return (
    <div className="app">
      <header>
        <h1>Embedding Explorer</h1>
        <nav>
          {TABS.map(t => (
            <button
              key={t}
              className={tab === t ? 'active' : ''}
              onClick={() => setTab(t)}
            >
              {t}
            </button>
          ))}
        </nav>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center' }}>
          {selected.size > 0 && (
            <>
              <span className="selected-count">{selected.size} selected</span>
              <button className="btn btn-ghost" onClick={clearSelection} style={{ padding: '3px 10px', fontSize: 12 }}>
                Clear
              </button>
            </>
          )}
        </div>
      </header>

      {hasTags && (
        <FilterBar
          tagColumns={tagColumns}
          tagValues={tagValues}
          filters={filters}
          onToggle={toggleFilter}
          onClear={clearFilters}
          total={total}
          visible={items.length}
        />
      )}

      <main>
        {tab === 'Data' && (
          <CachePanel
            items={items}
            total={total}
            hasFilters={hasFilters}
            selected={selected}
            tagColumns={tagColumns}
            onToggle={toggleSelect}
            onEmbed={handleEmbed}
            onDelete={handleDelete}
            onSelectAll={selectAllVisible}
            onClearSelection={clearSelection}
          />
        )}
        {tab === 'Fingerprint' && (
          <FingerprintChart selectedIds={selectedIds} />
        )}
        {tab === 'Diff' && (
          <DiffHeatmap selectedIds={selectedIds} />
        )}
        {tab === 'Matrix' && (
          <SimilarityMatrix selectedIds={selectedIds} />
        )}
        {tab === 'Matryoshka' && (
          <MatryoshkaCurve selectedIds={selectedIds} />
        )}
        {tab === 'UMAP' && (
          <UmapScatter
            selectedIds={selectedIds}
            filters={filters}
            tagColumns={tagColumns}
            tagValues={tagValues}
          />
        )}
      </main>
    </div>
  )
}
