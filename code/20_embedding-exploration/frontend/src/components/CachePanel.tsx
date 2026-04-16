import { useState } from 'react'
import type { CacheItem, Tags } from '../types'

interface Props {
  items: CacheItem[]
  total: number
  hasFilters: boolean
  selected: Set<number>
  tagColumns: string[]
  onToggle: (id: number) => void
  onEmbed: (text: string, tags?: Tags) => Promise<void>
  onDelete: (id: number) => Promise<void>
  onSelectAll: () => void
  onClearSelection: () => void
}

const TAG_COLORS: Record<string, string> = {
  male: '#60a5fa',
  female: '#f472b6',
  royal: '#facc15',
  common: '#94a3b8',
  short: '#34d399',
  long: '#a78bfa',
}

export default function CachePanel({
  items, total, hasFilters, selected, tagColumns,
  onToggle, onEmbed, onDelete, onSelectAll, onClearSelection,
}: Props) {
  const [input, setInput] = useState('')
  const [pendingTags, setPendingTags] = useState<Tags>({})
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [lastResult, setLastResult] = useState<string | null>(null)

  const isAlreadyCached = items.some(it => it.text.trim() === input.trim())

  const handleEmbed = async () => {
    const text = input.trim()
    if (!text) return
    setLoading(true)
    setError('')
    setLastResult(null)
    try {
      await onEmbed(text, Object.keys(pendingTags).length > 0 ? pendingTags : undefined)
      setLastResult(isAlreadyCached ? 'Already in cache.' : 'Embedded and cached.')
      setInput('')
      setPendingTags({})
    } catch (e: unknown) {
      setError(e instanceof Error ? e.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) handleEmbed()
  }

  const setTag = (col: string, val: string) => {
    // Decide add-vs-remove *outside* the updater so Strict Mode
    // double-invocation is idempotent.
    if (pendingTags[col] === val) {
      setPendingTags(prev => {
        const next = { ...prev }
        delete next[col]
        return next
      })
    } else {
      setPendingTags(prev => ({ ...prev, [col]: val }))
    }
  }

  return (
    <div className="flex-col">
      {/* Input area */}
      <div className="panel">
        <div className="panel-title">Add text</div>

        <textarea
          value={input}
          onChange={e => { setInput(e.target.value); setLastResult(null) }}
          onKeyDown={handleKeyDown}
          placeholder="Type or paste a phrase…"
          rows={3}
          style={{ width: '100%', resize: 'vertical', marginBottom: 10 }}
        />

        {/* Tag pickers (only if dataset has tags) */}
        {tagColumns.length > 0 && (
          <div className="pending-tags">
            {tagColumns.map(col => (
              <div className="pending-tag-row" key={col}>
                <span className="pending-tag-label">{col}</span>
                {(['male', 'female'].includes(col) || col === 'gender'
                  ? ['male', 'female']
                  : col === 'class'
                    ? ['royal', 'common']
                    : col === 'length'
                      ? ['short', 'long']
                      : []
                ).map(v => (
                  <button
                    key={v}
                    className={`chip${pendingTags[col] === v ? ' active' : ''}`}
                    onClick={() => setTag(col, v)}
                  >
                    {v}
                  </button>
                ))}
              </div>
            ))}
          </div>
        )}

        <div className="flex-row">
          <button
            className="btn btn-primary"
            onClick={handleEmbed}
            disabled={loading || !input.trim()}
          >
            {loading ? 'Embedding…' : isAlreadyCached ? 'Update tags' : 'Get Embedding'}
          </button>

          {lastResult && (
            <span style={{ fontSize: 12, color: '#4ade80' }}>{lastResult}</span>
          )}
          {error && (
            <span style={{ fontSize: 12, color: '#f87171' }}>{error}</span>
          )}
        </div>

        <p style={{ fontSize: 11, color: '#374151', marginTop: 8 }}>
          ⌘↵ to embed · text-embedding-3-large (3072 dims)
        </p>
      </div>

      {/* Items list */}
      <div className="panel">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 16, gap: 12, flexWrap: 'wrap' }}>
          <span className="panel-title" style={{ marginBottom: 0 }}>
            Items — {items.length}{hasFilters && total > items.length ? ` of ${total}` : ''}
          </span>
          {items.length > 0 && (
            <div className="flex-row">
              <button
                className="btn btn-ghost"
                style={{ padding: '3px 10px', fontSize: 12 }}
                onClick={onSelectAll}
                disabled={items.every(it => selected.has(it.id))}
              >
                Select all{hasFilters ? ' matching filter' : ''}
              </button>
              <button
                className="btn btn-ghost"
                style={{ padding: '3px 10px', fontSize: 12 }}
                onClick={onClearSelection}
                disabled={selected.size === 0}
              >
                Clear
              </button>
            </div>
          )}
        </div>

        {items.length === 0 ? (
          <div className="empty">
            {hasFilters ? 'No items match current filters.' : 'No embeddings yet. Add some text above.'}
          </div>
        ) : (
          <div className="cache-list">
            {items.map(item => (
              <div
                key={item.id}
                className={`cache-item${selected.has(item.id) ? ' selected' : ''}`}
              >
                <input
                  type="checkbox"
                  checked={selected.has(item.id)}
                  onChange={() => onToggle(item.id)}
                />
                <span className="item-text" title={item.text}>{item.text}</span>
                <span className="item-tags">
                  {Object.entries(item.tags).map(([k, v]) => (
                    <span
                      key={k}
                      className="tag-pill"
                      title={`${k}=${v}`}
                      style={{ color: TAG_COLORS[v] ?? '#818cf8' }}
                    >
                      {v}
                    </span>
                  ))}
                </span>
                <button
                  className="btn btn-danger"
                  onClick={() => onDelete(item.id)}
                >
                  ✕
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  )
}
