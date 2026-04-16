import type { TagFilters } from '../types'

interface Props {
  tagColumns: string[]
  tagValues: Record<string, string[]>
  filters: TagFilters
  onToggle: (col: string, value: string) => void
  onClear: () => void
  total: number
  visible: number
}

export default function FilterBar({
  tagColumns, tagValues, filters, onToggle, onClear, total, visible,
}: Props) {
  const anyValues = tagColumns.some(c => (tagValues[c]?.length ?? 0) > 0)
  if (!anyValues) return null

  const hasFilters = Object.keys(filters).length > 0

  return (
    <div className="filter-bar">
      <div className="filter-rows">
        {tagColumns.map(col => {
          const values = tagValues[col] ?? []
          if (values.length === 0) return null
          const active = filters[col] ?? []
          return (
            <div className="filter-row" key={col}>
              <span className="filter-label">{col}</span>
              <div className="chip-group">
                {values.map(v => (
                  <button
                    key={v}
                    className={`chip${active.includes(v) ? ' active' : ''}`}
                    onClick={() => onToggle(col, v)}
                  >
                    {v}
                  </button>
                ))}
              </div>
            </div>
          )
        })}
      </div>
      <div className="filter-meta">
        <span style={{ fontSize: 12, color: '#64748b' }}>
          {hasFilters ? `${visible} of ${total}` : `${total} items`}
        </span>
        {hasFilters && (
          <button className="btn btn-ghost" onClick={onClear} style={{ padding: '3px 10px', fontSize: 12 }}>
            Clear filters
          </button>
        )}
      </div>
    </div>
  )
}
