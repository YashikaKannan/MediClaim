import { X, RotateCcw } from 'lucide-react'

export type ActiveFilterItem = {
  id: string
  label: string
  value: string
  displayLabel: string
  onRemove: () => void
}

type ActiveFilterChipsProps = {
  chips: ActiveFilterItem[]
  onClearAll: () => void
}

export function ActiveFilterChips({ chips, onClearAll }: ActiveFilterChipsProps) {
  if (chips.length === 0) return null

  return (
    <div className="filter-chips-bar" aria-label="Active filters">
      <span className="filter-chips-title">Active Filters:</span>
      <div className="filter-chips-list">
        {chips.map((chip) => (
          <span key={chip.id} className="filter-chip">
            <span className="filter-chip-text">{chip.displayLabel}</span>
            <button
              type="button"
              className="filter-chip-remove"
              onClick={chip.onRemove}
              aria-label={`Remove filter ${chip.displayLabel}`}
              title={`Remove ${chip.displayLabel}`}
            >
              <X size={12} />
            </button>
          </span>
        ))}
        <button
          type="button"
          className="filter-clear-all-btn"
          onClick={onClearAll}
          title="Reset search and all filters"
        >
          <RotateCcw size={12} />
          Clear All
        </button>
      </div>
    </div>
  )
}
