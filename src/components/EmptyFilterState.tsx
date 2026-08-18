import { FilterX, RotateCcw } from 'lucide-react'

type EmptyFilterStateProps = {
  title?: string
  message?: string
  onClearFilters: () => void
}

export function EmptyFilterState({
  title = 'No matching records found',
  message = 'No records matched your search query and filter criteria. Try adjusting or clearing your filters.',
  onClearFilters,
}: EmptyFilterStateProps) {
  return (
    <div className="empty-filter-state">
      <div className="empty-filter-icon">
        <FilterX size={28} />
      </div>
      <h4>{title}</h4>
      <p>{message}</p>
      <button
        type="button"
        className="secondary-action empty-filter-btn"
        onClick={onClearFilters}
      >
        <RotateCcw size={14} />
        Clear Filters
      </button>
    </div>
  )
}
