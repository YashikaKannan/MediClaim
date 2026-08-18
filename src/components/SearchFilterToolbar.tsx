import { Search, X, ArrowUpDown } from 'lucide-react'

export type FilterOption = {
  value: string
  label: string
}

export type FilterConfig = {
  id: string
  label: string
  value: string
  options: FilterOption[]
  onChange: (val: string) => void
}

export type SortOption = {
  value: string
  label: string
}

type SearchFilterToolbarProps = {
  searchPlaceholder?: string
  searchValue: string
  onSearchChange: (val: string) => void
  filters?: FilterConfig[]
  sortOptions?: SortOption[]
  sortValue?: string
  onSortChange?: (val: string) => void
}

export function SearchFilterToolbar({
  searchPlaceholder = 'Search...',
  searchValue,
  onSearchChange,
  filters = [],
  sortOptions,
  sortValue,
  onSortChange,
}: SearchFilterToolbarProps) {
  return (
    <div className="filter-toolbar">
      <div className="filter-toolbar-search">
        <Search size={16} className="filter-search-icon" />
        <input
          type="text"
          value={searchValue}
          onChange={(e) => onSearchChange(e.target.value)}
          placeholder={searchPlaceholder}
          className="filter-search-input"
          aria-label={searchPlaceholder}
        />
        {searchValue && (
          <button
            type="button"
            className="filter-search-clear"
            onClick={() => onSearchChange('')}
            aria-label="Clear search input"
            title="Clear search"
          >
            <X size={14} />
          </button>
        )}
      </div>

      <div className="filter-toolbar-controls">
        {filters.map((filter) => (
          <div key={filter.id} className="filter-select-wrap">
            <label htmlFor={`filter-${filter.id}`} className="sr-only">
              {filter.label}
            </label>
            <select
              id={`filter-${filter.id}`}
              value={filter.value}
              onChange={(e) => filter.onChange(e.target.value)}
              className={`filter-select ${filter.value && filter.value !== 'all' ? 'active' : ''}`}
            >
              <option value="all">{filter.label}: All</option>
              {filter.options.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          </div>
        ))}

        {sortOptions && sortOptions.length > 0 && onSortChange && (
          <div className="filter-sort-wrap">
            <label htmlFor="filter-sort-select" className="sr-only">
              Sort By
            </label>
            <div className="sort-select-inner">
              <ArrowUpDown size={14} className="sort-icon" />
              <select
                id="filter-sort-select"
                value={sortValue}
                onChange={(e) => onSortChange(e.target.value)}
                className="filter-select sort-select"
              >
                {sortOptions.map((opt) => (
                  <option key={opt.value} value={opt.value}>
                    Sort: {opt.label}
                  </option>
                ))}
              </select>
            </div>
          </div>
        )}
      </div>
    </div>
  )
}
