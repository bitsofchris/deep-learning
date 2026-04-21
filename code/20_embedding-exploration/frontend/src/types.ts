export type Tags = Record<string, string>

export interface CacheItem {
  id: number
  text: string
  created_at: string
  tags: Tags
}

export interface VectorItem {
  id: number
  text: string
  tags: Tags
  embedding: number[]
}

export interface UmapPoint {
  id: number
  text: string
  tags: Tags
  x: number
  y: number
}

export interface TagMetric {
  knn_purity: number
  knn_k: number
  silhouette: number
  chance_baseline: number
  n_classes: number
  n_points: number
}

export interface UmapResult {
  points: UmapPoint[]
  dims: number
  count: number
  metrics: Record<string, TagMetric>
}

export interface CacheListResult {
  items: CacheItem[]
  total: number
}

export interface TagValuesResult {
  tag_columns: string[]
  tag_values: Record<string, string[]>
}

export interface SimilarityResult {
  texts: string[]
  matrix: number[][]
}

export interface MatryoshkaResult {
  text_a: string
  text_b: string
  dims: number[]
  similarities: number[]
}

export interface ArithmeticNeighbor {
  id: number
  text: string
  similarity: number
}

export interface ArithmeticResult {
  text_a: string
  text_b: string
  text_c: string
  nearest: ArithmeticNeighbor[]
}

export type TagFilters = Record<string, string[]>

export interface PcaItem {
  id: number
  text: string
  tags: Tags
  scores: number[]
}

export interface PcaResult {
  dims: number
  n_components: number
  count: number
  explained_variance_ratio: number[]
  items: PcaItem[]
}

export interface Direction {
  id: number
  name: string
  tag: string | null
  value_a: string | null
  value_b: string | null
  kind: 'mean_diff' | 'random'
  n_a: number | null
  n_b: number | null
  created_at: string
}

export interface DirectionsListResult {
  directions: Direction[]
  cosines: number[][]
}

export interface ProjectionPoint {
  id: number
  text: string
  tags: Tags
  score: number
}

export interface ProjectionResult {
  direction: Direction
  dims: number
  points: ProjectionPoint[]
}

export interface AucScanTop {
  dim: number
  auc: number
  signal: number
}

export interface AucScanResult {
  tag: string
  value_a: string
  value_b: string
  n_a: number
  n_b: number
  auc: number[]
  top: AucScanTop[]
  max_abs_signal: number
}

export interface DirectionMatryoshkaPoint {
  dim: number
  auc: number
  mean_gap: number
}

export interface DirectionMatryoshkaResult {
  tag: string
  value_a: string
  value_b: string
  n_a: number
  n_b: number
  points: DirectionMatryoshkaPoint[]
}
