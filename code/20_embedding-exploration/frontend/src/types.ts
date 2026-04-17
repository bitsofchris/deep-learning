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
