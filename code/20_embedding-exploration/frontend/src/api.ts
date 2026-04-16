import type {
  CacheListResult,
  VectorItem,
  UmapResult,
  SimilarityResult,
  MatryoshkaResult,
  ArithmeticResult,
  TagValuesResult,
  TagFilters,
  Tags,
} from './types'

const BASE = '/api'

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: res.statusText }))
    throw new Error(err.detail ?? res.statusText)
  }
  return res.json()
}

async function get<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`)
  if (!res.ok) throw new Error(res.statusText)
  return res.json()
}

async function del(path: string): Promise<void> {
  const res = await fetch(`${BASE}${path}`, { method: 'DELETE' })
  if (!res.ok) throw new Error(res.statusText)
}

export const api = {
  embed: (text: string, tags?: Tags): Promise<{ id: number; text: string; cached: boolean }> =>
    post('/embed', { text, tags: tags ?? null }),

  listCache: (filters?: TagFilters): Promise<CacheListResult> =>
    post('/cache/list', { filters: filters ?? null }),

  tagValues: (): Promise<TagValuesResult> =>
    get('/tag-values'),

  deleteCache: (id: number): Promise<void> =>
    del(`/cache/${id}`),

  getVectors: (ids: number[]): Promise<{ items: VectorItem[] }> =>
    post('/vectors', { ids }),

  umap: (opts: { ids?: number[]; filters?: TagFilters; dims?: number }): Promise<UmapResult> =>
    post('/umap', {
      ids: opts.ids ?? null,
      filters: opts.filters ?? null,
      dims: opts.dims ?? 3072,
    }),

  similarity: (ids: number[]): Promise<SimilarityResult> =>
    post('/similarity', { ids }),

  matryoshka: (id_a: number, id_b: number): Promise<MatryoshkaResult> =>
    post('/matryoshka', { id_a, id_b }),

  arithmetic: (id_a: number, id_b: number, id_c: number): Promise<ArithmeticResult> =>
    post('/arithmetic', { id_a, id_b, id_c }),
}
