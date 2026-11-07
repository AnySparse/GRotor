"""Paper inputs in publication order; tera designs are explicit opt-ins."""
PAPER_CASES = (
    ('B1', 'Ariane sample', 'ariane133_68'),
    ('B2', 'Ariane rank', 'ariane_68_rank'),
    ('B3', 'NVDLA sample', 'nvdla'),
    ('B4', 'NVDLA rank', 'nvdla_rank'),
    ('B5', 'BlackParrot sample', 'bsg_chip'),
    ('B6', 'BlackParrot rank', 'bsg_chip_rank'),
    ('B7', 'MemPool-Tile sample', 'mempool_tile'),
    ('B8', 'MemPool-Tile rank', 'mempool_tile_rank'),
    ('B9', 'MemPool-Group sample', 'mempool_group'),
    ('B10', 'MemPool-Group rank', 'mempool_group_rank'),
    ('B11', 'MemPool-Cluster sample', 'mempool_cluster'),
    ('B12', 'MemPool-Cluster rank', 'mempool_cluster_ranking'),
    ('B13', 'TeraPool-Cluster sample', 'tera_cluster'),
    ('B14', 'TeraPool-Cluster rank', 'tera_cluster_rank'),
)
DEFAULT_CASES = [case for _, _, case in PAPER_CASES[:12]]
LARGE_CASES = [case for _, _, case in PAPER_CASES[12:]]
BASELINE_CASES = [PAPER_CASES[i][2] for i in (0, 2, 4, 6, 8, 10)]
