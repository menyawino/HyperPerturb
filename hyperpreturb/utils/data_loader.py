import scanpy as sc
import pandas as pd

class GenomicsDataLoader:
    def __init__(self, data_dir="data/"):
        self.data_dir = data_dir
        
    def load_string(self, organism=9606, score_threshold=700):
        path = f"{self.data_dir}/raw/9606.protein.links.full.v12.0.txt"
        df = pd.read_csv(path, sep=' ')
        return df[df['combined_score'] > score_threshold]
    
    def load_hematopoietic(self):
        adata = sc.read_h5ad(f"{self.data_dir}/raw/hematopoietic.h5ad")
        sc.pp.normalize_total(adata, target_sum=1e4)
        sc.pp.log1p(adata)
        return adata