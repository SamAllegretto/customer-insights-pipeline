"""
Customer feedback clustering pipeline using UMAP + HDBSCAN with recursive clustering.
Based on the iterative dimensionality reduction and density-based clustering approach.
"""

from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime, timedelta, timezone
import logging
import argparse

import pandas as pd
import numpy as np
import umap
import hdbscan
from sklearn.preprocessing import StandardScaler

from src.config.settings import Settings
from src.data_access.sql_client import SQLClient
from src.data_access.cosmos_client import CosmosClient
from src.models.schemas import FeedbackRecord

logger = logging.getLogger(__name__)


class RecursiveClusteringPipeline:
    """
    Pipeline for clustering feedback using UMAP + HDBSCAN with recursive zooming.
    Implements the approach from the article for customer segmentation.
    """

    def __init__(
        self, 
        config: Settings,
        umap_params: Optional[Dict[str, Any]] = None,
        hdbscan_params: Optional[Dict[str, Any]] = None,
        recursive_depth: int = 1,
        min_cluster_size_pct: float = 0.01
    ):
        self.config = config
        self.sql_client = SQLClient(config)
        self.cosmos_client = CosmosClient(config)
        
        # Default UMAP parameters
        self.umap_params = umap_params or {
            'n_neighbors': 15,
            'n_components': 2,
            'metric': 'cosine',
            'random_state': 42
        }
        
        # Default HDBSCAN parameters
        self.hdbscan_params = hdbscan_params or {
            'min_cluster_size': 500,
            'min_samples': 10,
            'metric': 'euclidean'
        }
        
        self.recursive_depth = recursive_depth
        self.min_cluster_size_pct = min_cluster_size_pct
        self.cluster_hierarchy = {}

    def _apply_umap(self, embeddings: np.ndarray) -> np.ndarray:
        """Apply UMAP dimensionality reduction."""
        logger.info(f"Applying UMAP with params: {self.umap_params}")
        reducer = umap.UMAP(**self.umap_params)
        return reducer.fit_transform(embeddings)

    def _apply_hdbscan(self, reduced_data: np.ndarray, min_cluster_size: Optional[int] = None) -> np.ndarray:
        """Apply HDBSCAN clustering."""
        params = self.hdbscan_params.copy()
        if min_cluster_size:
            params['min_cluster_size'] = min_cluster_size
            
        logger.info(f"Applying HDBSCAN with params: {params}")
        clusterer = hdbscan.HDBSCAN(**params)
        return clusterer.fit_predict(reduced_data)

    def _recursive_cluster(
        self,
        embeddings: np.ndarray,
        indices: np.ndarray,
        parent_label: str = "root",
        current_depth: int = 0
    ) -> Dict[int, str]:
        """
        Recursively cluster data by zooming into each cluster.
        
        Args:
            embeddings: Original high-dimensional embeddings
            indices: Indices of data points to cluster
            parent_label: Label of parent cluster
            current_depth: Current recursion depth
            
        Returns:
            Dictionary mapping original indices to hierarchical cluster labels
        """
        if current_depth >= self.recursive_depth or len(indices) < self.hdbscan_params['min_cluster_size']:
            return {idx: parent_label for idx in indices}
        
        logger.info(f"Clustering {len(indices)} points at depth {current_depth} (parent: {parent_label})")
        
        # Get embeddings for this subset
        subset_embeddings = embeddings[indices]
        
        # Apply UMAP
        reduced_data = self._apply_umap(subset_embeddings)
        
        # Apply HDBSCAN with dynamic min_cluster_size based on subset size
        min_size = max(
            self.hdbscan_params['min_cluster_size'],
            int(len(indices) * self.min_cluster_size_pct)
        )
        labels = self._apply_hdbscan(reduced_data, min_cluster_size=min_size)
        
        # Store cluster info
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info(f"Found {n_clusters} clusters at depth {current_depth}")
        
        # Build hierarchical labels
        cluster_mapping = {}
        unique_labels = set(labels)
        
        for cluster_id in unique_labels:
            cluster_mask = labels == cluster_id
            cluster_indices = indices[cluster_mask]
            
            if cluster_id == -1:
                # Noise points
                hierarchical_label = f"{parent_label}.noise"
                cluster_mapping.update({idx: hierarchical_label for idx in cluster_indices})
            else:
                hierarchical_label = f"{parent_label}.{cluster_id}"
                
                # Recurse if we haven't reached max depth and cluster is large enough
                if current_depth < self.recursive_depth - 1 and len(cluster_indices) >= min_size * 2:
                    sub_clusters = self._recursive_cluster(
                        embeddings,
                        cluster_indices,
                        hierarchical_label,
                        current_depth + 1
                    )
                    cluster_mapping.update(sub_clusters)
                else:
                    cluster_mapping.update({idx: hierarchical_label for idx in cluster_indices})
        
        # Store in hierarchy
        self.cluster_hierarchy[parent_label] = {
            'n_points': len(indices),
            'n_clusters': n_clusters,
            'depth': current_depth
        }
        
        return cluster_mapping

    def run(
        self,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        days_back: Optional[int] = None,
        limit: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Execute the UMAP + HDBSCAN clustering pipeline with recursive zooming.

        Args:
            start_date: Filter feedback on or after this date (inclusive).
            end_date: Filter feedback on or before this date (inclusive).
            days_back: Look-back window in days from now.
            limit: Max records to process.

        Returns:
            Dictionary of clustering stats/results.
        """
        if days_back is not None:
            end_date = datetime.now(timezone.utc)
            start_date = end_date - timedelta(days=days_back)
            logger.info(f"Using look-back window: last {days_back} days")

        logger.info(f"Clustering with UMAP + HDBSCAN (recursive depth: {self.recursive_depth})")

        # Fetch feedback data
        self.sql_client.connect()
        self.cosmos_client.connect()
        feedback_records = self.cosmos_client.get_all_embeddings(
            start_date=start_date,
            end_date=end_date,
            limit=limit,
        )

        logger.info(f"Fetched {len(feedback_records)} feedback records for clustering")

        if not feedback_records:
            logger.info("No records to cluster.")
            return {
                "total_records": 0,
                "clusters": {},
                "hierarchy": {},
                "start_date": start_date,
                "end_date": end_date
            }

        # Convert to DataFrame
        df = pd.DataFrame([f.__dict__ for f in feedback_records])

        # Extract embeddings
        if 'embedding' in df.columns:
            embeddings = np.array(df['embedding'].tolist())
        elif 'vector' in df.columns:
            embeddings = np.array(df['vector'].tolist())
        else:
            raise ValueError("No 'embedding' or 'vector' column found in feedback data")

        logger.info(f"Embeddings shape: {embeddings.shape}")

        # Normalize embeddings
        scaler = StandardScaler()
        embeddings = scaler.fit_transform(embeddings)

        # Run recursive clustering
        indices = np.arange(len(embeddings))
        cluster_mapping = self._recursive_cluster(embeddings, indices)

        # Add cluster labels to dataframe
        df['cluster_label'] = df.index.map(cluster_mapping)
        df['cluster_depth'] = df['cluster_label'].apply(lambda x: x.count('.'))

        logger.info(f"Clustering complete: {df['cluster_label'].nunique()} unique clusters found")

        # Prepare result summary
        cluster_stats = df.groupby('cluster_label').agg({
            'feedback_id': ['count', list],
            'cluster_depth': 'first'
        }).to_dict('index')

        result = {
            "total_records": len(df),
            "n_clusters": df['cluster_label'].nunique(),
            "clusters": {
                label: {
                    'count': stats[('feedback_id', 'count')],
                    'feedback_ids': stats[('feedback_id', 'list')],
                    'depth': stats[('cluster_depth', 'first')]
                }
                for label, stats in cluster_stats.items()
            },
            "hierarchy": self.cluster_hierarchy,
            "start_date": start_date.isoformat() if start_date else None,
            "end_date": end_date.isoformat() if end_date else None,
            "algorithm": "UMAP + HDBSCAN (recursive)",
            "recursive_depth": self.recursive_depth
        }

        # Optionally persist results
        # self._save_results(df)

        return result

    def _save_results(self, df: pd.DataFrame):
        """Save clustering results back to database."""
        # Example: update SQL or Cosmos with cluster labels
        cluster_data = df[['feedback_id', 'cluster_label', 'cluster_depth']].to_dict('records')
        # self.sql_client.update_feedback_clusters(cluster_data)
        logger.info(f"Saved clustering results for {len(cluster_data)} records")


def main():
    parser = argparse.ArgumentParser(description="Run UMAP + HDBSCAN recursive clustering pipeline.")
    parser.add_argument("--lookback", type=int, help="Number of days to look back.")
    parser.add_argument("--start-date", type=str, help="Start date (YYYY-MM-DD).")
    parser.add_argument("--end-date", type=str, help="End date (YYYY-MM-DD).")
    parser.add_argument("--limit", type=int, default=None, help="Max records to cluster.")
    parser.add_argument("--recursive-depth", type=int, default=1, help="How many levels to recurse (1 = no recursion).")
    parser.add_argument("--min-cluster-size", type=int, default=50, help="Minimum cluster size for HDBSCAN.")
    parser.add_argument("--min-cluster-pct", type=float, default=0.01, help="Min cluster size as percentage of data.")
    parser.add_argument("--n-neighbors", type=int, default=15, help="UMAP n_neighbors parameter.")
    parser.add_argument("--n-components", type=int, default=2, help="UMAP n_components (dimensions).")

    args = parser.parse_args()

    config = Settings()

    # Parse dates
    start_date = datetime.fromisoformat(args.start_date) if args.start_date else None
    end_date = datetime.fromisoformat(args.end_date) if args.end_date else None

    # Configure parameters
    umap_params = {
        'n_neighbors': args.n_neighbors,
        'n_components': args.n_components,
        'metric': 'cosine',
        'random_state': 42
    }

    hdbscan_params = {
        'min_cluster_size': args.min_cluster_size,
        'min_samples': 10,
        'metric': 'euclidean'
    }

    pipeline = RecursiveClusteringPipeline(
        config,
        umap_params=umap_params,
        hdbscan_params=hdbscan_params,
        recursive_depth=args.recursive_depth,
        min_cluster_size_pct=args.min_cluster_pct
    )

    result = pipeline.run(
        start_date=start_date,
        end_date=end_date,
        days_back=args.lookback,
        limit=args.limit,
    )
    
    print(f"\nClustering Results:")
    print(f"Total records: {result['total_records']}")
    print(f"Clusters found: {result['n_clusters']}")
    print(f"Recursive depth: {result['recursive_depth']}")
    print(f"\nCluster breakdown:")
    for label, info in sorted(result['clusters'].items()):
        print(f"  {label}: {info['count']} reviews (depth {info['depth']})")


if __name__ == "__main__":
    main()