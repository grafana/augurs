# Time Series Clustering

Time series clustering is a technique used to group similar time series together. This can be useful for finding patterns in data, detecting anomalies, or reducing the dimensionality of large datasets.

`augurs` provides several clustering algorithms, including DBSCAN (Density-Based Spatial Clustering of Applications with Noise). DBSCAN is particularly well-suited for time series data as it:

- Doesn't require specifying the number of clusters upfront
- Can find arbitrarily shaped clusters
- Can identify noise points that don't belong to any cluster
- Works well with Dynamic Time Warping (DTW) distance measures

## Basic Example

Let's start with a simple example using DBSCAN clustering with DTW distance:

<!-- langtabs-start -->
```rust
use augurs::{
    clustering::{DbscanCluster, DbscanClusterer},
    dtw::Dtw,
};

fn main() {
    // Sample time series data
    let series: &[&[f64]] = &[
        &[0.0, 1.0, 2.0, 3.0, 4.0],
        &[0.1, 1.1, 2.1, 3.1, 4.1],
        &[5.0, 6.0, 7.0, 8.0, 9.0],
        &[5.1, 6.1, 7.1, 8.1, 9.1],
        &[10.0, 11.0, 12.0, 13.0, 14.0],
    ];

    // Compute distance matrix using DTW
    let distance_matrix = Dtw::euclidean()
        .with_window(2)
        .distance_matrix(series);

    // Perform clustering
    let clusters = DbscanClusterer::new(0.5, 2)
        .fit(&distance_matrix);

    // Print cluster assignments
    println!("Cluster assignments: {:?}", clusters);
    
    // Clusters are labeled: Noise for outliers, Cluster(n) for membership
    for (idx, cluster) in clusters.iter().enumerate() {
        match cluster {
            DbscanCluster::Noise => println!("Series {} is noise", idx),
            DbscanCluster::Cluster(c) => println!("Series {} is in cluster {}", idx, c),
        }
    }
}
```

```javascript
import { Dtw } from '@bsull/augurs/dtw';
import { DbscanClusterer } from '@bsull/augurs/clustering';

// Sample time series data
const series = [
    [0.0, 1.0, 2.0, 3.0, 4.0],
    [0.1, 1.1, 2.1, 3.1, 4.1],
    [5.0, 6.0, 7.0, 8.0, 9.0],
    [5.1, 6.1, 7.1, 8.1, 9.1],
    [10.0, 11.0, 12.0, 13.0, 14.0],
];

// Compute distance matrix using DTW
const dtw = new Dtw('euclidean', { window: 2 });
const distanceMatrix = dtw.distanceMatrix(series);

// Perform clustering
// epsilon: 0.5, minClusterSize: 2
const clusterer = new DbscanClusterer({ epsilon: 0.5, minClusterSize: 2 });
const clusters = clusterer.fit(distanceMatrix);

// Print cluster assignments
console.log("Cluster assignments:", clusters);

// Examine individual assignments
clusters.forEach((cluster, idx) => {
    if (cluster === -1) {
        console.log(`Series ${idx} is noise`);
    } else {
        console.log(`Series ${idx} is in cluster ${cluster}`);
    }
});
```

```python
import augurs as aug
import numpy as np

# Sample time series data
series = [
    np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
    np.array([0.1, 1.1, 2.1, 3.1, 4.1]),
    np.array([5.0, 6.0, 7.0, 8.0, 9.0]),
    np.array([5.1, 6.1, 7.1, 8.1, 9.1]),
    np.array([10.0, 11.0, 12.0, 13.0, 14.0]),
]

# Compute distance matrix using DTW
dtw = aug.Dtw(window=2, distance_fn='euclidean')
distance_matrix = dtw.distance_matrix(series)

# Perform clustering
# epsilon: 0.5, min_cluster_size: 2
clusterer = aug.Dbscan(epsilon=0.5, min_cluster_size=2)
clusters = clusterer.fit(distance_matrix)

# Print cluster assignments
print("Cluster assignments:", clusters)

# Examine individual assignments
for idx, cluster in enumerate(clusters):
    if cluster == -1:
        print(f"Series {idx} is noise")
    else:
        print(f"Series {idx} is in cluster {cluster}")
```
<!-- langtabs-end -->

## Understanding Parameters

### DTW Parameters

- **`window`**: Size of the Sakoe-Chiba band for constraining DTW computation
  - Smaller values are faster but may miss good alignments
  - Larger values are more accurate but slower
  - Typically set to 10-20% of series length

- **`distance_fn`**: Distance function to use (`'euclidean'` or `'manhattan'`)
  - Euclidean is most common
  - Manhattan can be more robust to outliers

- **`max_distance`**: Early termination threshold (Rust)
  - If distance exceeds this, computation stops early
  - Useful for large datasets with known distance bounds

### DBSCAN Parameters

- **`epsilon`**: Maximum distance between two points for one to be considered in the neighborhood of the other
  - Too small: many points classified as noise
  - Too large: all points in one cluster
  - Start with mean/median distance from distance matrix

- **`min_cluster_size`**: Minimum number of points required to form a dense region
  - Usually 2-5 for small datasets
  - Higher values (10-20) for larger datasets
  - Depends on your definition of "cluster"

## Best Practices

### 1. Distance Measure Selection

- **Use DTW** for time series that might be shifted or warped
- **Consider computational cost** - DTW is O(n²) for each pair
- **Experiment with window sizes** to balance accuracy and performance

<!-- langtabs-start -->
```rust
use augurs::dtw::Dtw;

fn main() {
    // Sample time series data
    let series: &[&[f64]] = &[
        &[0.0, 1.0, 2.0, 3.0, 4.0],
        &[0.1, 1.1, 2.1, 3.1, 4.1],
    ];
    
    // Example: comparing different window sizes
    let dtw_small = Dtw::euclidean().with_window(2);
    let dtw_large = Dtw::euclidean().with_window(10);

    let matrix_small = dtw_small.distance_matrix(series);
    let matrix_large = dtw_large.distance_matrix(series);
    
    println!("Small window matrix: computed successfully");
    println!("Large window matrix: computed successfully");
}
```

```javascript
import { Dtw } from '@bsull/augurs/dtw';

// Sample time series data
const series = [
    [0.0, 1.0, 2.0, 3.0, 4.0],
    [0.1, 1.1, 2.1, 3.1, 4.1],
];

// Example: comparing different window sizes
const dtwSmall = new Dtw('euclidean', { window: 2 });
const dtwLarge = new Dtw('euclidean', { window: 10 });

const matrixSmall = dtwSmall.distanceMatrix(series);
const matrixLarge = dtwLarge.distanceMatrix(series);

console.log("Small window matrix:", matrixSmall);
console.log("Large window matrix:", matrixLarge);
```

```python
import augurs as aug
import numpy as np

# Sample time series data
series = [
    np.array([0.0, 1.0, 2.0, 3.0, 4.0]),
    np.array([0.1, 1.1, 2.1, 3.1, 4.1]),
]

# Example: comparing different window sizes
dtw_small = aug.Dtw(window=2, distance_fn='euclidean')
dtw_large = aug.Dtw(window=10, distance_fn='euclidean')

matrix_small = dtw_small.distance_matrix(series)
matrix_large = dtw_large.distance_matrix(series)

print("Small window matrix:", matrix_small)
print("Large window matrix:", matrix_large)
```
<!-- langtabs-end -->

### 2. Parameter Tuning

Start with reasonable defaults and adjust based on results:

<!-- langtabs-start -->
```rust
use augurs::{DistanceMatrix, dtw::Dtw, clustering::DbscanClusterer};

fn find_optimal_epsilon(distance_matrix: &DistanceMatrix) -> f64 {
    // Extract all unique distances (upper triangle, excluding diagonal)
    let mut distances: Vec<f64> = Vec::new();
    let (rows, _) = distance_matrix.shape();
    for row in distance_matrix.iter() {
        for &dist in row {
            if dist > 0.0 {
                distances.push(dist);
            }
        }
    }
    distances.sort_by(|a, b| a.partial_cmp(b).unwrap());
    distances[distances.len() / 2]
}

fn main() {
    let series: &[&[f64]] = &[
        &[0.0, 1.0, 2.0],
        &[0.1, 1.1, 2.1],
        &[5.0, 6.0, 7.0],
    ];
    
    let dtw = Dtw::euclidean();
    let distance_matrix = dtw.distance_matrix(series);
    
    let epsilon = find_optimal_epsilon(&distance_matrix);
    println!("Optimal epsilon: {}", epsilon);
    
    let clusters = DbscanClusterer::new(epsilon, 2).fit(&distance_matrix);
    println!("Clusters: {:?}", clusters);
}
```

```javascript
import { Dtw } from '@bsull/augurs/dtw';
import { DbscanClusterer } from '@bsull/augurs/clustering';

function findOptimalEpsilon(distanceMatrix) {
    // Extract all unique distances (upper triangle, excluding diagonal)
    const distances = [];
    const size = distanceMatrix.length;
    for (let i = 0; i < size; i++) {
        for (let j = i + 1; j < size; j++) {
            distances.push(distanceMatrix[i][j]);
        }
    }
    distances.sort((a, b) => a - b);
    return distances[Math.floor(distances.length / 2)];
}

// Sample data
const series = [
    [0.0, 1.0, 2.0],
    [0.1, 1.1, 2.1],
    [5.0, 6.0, 7.0],
];

const dtw = new Dtw('euclidean', { window: 3 });
const distanceMatrix = dtw.distanceMatrix(series);

const epsilon = findOptimalEpsilon(distanceMatrix);
console.log("Optimal epsilon:", epsilon);

const clusterer = new DbscanClusterer({ epsilon, minClusterSize: 2 });
const clusters = clusterer.fit(distanceMatrix);
console.log("Clusters:", clusters);
```

```python
import augurs as aug
import numpy as np

def find_optimal_epsilon(distance_matrix):
    """Find optimal epsilon using median distance."""
    # Convert DistanceMatrix to numpy array
    dm = distance_matrix.to_numpy()
    # Extract all unique distances (upper triangle, excluding diagonal)
    distances = []
    size = len(dm)
    for i in range(size):
        for j in range(i + 1, size):
            distances.append(dm[i][j])
    return np.median(distances)

# Sample data
series = [
    np.array([0.0, 1.0, 2.0]),
    np.array([0.1, 1.1, 2.1]),
    np.array([5.0, 6.0, 7.0]),
]

dtw = aug.Dtw(window=3, distance_fn='euclidean')
distance_matrix = dtw.distance_matrix(series)

epsilon = find_optimal_epsilon(distance_matrix)
print(f"Optimal epsilon: {epsilon}")

clusterer = aug.Dbscan(epsilon=epsilon, min_cluster_size=2)
clusters = clusterer.fit(distance_matrix)
print(f"Clusters: {clusters}")
```
<!-- langtabs-end -->

### 3. Performance Optimization

- **Downsample** very long time series before clustering
- **Use DTW bounds** to speed up calculations (Rust)
- **Cache** distance matrices for repeated clustering experiments

### 4. Language-Specific Considerations

- **Rust**: Full control over performance, can enable parallel processing
- **JavaScript**: WASM-based, good for web applications, limited by browser memory
- **Python**: Excellent NumPy integration, great for data science workflows

## Advanced Example: Analyzing Cluster Quality

Here's a more sophisticated example that analyzes clustering quality:

<!-- langtabs-start -->
```rust
use augurs::{
    clustering::{DbscanCluster, DbscanClusterer},
    dtw::Dtw,
};

fn analyze_clustering(series: &[&[f64]], epsilon: f64, min_size: usize) {
    // Compute distance matrix
    let distance_matrix = Dtw::euclidean()
        .with_window(3)
        .distance_matrix(series);

    // Perform clustering
    let clusters = DbscanClusterer::new(epsilon, min_size)
        .fit(&distance_matrix);

    // Analyze results
    let mut cluster_counts = std::collections::HashMap::new();
    let mut noise_count = 0;

    for cluster in &clusters {
        match cluster {
            DbscanCluster::Noise => noise_count += 1,
            DbscanCluster::Cluster(c) => {
                *cluster_counts.entry(*c).or_insert(0) += 1;
            }
        }
    }

    println!("Total series: {}", series.len());
    println!("Number of clusters: {}", cluster_counts.len());
    println!("Noise points: {}", noise_count);
    
    for (cluster_id, count) in cluster_counts {
        println!("Cluster {} has {} members", cluster_id, count);
    }
}
```

```javascript
import { Dtw } from '@bsull/augurs/dtw';
import { DbscanClusterer } from '@bsull/augurs/clustering';

function analyzeClustering(series, epsilon, minSize) {
    // Compute distance matrix
    const dtw = new Dtw('euclidean', { window: 3 });
    const distanceMatrix = dtw.distanceMatrix(series);

    // Perform clustering
    const clusterer = new DbscanClusterer({ 
        epsilon: epsilon, 
        minClusterSize: minSize 
    });
    const clusters = clusterer.fit(distanceMatrix);

    // Analyze results
    const clusterCounts = new Map();
    let noiseCount = 0;

    for (const cluster of clusters) {
        if (cluster === -1) {
            noiseCount++;
        } else {
            clusterCounts.set(cluster, (clusterCounts.get(cluster) || 0) + 1);
        }
    }

    console.log(`Total series: ${series.length}`);
    console.log(`Number of clusters: ${clusterCounts.size}`);
    console.log(`Noise points: ${noiseCount}`);
    
    for (const [clusterId, count] of clusterCounts) {
        console.log(`Cluster ${clusterId} has ${count} members`);
    }
}

// Example usage
const series = [
    [0.0, 1.0, 2.0, 3.0],
    [0.1, 1.1, 2.1, 3.1],
    [5.0, 6.0, 7.0, 8.0],
    [5.1, 6.1, 7.1, 8.1],
    [10.0, 11.0, 12.0, 13.0],
];

analyzeClustering(series, 0.5, 2);
```

```python
import augurs as aug
import numpy as np
from collections import Counter

def analyze_clustering(series, epsilon, min_size):
    """
    Analyze clustering quality and distribution.
    
    Args:
        series: List of time series arrays
        epsilon: DBSCAN epsilon parameter
        min_size: Minimum cluster size
    """
    # Compute distance matrix
    dtw = aug.Dtw(window=3, distance_fn='euclidean')
    distance_matrix = dtw.distance_matrix(series)

    print(f"Total series: {len(series)}")
    print("Distance matrix computed successfully!")
    
    # Clustering analysis will be available when Python API is complete
    print("Full clustering analysis coming soon to Python bindings!")

# Example usage
series = [
    np.array([0.0, 1.0, 2.0, 3.0]),
    np.array([0.1, 1.1, 2.1, 3.1]),
    np.array([5.0, 6.0, 7.0, 8.0]),
    np.array([5.1, 6.1, 7.1, 8.1]),
    np.array([10.0, 11.0, 12.0, 13.0]),
]

analyze_clustering(series, 0.5, 2)
```
<!-- langtabs-end -->

## Common Use Cases

1. **Pattern Discovery**: Find groups of time series with similar behavior
2. **Anomaly Detection**: Identify series that don't belong to any cluster (noise)
3. **Data Summarization**: Reduce large datasets by clustering and using cluster representatives
4. **Market Segmentation**: Group customers or products with similar temporal patterns
5. **Sensor Networks**: Identify sensors with correlated readings

## Troubleshooting

### Too Many Noise Points
- Decrease `epsilon` (allow smaller neighborhoods)
- Decrease `min_cluster_size`
- Check if DTW window is too restrictive

### All Points in One Cluster
- Increase `epsilon` (require tighter neighborhoods)
- Increase `min_cluster_size`
- Use a more discriminative distance measure

### Slow Performance
- Reduce DTW window size
- Downsample time series
- Use fewer series or pre-filter data

## Next Steps

- Learn about [outlier detection](./outlier-detection.md) for complementary anomaly detection
- Explore [forecasting](./forecasting-with-prophet.md) on clustered series
- Check the [API documentation](../api/index.md) for advanced clustering options

This guide provides a comprehensive introduction to time series clustering with practical examples across Rust, JavaScript, and Python, enabling you to group and analyze temporal patterns in your data effectively.
