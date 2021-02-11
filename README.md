# TorchPQ
TorchPQ is a python library for Approximate Nearest Neighbor Search (ANNS) and Maximum Inner Product Search (MIPS) on GPU using Product Quantization (PQ) algorithm. TorchPQ is implemented mainly with PyTorch, with some extra CUDA kernels to accelerate clustering, indexing and searching.

## Install
First install a version of CuPy library that matches your CUDA version
```
pip install cupy-cuda90
pip install cupy-cuda100
pip install cupy-cuda101
...
```
Then install TorchPQ
```
pip install torchpq
```
for a full list of cupy-cuda versions, please go to [Installation Guide](https://docs.cupy.dev/en/stable/install.html#installing-cupy)

## Quick Start
### K-means clustering
```python
from torchpq.kmeans import KMeans
import torch

n_data = 1000000 # number of data points
d_vector = 128 # dimentionality / number of features
x = torch.randn(d_vector, n_data, device="cuda")
kmeans = KMeans(n_clusters=4096, distance="euclidean")
labels = kmeans.fit(x)
```
Notice that shape of the tensor that contains data points has to be ```[d_vector, n_data]```, this is consistant in TorchPQ.

### Multiple concurrent K-means
Sometimes, we have multiple independent datasets that we want to cluster,
instead of having multiple KMeans and performing the clustering sequentianlly,
we can run multiple kmeans concurrently with MultiKMeans
```python
from torchpq.kmeans import MultiKMeans
import torch

n_data = 1000000
n_kmeans = 16
d_vector = 64
x = torch.randn(n_kmeans, d_vector, n_data, device="cuda")
kmeans = MultiKMeans(n_clusters=256, distance="euclidean")
labels = kmeans.fit(x)
```
### Prediction with K-means
```
labels = kmeans.predict(x)
```

### Training IVFPQ
```
from torchpq import IVFPQ
import torch

x = torch.randn(d_vector, n_data)
index = IVFPQ(
  d_vector=d_vector,
  n_subvectors=64,
  n_cq_clusters=1024,
  n_pq_clusters=256,
  blocksize=128,
  distance="euclidean",
)

index.train(x)
```
There are some important parameters that needs to be explained:  
- d_vector: dimentionality of input vectors. `d_vector` needs to be divisible by `n_subvectors`, `d_vector` also needs to be a multiple of 4.*
- n_subvectors: number of subquantizers, essentially this is the byte size of each quantized vector, 64Byte/vector in the above example.**
- n_cq_clusters: number of coarse quantizer clusters
- n_pq_clusters: number of product quantizer clusters, this is assumed to be 256 throughout the entire project, and should not be changed.
- blocksize: initial capacity assigned to each voronoi cell of coarse quantizer.
`n_cq_clusters * blocksize` is the number of vectors that can be stored initially. if any cell has reached its capacity, that cell will be automatically expanded.
larger value for `blocksize` is recommended, if you need to add vectors frequently.

\* the second constraint could be removed in the future
\*\* actual byte size would be (n_subvectors+9) bytes, 8 bytes for ID and 1 byte for is_empty
### Adding vectors
```
ids = torch.arange(n_data, device="cuda")
index.add(x, input_ids=ids)
```
Each ID in `ids` is a unique int64 value that corresponds to a vector in `x`.
if `input_ids` is not provided to `index.add` (or `input_ids=None`), it will be set to `torch.arange(n_data, device="cuda") + previous_max_id`

### Removing vectors
```
index.remove(ids)
```
`index.remove(ids)` will virtually remove vectors with specified `ids` from storage.
It will ignore ids that doesn't exist.

### Topk search
```
n_query = 10000
query = torch.randn(d_vector, n_query, device="cuda:0")
topk_values, topk_ids = index.topk(query, k=100)
```
- when `distance="inner"`, `topk_values` are maximum inner product of queries and topk closest data points.
- when `distance="euclidean"`, `topk_values` are maximum negative squared L2 distance between queries and topk closest data points.
- when `distance="manhattan"`, `topk_values` are maximum negative L1 distance between queries and topk closest data points.
- when `distance="cosine"`, `topk_values` are maximum cosine similarity between queries and topk closest data points.

### Encoding and Decoding
```
code = index.encode(query)
reconstruction = index.decode(code)
```

## Benchmark
TODO
