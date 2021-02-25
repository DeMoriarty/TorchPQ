# TorchPQ
TorchPQ is a python library for **Approximate Nearest Neighbor Search** (ANNS) and **Maximum Inner Product Search** (MIPS) on GPU using **Product Quantization** (PQ) algorithm. TorchPQ is implemented mainly with PyTorch, with some extra CUDA kernels to accelerate clustering, indexing and searching.

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
Notice that the shape of the tensor that contains data points has to be ```[d_vector, n_data]```, this is consistant in TorchPQ.

#### Multiple concurrent K-means
Sometimes, we have multiple independent datasets that need to be clustered,
instead of running multiple KMeans sequentianlly,
we can perform multiple kmeans concurrently with **MultiKMeans**
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
#### Prediction with K-means
```
labels = kmeans.predict(x)
```

### IVFPQ
**I**n**V**erted **F**ile **P**roduct **Q**uantization (IVFPQ) is a type of ANN search algorithm that is designed to do fast and efficient vector search in million, or even billion scale vector sets. check the [original paper](https://hal.inria.fr/inria-00514462v2/document) for more details.  

#### Training
```python
from torchpq import IVFPQ

index = IVFPQ(
  d_vector=d_vector,
  n_subvectors=64,
  n_cq_clusters=1024,
  n_pq_clusters=256,
  blocksize=128,
  distance="euclidean",
)

x = torch.randn(d_vector, n_data)
index.train(x)
```
There are some important parameters that need to be explained:
- **d_vector**: dimentionality of input vectors. there are 2 constraints on `d_vector`: (1) it needs to be divisible by `n_subvectors`; (2) it needs to be a multiple of 4.*
- **n_subvectors**: number of subquantizers, essentially this is the byte size of each quantized vector, 64 byte per vector in the above example.**
- **n_cq_clusters**: number of coarse quantizer clusters
- **n_pq_clusters**: number of product quantizer clusters, this is assumed to be 256 throughout the entire project, and should **NOT** be changed.
- **blocksize**: initial capacity assigned to each voronoi cell of coarse quantizer.
`n_cq_clusters * blocksize` is the number of vectors that can be stored initially. if any cell has reached its capacity, that cell will be automatically expanded.
If you need to add vectors frequently, a larger value for `blocksize` is recommended.

\* the second constraint could be removed in the future  
\*\* actual byte size would be (n_subvectors+9) bytes, 8 bytes for ID and 1 byte for is_empty
#### Adding new vectors
```python
ids = torch.arange(n_data, device="cuda")
index.add(x, input_ids=ids)
```
Each ID in `ids` needs to be a unique int64 (`torch.long`) value that indentifies a vector in `x`.
if `input_ids` is not provided, it will be set to `torch.arange(n_data, device="cuda") + previous_max_id`

#### Removing vectors
```python
index.remove(ids)
```
`index.remove(ids)` will virtually remove vectors with specified `ids` from storage.
It ignores ids that doesn't exist.

#### Topk search
```python
n_query = 10000
query = torch.randn(d_vector, n_query, device="cuda:0")
topk_values, topk_ids = index.topk(query, k=100)
```
- when `distance="inner"`, `topk_values` are **inner product** of queries and topk closest data points.
- when `distance="euclidean"`, `topk_values` are **negative squared L2 distance** between queries and topk closest data points.
- when `distance="manhattan"`, `topk_values` are **negative L1 distance** between queries and topk closest data points.
- when `distance="cosine"`, `topk_values` are **cosine similarity** between queries and topk closest data points.

#### Encode and Decode
you can use IVFPQ as a vector codec for lossy compression of vectors
```python
code = index.encode(query)   # compression
reconstruction = index.decode(code) # reconstruction
```

### Save and Load
Most of the TorchPQ modules are inherited from `torch.nn.Module`, this means you can save and load them just like a regular pytorch model.
```python
# Save to PATH
torch.save(index.state_dict(), PATH)
# Load from PATH
index.load_state_dict(torch.load(PATH))
```

## Benchmark
All experiments were performed with a Tesla T4 GPU.

### SIFT1M
#### IVFPQ
[Faiss](https://github.com/facebookresearch/faiss) is one of the most well known ANN search libraries, and it also has a GPU implementation of IVFPQ, so we did some comparison experiments with faiss.  
<p float="left">
  <img src="/imgs/6.png" width="100%"/>
</p>  

- for all the IVF16384 variants, torchpq outperforms faiss when n_probe > 16.
- for IVF4096, torchpq has lower recall@1 compared to faiss, this could be caused by not encoding residuals. An option to encode residuals will be added soon.
#### IVFPQ+R
<p float="left">
  <img src="/imgs/tiny/1.png" width="49%"/>
  <img src="/imgs/tiny/2.png" width="49%"/>
</p>  

### GIST1M
coming soon...
