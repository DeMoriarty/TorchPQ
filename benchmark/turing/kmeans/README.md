# Benchmark

## K-Means
Performing K-Means clustering on float32 data randomly sampled from normal distribution.  
- Number of iterations is set to 15.   
- Tolerance is set to 0 in order to perform full 15 iterations of K-Means   
- Initial centroids are randomly chosen from training data   
- All runs are performed on a Tesla T4 GPU    

### Contestants:
- torchpq.clustering.KMeans
- faiss.Clustering
- [KeOps](https://www.kernel-operations.io/keops/_auto_tutorials/kmeans/plot_kmeans_torch.html)  


#### n_features=256, n_clusters=256, varying n_data
<p float="left">
  <img src="imgs/n_clusters=256 n_features=256.png" width="100%"/>
</p>  

#### n_features=256, n_clusters=16384, varying n_data
<p float="left">
  <img src="imgs/n_clusters=16384 n_features=256.png" width="100%"/>
</p>  

#### n_features=128, n_data=1,000,000, varying n_clusters
<p float="left">
  <img src="imgs/n_data=1000000 n_features=128.png" width="100%"/>
</p>  

#### n_clusters=1024, n_data=1,000,000, varying n_features
<p float="left">
  <img src="imgs/n_data=1000000 n_clusters=1024.png" width="100%"/>
</p>  
note: faiss and keOps went OOM when n_features > 512
