rm -r Downloads/testPath
rm Downloads/image.html
scp -r gpuuser@10.8.1.77:deepak/Project/Multi-Slab/Output Downloads/
scp gpuuser@10.8.1.77:deepak/Project/Multi-Slab/image.html Downloads/
scp -r gpuuser@10.8.1.77:deepak/Project/Multi-Slab/testPath Downloads/testPath
