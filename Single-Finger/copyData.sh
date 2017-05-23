rm -r Downloads/testPath
rm Downloads/image.html
scp -r gpuuser@10.8.1.77:deepak/Project/Single-Finger/Output Downloads/
scp -r gpuuser@10.8.1.77:deepak/Project/Single-Finger/testPath Downloads/testPath
scp gpuuser@10.8.1.77:deepak/Project/Single-Finger/image.html Downloads/
