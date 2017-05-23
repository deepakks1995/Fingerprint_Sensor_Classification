rm -r Downloads/testPath
rm Downloads/image.html
scp -r gpuuser@10.8.1.77:deepak/Project/Single-Finger-1000/Output Downloads/
scp gpuuser@10.8.1.77:deepak/Project/Single-Finger-1000/image.html Downloads/
scp -r gpuuser@10.8.1.77:deepak/Project/Single-Finger-1000/testPath Downloads/testPath
