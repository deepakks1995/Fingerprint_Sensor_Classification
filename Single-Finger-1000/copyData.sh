rm -r Downloads/testPath
rm Downloads/image.html
scp -r gpuuser@10.8.1.77:deepak/Single-Finger/Output Downloads/Output
scp -r gpuuser@10.8.1.77:deepak/Single-Finger-1000/testPath Downloads/testPath
scp gpuuser@10.8.1.77:deepak/Single-Finger-1000/image.html Downloads/
