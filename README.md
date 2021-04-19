# P1-Hierarchical-Clustering

# Naive algorithm implementation 

## Without Docker
    `pip3 install -r requirements.txt`
    `cd Task-1`
    `python3.7 naive.py`

## With Docker
    `docker build -t task1 -f Dockerfile.task1 .`
    `docker run -it task1:latest`

To change the size of the dataset used for this implementation, change line 287 in the naive.py file.
# Nearest-neighbor chain algorithm implementation

## Without Docker
    `pip3 install -r requirements.txt`
    `cd Task-2`
    `python3.7 nearest_neighbor.py`

## With Docker
    `docker build -t task2 -f Dockerfile.task2 .`
    `docker run -it task2:latest `
