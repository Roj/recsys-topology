import km
import sklearn as sk
import numpy as np   
import csv
import pandas as pd

data = np.genfromtxt("ml-100k/u1.base", delimiter="\t")

# Assume, for now, that IDs are consecutive.
# They mostly are.
U = int(np.max(data[:, 0])) #Number of users
M = int(np.max(data[:, 1])) #Number of movies

useritem = np.zeros((U,M))
for row in data:
    useritem[int(row[0])-1,int(row[1])-1]=row[2]

# Normalize the matrix such that rows sum to 1
rowsums = useritem.sum(axis = 1)
useritem = useritem/rowsums[:, np.newaxis]

del data

# Search for labels (movie names)
labels = []
with open("ml-100k/u.item", encoding = "ISO-8859-1") as labelsfile:
	labelscsv = csv.reader(labelsfile, delimiter="|")
	for label in labelscsv:
		labels.append(label[1])

# Mapper
mapper = km.KeplerMapper(verbose = 2)
# Temporary test: transpose the matrix to see how each movie is projected
lens = mapper.fit_transform(
    useritem.T, 
    projection=sk.decomposition.PCA(n_components=1) 
)
graph = mapper.map(
    lens,
    useritem.T,
    nr_cubes=15,
    overlap_perc=0.2
)

mapper.visualize(graph, path_html="output.html", custom_tooltips=np.array(labels))
