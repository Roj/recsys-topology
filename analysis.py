import km
import sklearn as sk
import numpy as np
import csv
import sys
import argparse


# Create configuration arguments
parser = argparse.ArgumentParser(
	description = "Topological Analysis of Recommender Systems"
)

parser.add_argument(
	"-l", "--label",
	default = "N",
	choices = ["N", "G"],
	help = "N: names\n G: genres",
	dest = "label"
)
parser.add_argument(
	"-o", "--output",
	default = "output.html",
	dest = "output"
)
config = parser.parse_args(sys.argv[1:])


data = np.genfromtxt("ml-100k/u1.base", delimiter="\t")
for i in range(2,5+1):
	filename = "ml-100k/u{0}.base".format(i)
	data = np.append(data, np.genfromtxt(filename, delimiter="\t"), axis = 0)

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

# Search for genres
genres = []
with open("ml-100k/u.genre", encoding = "ISO-8859-1") as genresfile:
	genrescsv = csv.reader(genresfile, delimiter="|")
	for genre in genrescsv:
		try:
			genres.append(genre[0])
		except IndexError as e:
			pass


# Search for labels (movie names or movie genres)
labels = []
with open("ml-100k/u.item", encoding = "ISO-8859-1") as labelsfile:
	labelscsv = csv.reader(labelsfile, delimiter="|")
	for label in labelscsv:
		if config.label == "N":
			labels.append(label[1])
		elif config.label == "G":
			labels.append([genres[i] for i,t in enumerate(label[5:]) if t == "1"])

# Mapper
mapper = km.KeplerMapper(verbose = 2)
# Temporary test: transpose the matrix to see how each movie is projected
lens = mapper.fit_transform(
	useritem.T,
	scaler=None,
	projection= sk.decomposition.PCA(n_components=1)
)   

graph = mapper.map(
	lens,
	useritem.T,
	nr_cubes=100,
	overlap_perc=0.3,
	clusterer=sk.cluster.DBSCAN(eps=0.8, min_samples=3, metric="cosine", algorithm="brute")
)

mapper.visualize(
	graph,
	path_html = config.output,
	custom_tooltips = np.array(labels)
)
