#!/bin/bash

function check_download {
	if [ $? -eq 0 ]; then 
		echo "Download OK" 
	else 
		echo "There were problems downloading the file."
		exit
	fi
}
# Python dependencies
# Most are installed with pip3. However, Kepler Mapper is not there, so we
# need to manually fetch it.
if [ ! -f "km.py" ]; then 
	echo "Kepler Mapper not found"
	echo "Downloading Kepler Mapper"
	wget https://raw.githubusercontent.com/MLWave/kepler-mapper/master/km.py
	check_download
else
	echo "Kepler Mapper found"
fi
# Datasets
if [ ! -d "ml-100k" ]; then
	echo "MovieLens dataset not found"
	if [ -f "ml-100k.zip" ]; then
		echo "Found the zip though:"
		unzip ml-100k.zip
		exit
	fi
	echo "Downloading MovieLens 100k..."
	wget http://files.grouplens.org/datasets/movielens/ml-100k.zip
	check_download
	unzip ml-100k.zip
else
	echo "MovieLens dataset found"
fi

