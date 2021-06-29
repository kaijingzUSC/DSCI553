
# export PYSPARK_PYTHON=python3.6 

# spark-submit --packages graphframes:graphframes:0.6.0-spark2.3-s_2.11 task1.py 7 $ASNLIB/publicdata/ub_sample_data.csv output1.txt

# export PYSPARK_PYTHON=/usr/local/bin/python3.6
# export PYSPARK_DRIVER_PYTHON=/usr/local/bin/python3.6

import sys
import time
import json
import random
from operator import add
from functools import reduce
from graphframes import GraphFrame
from itertools import combinations
from pyspark import SparkContext, SparkConf, StorageLevel, SQLContext

def boolfunc1(x, y):
	return x[2] >= y

def splitfunc(x):
	return x.split(',')

def main(argv):
	# Take arguments
	threshold = int(argv[1])
	inputfile = argv[2]
	outputfile = argv[3]

	# Initializing Spark
	conf = SparkConf().setAppName('assignment4').setMaster('local[*]')
	sc = SparkContext().getOrCreate(conf)
	sc.setLogLevel("WARN")
	dateRDD = sc.textFile(inputfile)

	data = dateRDD.map(lambda x: splitfunc(x)).filter(lambda x: x[0] != 'user_id')
	user_pairs = data.map(lambda x: [x[1],x[0]]).groupByKey().mapValues(sorted)\
	.mapValues(lambda x: combinations(x, 2)).flatMap(lambda x: x[1])\
	.flatMap(lambda x: [[x, 1], [x[::-1],1]]).reduceByKey(add)\
	.map(lambda x: [x[0][0],x[0][1], x[1]]).filter(lambda x: boolfunc1(x, threshold))
	users = user_pairs.flatMap(lambda x: x[:2]).distinct().map(lambda x: [x])

	sql_sc = SQLContext(sc)
	vertices = sql_sc.createDataFrame(users, ['id'])
	edges = sql_sc.createDataFrame(user_pairs, ["src", "dst", "intersection"])
	graph = GraphFrame(vertices, edges)
	result = graph.labelPropagation(maxIter=5).select('id', 'label')\
	.rdd.map(lambda x: [x[1],x[0]]).groupByKey().map(lambda x: sorted(list(x[1])))\
	.sortBy(lambda x: x[0]).sortBy(len)

	# Writing file
	with open(outputfile, 'w') as f:
		for item in result.collect():
			f.write("'" + "', '".join(item) + "'" + "\n")

if __name__ == '__main__':
	start_time = time.time()
	if len(sys.argv) <= 1:
		sys.exit()
	main(sys.argv)
	print("Duration: %s" % int(time.time() - start_time))

