# export PYSPARK_PYTHON=python3.6 
# spark-submit task2.py $ASNLIB/publicdata/review.json $ASNLIB/publicdata/business.json task2_no_spark_ans no_spark 20
# spark-submit task2.py $ASNLIB/publicdata/review.json $ASNLIB/publicdata/business.json task2_spark_ans spark 20

# export PYSPARK_PYTHON=/usr/local/bin/python3.6
# export PYSPARK_DRIVER_PYTHON=/usr/local/bin/python3.6

import sys
import json
from functools import reduce
from operator import add
from collections import defaultdict
from pyspark import SparkContext, SparkConf, StorageLevel

def div(x, y):
	return x / y

def sum1(x, y):
	return (x[0] + y[0], x[1] + y[1])

def forfloat(x):
	return [float(v) for v in x]

def forstrip(x):
	return [v.strip() for v in x.split(',')]

def main(argv):
	# Take arguments
	review = argv[1]
	business = argv[2]
	output = argv[3]
	ifspark = argv[4]
	topn = int(argv[5])

	if ifspark == 'spark':
		# Initializing Spark
		conf = SparkConf().setAppName('assignment1').setMaster('local[*]')
		sc = SparkContext().getOrCreate(conf)

		# Initial Json
		jsonStrs = sc.textFile(review)
		reviewsRDD = jsonStrs.map(lambda s : json.loads(s))
		jsonStrs2 = sc.textFile(business)
		businessRDD = jsonStrs2.map(lambda s : json.loads(s))

		starRDD = reviewsRDD.map(lambda x: (x['business_id'], x['stars']))
		categoryRDD = businessRDD.map(lambda x: (x['business_id'], x['categories']))

		scoreRDD = starRDD.groupByKey().mapValues(lambda x: forfloat(x)).map(lambda y: (y[0], (sum(y[1]), len(y[1]))))
		tempcategoryRDD = categoryRDD.filter(lambda x: (x[1] != None) and x[1] != "").mapValues(lambda y: forstrip(y))

		joinRDD = tempcategoryRDD.leftOuterJoin(scoreRDD)

		sortRDD = joinRDD.map(lambda x: x[1]).filter(lambda y: y[1] != None)\
						.flatMap(lambda y: [(c, y[1]) for c in y[0]])\
						.reduceByKey(lambda a, b: sum1(a, b))\
						.mapValues(lambda z: float(div(z[0], z[1])))\
						.takeOrdered(topn, lambda y: (-y[1], y[0]))
		res = {'result': sortRDD}

	elif ifspark == 'no_spark':
		reviewpy = defaultdict(list)
		f = open(review, 'r')
		for line in f:
			load_dict = json.loads(line)
			reviewpy[load_dict['business_id']].append(load_dict['stars'])
		f.close()

		for i in reviewpy:
			reviewpy[i] = [(sum(reviewpy[i]), len(reviewpy[i]))]

		f = open(business, 'r')
		for line in f:
			load_dict = json.loads(line)
			if load_dict.get('categories') and load_dict['business_id'] in reviewpy:
				reviewpy[load_dict['business_id']].append(load_dict['categories'].split(', '))
		f.close()

		grouppy = defaultdict(list)
		for i in reviewpy:
			if len(reviewpy[i]) > 1:
				for j in reviewpy[i][1]:
					grouppy[j].append(reviewpy[i][0])

		for i in grouppy:
			temp = reduce(lambda x, y: sum1(x, y), grouppy[i])
			grouppy[i] = div(temp[0], temp[1])

		ret = sorted(grouppy.items(), key=lambda x: (-x[1], x[0]))[:topn]
		res = {'result': ret}

	# Writing JSON data
	with open(output, 'w') as f:
		json.dump(res, f)

if __name__ == '__main__':
	if len(sys.argv) <= 1:
		sys.exit()
	main(sys.argv)