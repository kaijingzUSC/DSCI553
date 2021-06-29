
# export PYSPARK_PYTHON=python3.6 

# spark-submit task2predict.py $ASNLIB/publicdata/test_review_ratings.json task2.model task2.predict

# export PYSPARK_PYTHON=/usr/local/bin/python3.6
# export PYSPARK_DRIVER_PYTHON=/usr/local/bin/python3.6
import re
import sys
import math
import time
import json
import random
import itertools
import collections
from operator import add
from pyspark import SparkContext, SparkConf, StorageLevel

def divide(x, y):
	return x / y

def multi(x, y):
	return x * y

def predict1(user, busi, x):
	user_profile = user[x[0]]
	business_profile = busi[x[1]]
	intersection_res = set(user_profile).intersection(set(business_profile))
	base = multi(math.sqrt(len(user_profile)), math.sqrt(len(business_profile)))
	if(base == 0):
		return [x[0],x[1], 0.0]
	res = divide(len(intersection_res), base)
	return [x[0],x[1], res]

def bool1(x):
	return x[2] >= 0.01

def retkey(x):
	return [x['user_id'],x['business_id']]

def main(argv):
	# Take arguments
	testfile = argv[1]
	modelfile = argv[2]
	outputfile = argv[3]

	# Initializing Spark
	conf = SparkConf().setAppName('assignment3').setMaster('local[*]')
	sc = SparkContext().getOrCreate(conf)
	RDD = sc.textFile(modelfile)
	model = RDD.map(json.loads)

	user_dict = collections.defaultdict(list)
	user_profile = model.filter(lambda x: x['type'] == 'user').map(lambda x: [x['name'], x['profile']]).collect()
	i = 0
	while (i < len(user_profile)):
		item = user_profile[i]
		user_dict[item[0]] = item[1]
		i += 1

	business_dict = collections.defaultdict(list)
	business_profile = model.filter(lambda x: x['type'] == 'business').map(lambda x: [x['name'], x['profile']]).collect()
	i = 0
	while (i < len(business_profile)):
		item = business_profile[i]
		business_dict[item[0]] = item[1]
		i += 1

	test = sc.textFile(testfile).map(json.loads).map(lambda x: retkey(x))
	result = test.map(lambda x: predict1(user_dict, business_dict, x)).filter(lambda x: bool1(x))\
	.map(lambda x: json.dumps({"user_id": x[0], "business_id": x[1], "stars": x[2]})).collect()
	
	# Writing file
	with open(outputfile, 'w') as f:
		for item in result:
			f.write(item)
			f.write('\n')

if __name__ == '__main__':
	start_time = time.time()
	if len(sys.argv) <= 1:
		sys.exit()
	main(sys.argv)
	print("Duration: %s" % int(time.time() - start_time))

