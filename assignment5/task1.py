
# export PYSPARK_PYTHON=python3.6 

# spark-submit task1.py $ASNLIB/publicdata/business_first.json $ASNLIB/publicdata/business_second.json output1.csv

# export PYSPARK_PYTHON=/usr/local/bin/python3.6
# export PYSPARK_DRIVER_PYTHON=/usr/local/bin/python3.6

import csv
import sys
import time
import json
import random
import binascii
import itertools
from datetime import datetime
from pyspark import SparkContext, SparkConf, StorageLevel

def gethash(list1, list2, m, num):
	ans =[]
	i = 0
	while (i < len(list1)):
		ans.append((list1[i] * num + list2[i]) % m)
		i += 1
	return ans

def predict(list1, list2, m, num, bit_array):
	if num == "": 
		return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
	num = int(binascii.hexlify(num.encode("utf8")), 16)
	hash_res = gethash(list1, list2, m, num)
	i = 0
	while (i < len(hash_res)):
		item = hash_res[i]
		if bit_array[item] != 1: 
			return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
		i += 1
	return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def boolfunc(x):
	if "city" not in x:
		return ""
	else:
		return x["city"]

def listfunc(x):
	return [i for i in range(x)]

def multi(x, y):
	return x * y

def main(argv):
	# Take arguments
	trainfile = argv[1]
	predictfile = argv[2]
	outputfile = argv[3]

	# Initializing Spark
	conf = SparkConf().setAppName('assignment5').setMaster('local[*]')
	sc = SparkContext().getOrCreate(conf)
	trainRDD = sc.textFile(trainfile)
	testRDD = sc.textFile(predictfile)

	train_city = trainRDD.map(json.loads).map(lambda x: boolfunc(x))\
	.filter(lambda x: x != "").distinct()\
	.map(lambda s: int(binascii.hexlify(s.encode("utf8")), 16))
	test_city = testRDD.map(json.loads).map(lambda x: boolfunc(x))

	m = multi(train_city.count(), 50)
	list1 = random.sample(listfunc(m), 5)
	list2 = random.sample(listfunc(m), 5)
	bit_array = multi([0], m)
	train_res = train_city.flatMap(lambda x: gethash(list1, list2, m, x)).distinct()
	for num in train_res.collect(): 
		bit_array[num] = 1
	predict_res =  test_city.map(lambda x: predict(list1, list2, m, x, bit_array))

	# Writing file
	with open(outputfile, 'w') as f:
		wr = csv.writer(f, delimiter=' ')
		wr.writerow(predict_res.collect())

if __name__ == '__main__':
	start_time = time.time()
	if len(sys.argv) <= 1:
		sys.exit()
	main(sys.argv)
	print("Duration: %s" % int(time.time() - start_time))

