
# export PYSPARK_PYTHON=python3.6 

# spark-submit task2.py 9999 output2.csv

# export PYSPARK_PYTHON=/usr/local/bin/python3.6
# export PYSPARK_DRIVER_PYTHON=/usr/local/bin/python3.6

import sys
import json
import random
import time
import csv
import math
import collections
from datetime import datetime
from pyspark.streaming import StreamingContext
from pyspark import SparkContext, SparkConf, StorageLevel

global bool1

def divide(x, y):
	return x / y

def forfunc(x):
	return [random.randint(0, (2 ** 16) - 1) for i in range(x)]

def multi(x, y):
	return x * y

def add(x, y):
	return x + y

def Fajoet_Martion(data):
	n = 2 ** 10 - 1
	hash_number = 12
	stream = data.collect()
	truth = len(set(stream))
	if bool1:
		outputfile = open(output_file, 'w')
		outputfile.write("Time, Ground Truth, Estimation\n")
		bool1 = False
	else:
		outputfile = open(output_file, 'a')

	starttime = time.time()
	start = datetime.datetime.fromtimestamp(starttime).strftime('%Y-%m-%d %H:%M:%S')
	list1 = forfunc(hash_number)
	list2 = forfunc(hash_number)
	listR = list()
	i = 0
	while (i < hash_number):
		max_zero = -1
		j = 0
		while (j < len(stream)):
			data = stream[j]
			codata = int(binascii.hexlify(data.encode("utf8")), 16)
			hash_value = add(multi(list1[i], codata), list2[i]) % n
			binary_val = format(hash_value, '032b')
			if hash_value == 0:
				num_zero = 0
			else:
				num_zero = len(str(binary_val)) - len(str(binary_val).rstrip("0"))
			if num_zero > max_zero:
				max_zero = num_zero
			j += 1
		listR.append(2 ** max_zero)
		i += 1

	avg1 = divide(sum(listR[:4]), len(listR[:4]))
	avg2 = divide(sum(listR[4:8]), len(listR[4:8]))
	avg3 = divide(sum(listR[8:12]), len(listR[8:12]))
	ls = [avg1,avg2,avg3]
	ls.sort()
	outputfile.write(str(start) + "," + str(truth) + "," + str(int(ls[1])) + "\n")
	outputfile.close()

def main(argv):
	# Take arguments
	port = int(argv[1])
	output_file = argv[2]

	# Initializing Spark
	conf = SparkConf().setAppName('assignment5').setMaster('local[*]')
	sc = SparkContext().getOrCreate(conf)

	ssc = StreamingContext(sc, 5)
	lines = ssc.socketTextStream("localhost", port)
	bool1 = True
	rdd = (lines.window(30, 10).map(json.loads).map(lambda x: x["city"])\
		.foreachRDD(Fajoet_Martion))

if __name__ == '__main__':
	start_time = time.time()
	if len(sys.argv) <= 1:
		sys.exit()
	main(sys.argv)
	print("Duration: %s" % int(time.time() - start_time))

