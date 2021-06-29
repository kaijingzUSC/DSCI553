
# export PYSPARK_PYTHON=python3.6

# spark-submit task2train.py $ASNLIB/publicdata/train_review.json task2.model $ASNLIB/publicdata/stopwords

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

def getword(stopwords, text):
	total = " ".join(text)
	text = re.sub('[^A-Za-z]+', ' ', total).split()
	freq = collections.defaultdict(int)
	max_freq = 0
	i = 0
	while (i < len(text)): 
		x = text[i].lower()
		if x not in stopwords:
			freq[x] += 1
			max_freq =max(max_freq, freq[x])
		i += 1

	ans = list()
	for x in freq.keys():
		if freq[x] >= len(text)*0.000001:
			ans.append([x, freq[x]/max_freq])
		i += 1
	return ans

def func_TF_IDF(IDF_dict, tf):
	ans = list()
	i = 0
	while (i < len(tf)):
		item = tf[i]
		ans.append([item[0], item[1]*IDF_dict[item[0]]])
		i += 1
	ans = sorted(ans, key = lambda x: x[1], reverse= True)[:min(200, len(ans))]

	res = list()
	i = 0
	while (i < len(ans)):
		item = ans[i]
		res.append(item[0])
		i += 1
	return res

def build_profile(profile_dict, value_list):
	ans = []
	freq = collections.defaultdict(int)
	for item in value_list:
		i = 0
		while (i < len(profile_dict[item])):
			word = profile_dict[item][i]
			freq[word] += 1
			i += 1
	ans = sorted(list(freq.keys()), key = lambda x: freq[x], reverse = True)[:min(400, len(freq.keys()))]
	return ans

def list1(x):
	return [x[0],x[1][0]]

def single1(x):
	return x[1]

def list2(x):
	return [x[0],1]

def main(argv):
	# Take arguments
	inputfile = argv[1]
	outputfile = argv[2]
	stopword = argv[3]

	# Initializing Spark
	conf = SparkConf().setAppName('assignment3').setMaster('local[*]')
	sc = SparkContext().getOrCreate(conf)
	RDD = sc.textFile(inputfile)
	review = RDD.map(json.loads).map(lambda x: [x['user_id'], [x['business_id'], x['text']]])
	user_business = review.map(lambda x: list1(x)).groupByKey()

	TF = review.map(lambda x: single1(x)).groupByKey().mapValues(lambda x: getword(stopword, x))
	total_business = TF.count()
	IDF = TF.flatMap(lambda x: single1(x)).map(lambda x: list2(x)).reduceByKey(add).map(lambda x: [x[0],math.log2(total_business/x[1])]).collect()
	IDF_dict = dict()

	i = 0
	while (i < len(IDF)):
		item = IDF[i]
		IDF_dict[item[0]] = item[1]
		i += 1

	TF_IDF = TF.mapValues(lambda x: func_TF_IDF(IDF_dict,x)).collect()
	business_profile_dict = collections.defaultdict(set)

	# Writing file
	with open(outputfile, 'w') as f:
		i = 0
		while (i < len(TF_IDF)):
			item = TF_IDF[i]
			json.dump({'type': 'business', 'name': item[0], 'profile': item[1]}, f)
			business_profile_dict[item[0]] = item[1]
			f.write('\n')
			i += 1

		user_profile = user_business.mapValues(lambda x: build_profile(business_profile_dict, x)).map(lambda item: json.dumps({'type': 'user', 'name': item[0], 'profile': item[1]})).collect()

		for item in user_profile:
			f.write(item)
			f.write('\n')

if __name__ == '__main__':
	start_time = time.time()
	if len(sys.argv) <= 1:
		sys.exit()
	main(sys.argv)
	print("Duration: %s" % int(time.time() - start_time))

