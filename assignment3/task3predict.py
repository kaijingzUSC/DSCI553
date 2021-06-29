
# export PYSPARK_PYTHON=python3.6 

# spark-submit task3predict.py $ASNLIB/publicdata/train_review.json $ASNLIB/publicdata/test_review_ratings.json task3item.model task3item.predict item_based
# spark-submit task3predict.py $ASNLIB/publicdata/train_review.json $ASNLIB/publicdata/test_review_ratings.json task3user.model task3user.predict user_based

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

def build_dict(value_list):
	ans = collections.defaultdict(float)
	for item in value_list:
		ans[item[0]] = item[1]
	return ans

def single1(x):
	return x[1]

def add1(x, y):
	return list(x) + list(y)

def merage(value1, value2):
	if value1 != None and value2 != None:
		return sorted(add1(value1, value2), key = lambda x: single1(x), reverse = True)
	if value1 == None:
		if value2 != None:
			return sorted(list(value2), key = lambda x: single1(x), reverse = True)
		else: return None
	if value2 == None:
		if value1 != None:
			return sorted(list(value1), key = lambda x: single1(x), reverse = True)
		else: return None

def len1(x):
	return len(x)

def divide(x, y):
	return x / y

def multi(x, y):
	return x * y

def predict(model_dict, train_dict, item_avg_dict, item):
	user_inf = train_dict[item[0]]
	model_inf = model_dict[item[1]]
	if len1(model_inf) == 1: return None
	base = 0
	total = 0
	for business_inf in model_inf:
		if business_inf[0] in user_inf:
			base += abs(business_inf[1])
			total += multi(user_inf[business_inf[0]], business_inf[1])
	if base == 0 or total==0:
		return None
	return [item[0], single1(item), divide(total, base)]

def pred_ubase(model_dict, train_dict, rates, item):
	business_inf = train_dict[single1(item)]
	model_inf = model_dict[item[0]]
	if len1(model_inf) == 1: return None
	base = 0
	total = 0
	avg = divide(rates[item[0]][0], (rates[item[0]][1]+1))
	for usinf in model_inf:
		if usinf[0] in business_inf.keys():
			if usinf[1] < 0:
				ab = -usinf[1]
			else:
				ab = usinf[1]
			base += ab
			total += (business_inf[usinf[0]] - (rates[usinf[0]][0] - business_inf[usinf[0]]) / rates[usinf[0]][1]) \
			* usinf[1]
	if base == 0 or total == 0:
		return None
	res = [item[0], item[1], avg + divide(total, base)]
	return res

def boolfff(x):
	return x['u1']!=x['u2']

def boolfff2(x):
	return [x['user_id'], x['business_id']]

def boolff3(x):
	return x['b1'] != x['b2']

def main(argv):
	# Take arguments
	trainfile = argv[1]
	testfile = argv[2]
	modelfile = argv[3]
	outputfile = argv[4]
	cf_type = argv[5]

	# Initializing Spark
	conf = SparkConf().setAppName('assignment3').setMaster('local[*]')
	sc = SparkContext().getOrCreate(conf)
	
	if cf_type == 'user_based':
		RDD = sc.textFile(modelfile).map(json.loads)
		mod1 = RDD.filter(lambda x: boolfff(x)).map(lambda x: [x['u1'], (x['u2'], x['sim'])]).groupByKey()
		mod2 = RDD.filter(lambda x: boolfff(x)).map(lambda x: [x['u2'], (x['u1'], x['sim'])]).groupByKey()
		model = mod1.fullOuterJoin(mod2).map(lambda x: [x[0], merage(x[1][0], x[1][1])]).collect()
		model_dict = collections.defaultdict(list)
		i = 0
		while (i < len(model)):
			item = model[i]
			model_dict[item[0]] = item[1]
			i += 1
		RDD1 = sc.textFile(testfile).map(json.loads)
		RDD2 = sc.textFile(trainfile).map(json.loads)
		test = RDD1.map(lambda x: [x['user_id'], x['business_id']])
		train = RDD2.map(lambda x: [x['business_id'],(x['user_id'],x['stars'])]).groupByKey().mapValues(build_dict).collect()
		user_rate = RDD2.map(lambda x: (x['user_id'],x['stars'])).groupByKey().mapValues(lambda x: [sum(list(x)), len(list(x))-1]).collect()
		rates = collections.defaultdict(list)
		i = 0
		while (i < len(user_rate)):
			item = user_rate[i]
			rates[item[0]]= item[1]
			i += 1
		train_dict = collections.defaultdict(dict)
		i = 0
		while (i < len(train)):
			item = train[i]
			train_dict[item[0]] = item[1]
			i += 1
		result = test.map(lambda x: pred_ubase(model_dict,train_dict, rates, x)).filter(lambda x: x!=None).map(lambda item: json.dumps({'user_id': item[0], 'business_id': item[1], 'stars': item[2]})).collect()
		with open(outputfile, 'w') as f:
			for item in result:
				f.write(item)
				f.write('\n')
	elif cf_type == 'item_based':
		RDD = sc.textFile(modelfile).map(json.loads)
		mod1 = RDD.filter(lambda x: boolff3(x)).map(lambda x: [x['b1'],(x['b2'], x['sim'])]).groupByKey()
		mod2 = RDD.filter(lambda x: boolff3(x)).map(lambda x: [x['b2'],(x['b1'], x['sim'])]).groupByKey()
		model = mod1.fullOuterJoin(mod2).map(lambda x: [x[0], merage(x[1][0], x[1][1])]).collect()
		model_dict = collections.defaultdict(list)
		i = 0
		while (i < len(model)):
			item = model[i]
			model_dict[item[0]] = item[1]
			i += 1
		RDD1 = sc.textFile(testfile).map(json.loads)
		RDD2 = sc.textFile(trainfile).map(json.loads)
		test = RDD1.map(lambda x: boolfff2(x))
		train = RDD2.map(json.loads).map(lambda x: [x['user_id'], (x['business_id'], x['stars'])]).groupByKey().mapValues(build_dict).collect()
		itavg = RDD2.map(json.loads).map(lambda x: [x['business_id'], x['stars']]).groupByKey().mapValues(lambda x: divide(sum(list(x)), len(list(x)))).collect()
		item_avg_dict = collections.defaultdict(int)
		i = 0
		while (i < len(itavg)):
			item = itavg[i]
			item_avg_dict[item[0]] = item[1]
			i += 1
		train_dict = collections.defaultdict(dict)
		i = 0
		while (i < len(train)):
			item = train[i]
			train_dict[item[0]] = item[1]
			i += 1
		result = test.map(lambda x: predict(model_dict, train_dict, item_avg_dict, x)).filter(lambda x: x!=None).collect()
		with open(outputfile, 'w') as f:
			for item in result:
				json.dump({'user_id': item[0], 'business_id': item[1], 'stars': item[2]}, f)
				f.write('\n')

if __name__ == '__main__':
	start_time = time.time()
	if len(sys.argv) <= 1:
		sys.exit()
	main(sys.argv)
	print("Duration: %s" % int(time.time() - start_time))
