
# export PYSPARK_PYTHON=python3.6 

# spark-submit task3train.py $ASNLIB/publicdata/train_review.json task3item.model item_based
# spark-submit task3train.py $ASNLIB/publicdata/train_review.json task3user.model user_based

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

def single1(x):
	return x[1]

def create_list(tlist):
	sum_value = 0
	rate_dict = collections.defaultdict(int)
	for item in tlist:
		rate_dict[item[0]] = single1(item)
		sum_value += single1(item)
	return rate_dict

def divide(x, y):
	return x / y

def create_model(buis_base, comb_value):
	d1 = buis_base[comb_value[0]]
	d2 = buis_base[comb_value[1]]
	set1 = set(d1.keys())
	set2 = set(d2.keys())
	intersect = set1.intersection(set2)
	if len(intersect) >= 3:
		sum1 = 0
		sum2 = 0
		total = 0 
		base1 = 0
		base2 =0
		i = 0
		intersect = list(intersect)
		while i < (len(intersect)): 
			sum1 += d1[intersect[i]]
			sum2 += d2[intersect[i]]
			i += 1
		avg1 =  divide(sum1, len(intersect))
		avg2 = divide(sum2, len(intersect))

		i = 0
		while (i < len(intersect)):
			total += (d1[intersect[i]]-avg1) * (d2[intersect[i]]-avg2)
			base1 += (d1[intersect[i]]-avg1) * (d1[intersect[i]]-avg1)
			base2 += (d2[intersect[i]]-avg2) * (d2[intersect[i]]-avg2)
			i += 1

		if total <=0 or base1 == 0 or base2 ==0: 
			return None
		sim = total/(math.sqrt(base1) * math.sqrt(base2))
		res = [comb_value[0], comb_value[1], sim]
		return res
	return None

def build_user_list(users_dict, rated_list):
	res = []
	for x in set(rated_list):
		res.append(users_dict[x])
	if len(res) >= 3:
		return sorted(res)
	return None

def cal_min_hash(a, b, m, value_list):
	ans = list()
	i = 0
	while (i < len(b)):
		cur_min = m + 1
		j = 0
		while (j < len(value_list)):
			x = value_list[j]
			cur_min= min(cur_min, (a[i]*x + b[i])%m)
			j += 1
		ans.append(cur_min)
		i += 1
	return ans

def strc(x, i):
	return "0" + "," + str(x[i-2]) + str(x[i-1])

def split_list(value_list, size):
	bands = []
	for i in range(size, len(value_list)+1, size):
		cur = strc(value_list, i)
		bands.append(cur)
	return bands

def combination(value_list):
	value_list = sorted(value_list)
	if(len(value_list) == 2):
		return [value_list]
	ans = []
	i = 0
	while (i < len(value_list)):
		j = i + 1
		while (j < len(value_list)):
			ans.append(sorted([value_list[i], value_list[j]]))
			j += 1
		i += 1
	return ans

def strcom1(x):
	return x[0] + "," + x[1]

def foo0(x):
	return x[0]

def foo1(x):
	return x[1]

def bool0(x):
	return x[0]!= x[1]

def retlist(x):
	return [i for i in range(len(x))]

def boolfunc(l, x):
	return len(l[x]) >= 3

def main(argv):
	# Take arguments
	inputfile = argv[1]
	outputfile = argv[2]
	cf_type = argv[3]

	# Initializing Spark
	conf = SparkConf().setAppName('assignment3').setMaster('local[*]')
	sc = SparkContext().getOrCreate(conf)
	RDD = sc.textFile(inputfile)
	review = RDD.map(json.loads)
	
	# Writing file
	if cf_type == 'user_based':
		business_list = review.map(lambda x: x['business_id']).distinct().collect()
		business_dict = collections.defaultdict(int)
		i = 0 
		while (i < len(business_list)):
			x = business_list[i]
			business_dict[x] = i
			i += 1
		rated = review.map(lambda x: [x['user_id'], x['business_id']]).groupByKey().mapValues(lambda x: build_user_list(business_dict, x)).filter(lambda x: x[1] != None).sortByKey()
		min_hash = rated.mapValues(lambda x: cal_min_hash(random.sample(retlist(business_list), 450), random.sample(retlist(business_list), 450), len(business_list), x))
		bands = min_hash.flatMapValues(lambda x: split_list(x, 2)).map(lambda x: [x[1],x[0]]).groupByKey().filter(lambda x: len(x[1]) >= 2).map(lambda x: list(x[1]))

		u_base = review.map(lambda x: [x['user_id'], (x['business_id'],x['stars'])]).groupByKey().mapValues(create_list).collect()

		b_dict = {}
		for item in u_base:
			key = foo0(item)
			if key in b_dict.keys():
				b_dict[key] = foo1(item)
			else:
				b_dict[key] = {}
				b_dict[key] = foo1(item)
		candidates = bands.flatMap(lambda x: combination(x)).map(lambda x: tuple(x)).distinct().filter(lambda x: bool0(x))

		model = candidates.map(lambda x: create_model(b_dict, x)).filter(lambda x: x!=None).collect()
		with open(outputfile, 'w') as file:
			for item in model:
				if item != None:
					json.dump({'u1': item[0], 'u2': item[1], 'sim': item[2]}, file)
					file.write('\n')
	elif cf_type == 'item_based':
		buis_base = review.map(lambda x: [x['business_id'], (x['user_id'],x['stars'])]).groupByKey().mapValues(create_list).collect()
		b_dict = {}
		for item in buis_base:
			key = foo0(item)
			if key in b_dict.keys():
				b_dict[key] = foo1(item)
			else:
				b_dict[key] = {}
				b_dict[key] = foo1(item)
		comb_value = review.map(lambda x: x['business_id']).distinct().filter(lambda x : boolfunc(b_dict, x))
		comb1 = comb_value.cartesian(comb_value).map(lambda x: sorted(x))
		model = comb1.map(lambda x: create_model(b_dict, x)).collect()
		tempdic = collections.defaultdict(int)
		with open(outputfile, 'w') as file:
			for item in model:
				if (item != None) and (item[2] > 0) and (tempdic[strcom1(item)] != 1):
					tempdic[strcom1(item)] = 1
					json.dump({'b1': item[0], 'b2': item[1], 'sim': item[2]}, file)
					file.write('\n')

if __name__ == '__main__':
	start_time = time.time()
	if len(sys.argv) <= 1:
		sys.exit()
	main(sys.argv)
	print("Duration: %s" % int(time.time() - start_time))
