
# export PYSPARK_PYTHON=python3.6 

# spark-submit task1.py $ASNLIB/publicdata/train_review.json task1.res

# export PYSPARK_PYTHON=/usr/local/bin/python3.6
# export PYSPARK_DRIVER_PYTHON=/usr/local/bin/python3.6

import sys
import time
import json
import random
import itertools
from pyspark import SparkContext, SparkConf, StorageLevel

def item1(x):
	return (x[1])

def list1(x):
	return [x[1], x[0]]

def list2(x):
	return (list(x[1]))

def create_list(usr_dic, r_list):
	res = list()
	i = 0
	srl = set(r_list)
	srl = list(srl)
	while (i < len(srl)):
		res.append(usr_dic[srl[i]])
		i += 1
	return sorted(res)

def func_min_hash(a, b, m, value_list):
	ans = list()
	i = 0
	while (i < len(b)):
		cur_min = m + 1
		j = 0
		while (j < len(value_list)):
			cur_min= min(cur_min, (a[i]*value_list[j] + b[i])%m)
			j += 1
		ans.append(cur_min)
		i += 1
	return ans

def strcombine(x, i):
	return str(x[i-2])+","+str(x[i-1])

def split_list(value_list, size):
	bands = list()
	index = 0
	for i in range(size, len(value_list)+1, size):
		cur = str(index)+","
		cur += strcombine(value_list, i)
		bands.append(cur)
	return bands

def combination(value_list):
	value_list = sorted(value_list)
	if(len(value_list)==2):
		return [value_list]
	ans = list()
	i = 0
	while (i < len(value_list)):
		j = i + 1
		while (j < len(value_list)):
			ans.append(sorted([value_list[i], value_list[j]]))
			j += 1
		i += 1
	return ans

def bool1(x):
	return x[1] >= 0.05

def main(argv):
	# Take arguments
	inputfile = argv[1]
	outputfile = argv[2]

	# Initializing Spark
	conf = SparkConf().setAppName('assignment3').setMaster('local[*]')
	sc = SparkContext().getOrCreate(conf)
	RDD = sc.textFile(inputfile)
	review = RDD.map(json.loads).map(lambda x: [x['business_id'], x['user_id']])
	users_list = review.map(lambda x: item1(x)).sortBy(lambda x: x).distinct().collect()

	user_dict = {}
	index = 0
	print(len(users_list))
	while (index < len(users_list)):
		user_dict[users_list[index]] = index
		index += 1

	rated = review.groupByKey().mapValues(lambda x: create_list(user_dict, x)).sortByKey()
	a_list = random.sample([i for i in range(len(users_list))], 1100)
	b_list = random.sample([i for i in range(len(users_list))], 1100)
	min_hash = rated.mapValues(lambda x: func_min_hash(a_list,b_list, len(users_list), x))
	bands = min_hash.flatMapValues(lambda x: split_list(x, 2)).map(lambda x: list1(x))\
	.groupByKey().filter(lambda x: len(x[1]) >= 2).map(lambda x: list2(x))

	rate = rated.collect()
	rate_dict = {}
	i = 0
	while (i < len(rate)):
		rate_dict[rate[i][0]] = rate[i][1]
		i += 1

	# Fill into sets
	candidates = bands.flatMap(lambda x: itertools.combinations(x, 2)).distinct()\
	.map(lambda x: tuple([x, len(set(rate_dict[x[0]])&(set(rate_dict[x[1]])))/len(set(rate_dict[x[0]])\
	.union(set(rate_dict[x[1]])))])).filter(lambda x: bool1(x))\
	.map(lambda x: json.dumps({"b1": x[0][0], "b2": x[0][1], "sim": x[1]}))

	# Writing file
	with open(outputfile, 'w') as f:
		for item in candidates.collect():
			f.write(item)
			f.write('\n')

if __name__ == '__main__':
	start_time = time.time()
	if len(sys.argv) <= 1:
		sys.exit()
	main(sys.argv)
	print("Duration: %s" % int(time.time() - start_time))

