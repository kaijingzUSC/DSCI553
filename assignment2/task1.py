
# export PYSPARK_PYTHON=python3.6 
# spark-submit task1.py 1 5 $ASNLIB/publicdata/small1.csv case1_sup5
# spark-submit task1.py 1 4 $ASNLIB/publicdata/small1.csv case1_sup4
# spark-submit task1.py 2 9 $ASNLIB/publicdata/small1.csv case2_sup9

# spark-submit task1.py 1 4 $ASNLIB/publicdata/small2.csv case1_sup4
# spark-submit task1.py 2 9 $ASNLIB/publicdata/small2.csv case2_sup9

# export PYSPARK_PYTHON=/usr/local/bin/python3.6
# export PYSPARK_DRIVER_PYTHON=/usr/local/bin/python3.6

import sys
import time
import itertools
from operator import add
from pyspark import SparkContext, SparkConf, StorageLevel

def split1(x):
	return x.split(',')

def bool1(x, y):
	return x!=y

def tuple1(x):
	return tuple(x)

def pair1(x):
	return (1,x[1])

def pair2(x):
	return (x[1],x[0])

def pair3(x):
	return (x[0])

def pair4(x):
	return (x[1])

def pair5(x):
	return (len(x), x)

def frequent(baskets, distinctItems, support, allnum):
	candies = []
	i = -1
	while(i < (len(distinctItems)-1)):
		i += 1
		candies.append(tuple([distinctItems[i]]))

	freqSub = {}
	sub_s = int(support * len(baskets) / allnum)
	k = 0
	while candies:
		k += 1
		tempcount = {}
		i = -1
		while(i < (len(baskets)-1)):
			i += 1
			basket = baskets[i]
			if len(basket) >= k:
				j = -1
				while(j < (len(candies)-1)):
					j += 1
					candy = candies[j]
					if set(candy).issubset(set(basket)):
						try: tempcount[candy] += 1
						except: tempcount[candy] = 1
		temp = []
		for items, count in tempcount.items():
			if count >= sub_s:
				temp.append(items)
		freqSub[k] = sorted(temp)

		candies = []
		for index, x in enumerate(freqSub[k][:-1]):
			for y in freqSub[k][index+1:]:
				if x[:-1] == y[:-1]:
					candies.append(tuple(sorted(list(set(x) | set(y)))))
				if x[:-1] != y[:-1]:
					break

	allsubFreq = []
	for key in freqSub.keys():
		allsubFreq.extend(freqSub[key])

	yield allsubFreq

def fullcount(basket, candidates):
	result = []
	i = -1
	while(i < (len(candidates)-1)):
		i += 1
		candy = candidates[i]
		if set(candy).issubset(set(basket)):
			result.extend([(tuple(candy), 1)])
	yield result

def main(argv):
	# Take arguments
	case_num = int(argv[1])
	support = int(argv[2])
	inputfile = argv[3]
	outputfile = argv[4]

	# Initializing Spark
	conf = SparkConf().setAppName('assignment2').setMaster('local[*]')
	sc = SparkContext().getOrCreate(conf)
	csvRDD = sc.textFile(inputfile, 2)
	header = csvRDD.first()

	if case_num == 1: #group by users
		inputRDD = csvRDD.filter(lambda line: bool1(line, header)).map(lambda line: split1(line)).map(lambda t:tuple1(t))\
		.groupByKey().mapValues(set).mapValues(sorted).mapValues(tuple).map(lambda t: pair1(t)).persist(StorageLevel.DISK_ONLY)
	elif case_num == 2: #group by business_ids
		inputRDD = csvRDD.filter(lambda line: bool1(line, header)).map(lambda line: split1(line)).map(lambda t:pair2(t))\
		.groupByKey().mapValues(set).mapValues(sorted).mapValues(tuple).map(lambda t: pair1(t)).persist(StorageLevel.DISK_ONLY)

	distinctItems = inputRDD.flatMapValues(tuple).map(lambda t : pair2(t)).groupByKey().map(lambda t : pair3(t)).collect()
	distinctItems.sort()

	whole_number = inputRDD.count()
	baskets = inputRDD.map(lambda t: pair4(t)).persist(StorageLevel.DISK_ONLY)

	freqsample = baskets.mapPartitions(lambda part: frequent(list(part), distinctItems, support, whole_number))\
	.flatMap(lambda x: x).distinct().sortBy(lambda t: pair5(t)).collect()

	freqi = baskets.flatMap(lambda basket: fullcount(basket, freqsample)).flatMap(lambda x: x).reduceByKey(add)\
	.filter(lambda items: pair4(items) >= support).map(lambda items: pair3(items)).sortBy(lambda t: pair5(t)).collect()

	# Fill into sets
	candidates_set = {}
	for key, group in itertools.groupby(freqsample, lambda items: len(items)):
		candidates_set[key] = sorted(list(group), key=lambda x : x) 
	frequent_set = {}
	for key, group in itertools.groupby(freqi, lambda items: len(items)):
		frequent_set[key] = sorted(list(group), key=lambda x : x)
	l = len(frequent_set.keys())

	# Writing file
	with open(outputfile, 'w') as f:
		f.write('Candidates:\n')
		output = ''
		for key in candidates_set.keys():
			output += str([list(x) for x in candidates_set[key]])[1:-1].replace('[', '(').replace(' (', '(').replace(']', ')')+'\n\n'
		f.write(output)

		f.write('Frequent Itemsets:\n')
		output = ''
		i = 1
		for key in frequent_set.keys():
			if i < l:
				output += str([list(x) for x in frequent_set[key]])[1:-1].replace('[', '(').replace(' (', '(').replace(']', ')')+'\n\n'
			else:
				output += str([list(x) for x in frequent_set[key]])[1:-1].replace('[', '(').replace(' (', '(').replace(']', ')')
			i += 1
		f.write(output)

if __name__ == '__main__':
	start_time = time.time()
	if len(sys.argv) <= 1:
		sys.exit()
	main(sys.argv)
	print("Duration: %s" % int(time.time() - start_time))

