# export PYSPARK_PYTHON=python3.6 
# spark-submit task1.py $ASNLIB/publicdata/review.json task1_ans $ASNLIB/publicdata/stopwords 2018 10 10

# export PYSPARK_PYTHON=/usr/local/bin/python3.6
# export PYSPARK_DRIVER_PYTHON=/usr/local/bin/python3.6

import sys
import json
from operator import add
from pyspark import SparkContext, SparkConf, StorageLevel

def lowersplit(x):
	return x.lower().split(' ')

def main(argv):
	# Take arguments
	review = argv[1]
	output = argv[2]
	stopwords = argv[3]
	year = str(argv[4])
	topm = int(argv[5])
	topn = int(argv[6])

	# Initializing Spark
	conf = SparkConf().setAppName('assignment1').setMaster('local[*]')
	sc = SparkContext().getOrCreate(conf)

	# Initial Json
	jsonStrs = sc.textFile(review)
	reviewsRDD = jsonStrs.map(lambda s : json.loads(s))

	#A
	a = reviewsRDD.map(lambda x: x['review_id']).count()

	#B
	b = reviewsRDD.filter(lambda x: x['date'][:4] == year).count()

	#C
	c = reviewsRDD.map(lambda x: x['user_id']).distinct().count()

	#D
	d = reviewsRDD.map(lambda x: (x['user_id'], 1)).reduceByKey(add).persist(StorageLevel.DISK_ONLY).takeOrdered(topm, lambda y: (-y[1], y[0]))
	d = [list(d[i]) for i in range(topm)]

	#E
	eRDD = reviewsRDD.map(lambda x: (x['review_id'], x['text']))
	stopwordset = set()
	punc = set()
	punc.update(['(', '[', ',', '.', '!', '?', ':', ';', ']', ')',''])

	with open(stopwords, 'r') as f:
		for word in f.readlines():
			stopwordset.add(word[:-1])

	def func(word):
		if word not in stopwordset:
			return ''.join(c for c in word if c not in punc)

	e = eRDD.map(lambda x: x[1]).flatMap(lambda y: lowersplit(y)) \
	.map(lambda w: (func(w), 1)).filter(lambda x: (x[0] != None) and (x[0] != "")) \
	.reduceByKey(add).takeOrdered(topn, lambda x: (-x[1], x[0]))
	e = list(map(lambda x: x[0], e))

	# Writing JSON data
	res = {'A': a, 'B': b, 'C': c, 'D': d, 'E': e}
	with open(output, 'w') as f:
	    json.dump(res, f)

if __name__ == '__main__':
	if len(sys.argv) <= 1:
		sys.exit()
	main(sys.argv)