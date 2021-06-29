# export PYSPARK_PYTHON=python3.6 
# spark-submit task3.py $ASNLIB/publicdata/review.json task3_default_ans default 20 50
# spark-submit task3.py $ASNLIB/publicdata/review.json task3_customized_ans customized 20 50

# export PYSPARK_PYTHON=/usr/local/bin/python3.6
# export PYSPARK_DRIVER_PYTHON=/usr/local/bin/python3.6

import sys
import json
from operator import add
from pyspark import SparkContext, SparkConf, StorageLevel

def main(argv):
	# Take arguments
	review = argv[1]
	output = argv[2]
	partition = argv[3]
	partn = int(argv[4])
	supn = int(argv[5])

	# Initializing Spark
	conf = SparkConf().setAppName('assignment1').setMaster('local[*]')
	sc = SparkContext().getOrCreate(conf)

	# Initial Json
	jsonStrs = sc.textFile(review)
	reviewsRDD = jsonStrs.map(lambda s : json.loads(s)).map(lambda x: (x['business_id'], 1))

	def hash_key(key):
		return hash(key)

	if partition == "customized":
		reviewsRDD = reviewsRDD.partitionBy(int(partn), hash_key)

	res = {'n_partitions': reviewsRDD.getNumPartitions(),\
	'n_items': reviewsRDD.glom().map(len).collect(),\
	'result': reviewsRDD.reduceByKey(add).filter(lambda x: x[1] > int(supn)).collect()}

	# Writing JSON data
	with open(output, 'w') as f:
		json.dump(res, f)

if __name__ == '__main__':
	if len(sys.argv) <= 1:
		sys.exit()
	main(sys.argv)