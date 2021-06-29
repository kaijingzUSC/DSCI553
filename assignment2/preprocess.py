
# export PYSPARK_PYTHON=python3.6 

import os
import csv
import json
from pyspark import SparkContext

def main(argv):
	# Initializing Spark
	conf = SparkConf().setAppName('assignment2').setMaster('local[*]')
	sc = SparkContext().getOrCreate(conf)

	review_rdd = sc.textFile('review.json').map(json.loads).map(lambda x: [x['business_id'], x['user_id']])
	business_rdd = sc.textFile('business.json').map(json.loads).map(lambda x: [x['business_id'], x['state']]).filter(lambda x: x[1] == 'NV')

	user_business = review_rdd.join(business_rdd).map(lambda x: x[1][0]+","+x[0]).distinct().map(lambda x: x.split(',')).sortByKey()

	with open("user_business.csv", "w") as f: 
		writer = csv.writer(f)
		writer.writerow(["user_id", "business_id"])
		writer.writerows(user_business.collect())

if __name__ == '__main__':
	main(sys.argv)

