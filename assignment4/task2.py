
# export PYSPARK_PYTHON=python3.6 

# spark-submit task2.py 7 $ASNLIB/publicdata/ub_sample_data.csv betouput2.txt comoutput2.txt

# export PYSPARK_PYTHON=/usr/local/bin/python3.6
# export PYSPARK_DRIVER_PYTHON=/usr/local/bin/python3.6

import sys
import time
import json
import random
import collections
from operator import add
from copy import deepcopy
from functools import reduce
from itertools import combinations
from pyspark import SparkContext, SparkConf, StorageLevel, SQLContext

def minus(x, y):
	return x - y

def multi(x, y):
	return x * y

def divide(x, y):
	return x / y

class Node:
	def __init__(self, value, level):
		self.val = value
		self.level = level
		self.childrens = []
		self.short_path_num = 0

def getcredit(node, tree):
	if len(tree[node].childrens) == 0: 
		return 1
	sum_credit = 1
	i = 0
	templist = tree[node].childrens
	while (i < len(templist)):
		child = templist[i]
		child_credit = getcredit(child, tree) * divide(tree[node].short_path_num, tree[child].short_path_num)
		sum_credit += child_credit
		i += 1
	return sum_credit

def getbetweenness(cur_edges, node):
	q = [node]
	level = 0
	tree = {node: Node(node, level)}
	if True:
		tree[node].short_path_num = 1
	while True:
		if len(q) == 0:
			break
		cur = q.pop(0)
		i = 0
		templist = cur_edges[cur]
		while (i < len(templist)):
			n = templist[i]
			if n not in tree:
				q.append(n)
				child = Node(n, tree[cur].level + 1)
				child.short_path_num = tree[cur].short_path_num + 1
				tree[cur].childrens.append(n)
				tree[n] = child
			elif n in tree and tree[cur].level == minus(tree[n].level, 1):
				tree[n].short_path_num = tree[cur].short_path_num + 1
				tree[cur].childrens.append(n)
			i += 1
	ans = []
	for key, value in tree.items():
		for c in value.childrens:
			c_credit = getcredit(c, tree) * (value.short_path_num / tree[c].short_path_num)
			btw = [tuple(sorted([key, c])), c_credit]
			ans.append(btw)

	return ans

def createcommunity(x):
	commuities = list()
	users = set(x.keys())
	visited = set()
	while len(users) > 0:
		cur_node = users.pop()
		commuity = set()
		q = [cur_node]
		while len(q) > 0:
			prev = q.pop(0)
			commuity.add(prev)
			for n in x[prev]:
				if n not in visited: 
					q.append(n)
					commuity.add(n)
					visited.add(n)
		commuities.append(commuity)
		users = minus(users, commuity)
	return commuities

def getmod(cur_edges, m, edges, count_dict):
	commuities = createcommunity(cur_edges)
	G = 0
	index = 0
	while (index < len(commuities)):
		commuity = commuities[index]
		index += 1
		g = 0
		for i in commuity:
			for j in commuity:
				if i != j:
					ki = count_dict[i]
					kj =  count_dict[j]
					if i in edges[j]: Aij = 1
					else: Aij = 0
					g += (Aij - divide(multi(ki, kj), multi(2, m)))
		G += g
	return G / multi(2, m)

def boolfunc1(x, y):
	return x[2] >= y

def splitfunc(x):
	return x.split(',')

def lenpair(x):
	return [len(x), x[0]]

def listpair(x):
	return [x[1],x[0]]

def main(argv):
	# Take arguments
	threshold = int(argv[1])
	inputfile = argv[2]
	bet_outputfile = argv[3]
	com_outputfile = argv[4]

	# Initializing Spark
	conf = SparkConf().setAppName('assignment4').setMaster('local[*]')
	sc = SparkContext().getOrCreate(conf)
	sc.setLogLevel("WARN")
	dateRDD = sc.textFile(inputfile)

	data = dateRDD.map(lambda x: splitfunc(x)).filter(lambda x: x[0] != 'user_id')
	usr_pair = data.map(lambda x: listpair(x)).groupByKey().mapValues(sorted)\
	.mapValues(lambda x: combinations(x, 2)).flatMap(lambda x: x[1])\
	.map(lambda x: [x, 1]).reduceByKey(add).map(lambda x: [x[0][0],x[0][1], x[1]])\
	.filter(lambda x: boolfunc1(x, threshold))
	users = usr_pair.flatMap(lambda x: x[:2]).distinct()
	edges = collections.defaultdict(list)
	count = 0
	for item in usr_pair.collect():
		count += 1
		edges[item[0]].append(item[1])
		edges[item[1]].append(item[0])
	count_dict = {}
	for key, value in edges.items():
		count_dict[key] = len(value)

	betweenness = users.flatMap(lambda x: getbetweenness(edges, x)).reduceByKey(add)\
	.mapValues(lambda x: x / 2).sortByKey().sortBy(lambda x : -x[1])

	# Writing bet_outputfile file
	with open(bet_outputfile, 'w') as f1:
		for item in betweenness.collect():
			f1.write("(" + "'" + item[0][0] + "'" + ", " + "'" + \
				item[0][1] + "'" + ")" + ", " + str(item[1]) + "\n")

	cur_edges = deepcopy(edges)
	cur_G = -float("inf")
	max_edges = edges
	max_G = cur_G
	num = count
	while True:
		if num == 0:
			break
		betweenness = users.flatMap(lambda x: getbetweenness(cur_edges, x))\
		.reduceByKey(add).mapValues(lambda x: divide(x, 2)).sortByKey()\
		.sortBy(lambda x : -x[1])
		indexedge = betweenness.take(1)[0]
		cur_edges[indexedge[0][0]].remove(indexedge[0][1])
		cur_edges[indexedge[0][1]].remove(indexedge[0][0])

		cur_G = getmod(cur_edges, count, edges, count_dict)
		if cur_G > max_G:
			max_G = cur_G
			max_edges = deepcopy(cur_edges)
		num -= 1

	communities = createcommunity(max_edges)
	communities = map(lambda x: sorted(x), communities)
	communities = sorted(communities, key = lambda x: lenpair(x))

	# Writing com_outputfile file
	with open(com_outputfile, 'w') as f2:
		for item in communities:
			f2.write("'" + "', '".join(item) + "'" + "\n")

if __name__ == '__main__':
	start_time = time.time()
	if len(sys.argv) <= 1:
		sys.exit()
	main(sys.argv)
	print("Duration: %s" % int(time.time() - start_time))

