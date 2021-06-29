import sys
from pyspark import SparkContext, SparkConf, StorageLevel
import itertools
from operator import add
import time

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

def Size3Candidate(preFreq, k): #preFreq> list of lists    
    candidates = []
    for index, f1 in enumerate(preFreq[:-1]):
        for f2 in preFreq[index+1:]:
            if f1[:-1] == f2[:-1]:
                comb = tuple(sorted(list(set(f1) | set(f2))))
                subs = [sub for sub in itertools.combinations(comb, k-1)]
                if set(subs).issubset(set(preFreq)):
                    candidates.append(comb)  
            if f1[:-1] != f2[:-1]:
                break             
    return candidates


def frequent(baskets, distinctItems, support, numofwhole):
    sub_s = support * len(baskets) / numofwhole
    
    freqSub = {}
    candies = []
    i = -1
    while(i < (len(distinctItems)-1)):
        i += 1
        candies.append(tuple([distinctItems[i]]))

    k=0

    while candies:
        k+=1
        tempcount = {}
        i = -1
        while(i < (len(baskets)-1)):
            i += 1
            basket = baskets[i]

            if k==1: #count singles
                for single in itertools.combinations(basket, 1):
                    try: tempcount[single] +=1
                    except: tempcount[single] =1
            if k == 2: #count pairs
                thinbasket = sorted(list(set(basket) & set(flatSingle))) #only frequent single can produce frequent pair
                for pair in itertools.combinations(thinbasket, 2):
                    try: tempcount[pair] +=1
                    except: tempcount[pair] =1
            
            if k > 2: #count triples..... where candidates number is much less than if we iterate through subset of basket
                if len(basket)>=k:
                    for candy in candies:
                        if set(candy).issubset(set(basket)):
                            try: tempcount[candy] +=1
                            except: tempcount[candy] =1


        temp = []
        for items, count in tempcount.items():
            if count >= sub_s:
                temp.append(items)
        freqSub[k] = sorted(temp)

        if k == 1: 
            flatSingle = [single[0] for single in freqSub[k]]
            candies = [pair for pair in itertools.combinations(flatSingle, 2)]         
        if k > 1 : candies = Size3Candidate(freqSub[k], k+1)
                
    allsubFreq = []
    for _, candy in freqSub.items():
        allsubFreq.extend(candy)
    
    yield allsubFreq


def countOnWhole(basket, candidates):
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
    threshold = int(argv[1])
    support = int(argv[2])
    inputfile = argv[3]
    outputfile = argv[4]

    # Initializing Spark
    conf = SparkConf().setAppName('assignment2').setMaster('local[*]')
    sc = SparkContext().getOrCreate(conf)
    csvRDD = sc.textFile(inputfile)
    header = csvRDD.first()

    inputRDD = csvRDD.filter(lambda line: bool1(line, header)).map(lambda line: split1(line)).map(lambda t:tuple1(t))\
    .groupByKey().mapValues(set).mapValues(sorted).mapValues(tuple)

    inputRDDfilter = inputRDD.filter(lambda t:len(t[1]) > threshold).map(lambda t: pair1(t)).persist(StorageLevel.DISK_ONLY)
    
    #count distinct items
    distinctItems = inputRDDfilter.flatMapValues(tuple)\
    .map(lambda t : pair2(t)).groupByKey().map(lambda t : pair3(t)).collect()
    distinctItems.sort()

    #whole baskets number
    whole_number = inputRDDfilter.count()

    baskets = inputRDDfilter.map(lambda t: pair4(t)).persist(StorageLevel.DISK_ONLY)

    #first phase > find candidate in subsets
    FreqinSample = baskets.mapPartitions(lambda part:frequent(list(part), distinctItems, support, whole_number))\
    .flatMap(lambda x: x).distinct().sortBy(lambda t: (len(t), t)).collect()


    #2nd phase > count on whole
    frequentI = baskets.flatMap(lambda basket: countOnWhole(basket, FreqinSample)).flatMap(lambda x: x).reduceByKey(add)\
    .filter(lambda items: items[1] >= support).map(lambda items:items[0]).sortBy(lambda t: pair5(t)).collect()

    candidates_set = {}
    for key, group in itertools.groupby(FreqinSample, lambda items: len(items)):
        candidates_set[key] = sorted(list(group), key=lambda x : x) 
    frequent_set = {}
    for key, group in itertools.groupby(frequentI, lambda items: len(items)):
        frequent_set[key] = sorted(list(group), key=lambda x : x)

    # Writing file
    with open(outputfile, 'w') as f:
        f.write('Candidates:\n')
        output = ''
        for key in candidates_set.keys():
            output += str([list(x) for x in candidates_set[key]])[1:-1].replace('[', '(').replace(' (', '(').replace(']', ')')+'\n\n'
        f.write(output)

        f.write('Frequent Itemsets:\n')
        output = ''
        for key in frequent_set.keys():
            output += str([list(x) for x in frequent_set[key]])[1:-1].replace('[', '(').replace(' (', '(').replace(']', ')')+'\n\n'
        f.write(output)

if __name__ == '__main__':
    if len(sys.argv) <= 1:
        sys.exit()
    main(sys.argv)




