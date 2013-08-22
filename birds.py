import random
import os
import numpy as np
import matplotlib.image as mpimg
from sklearn.ensemble import RandomForestClassifier

nclasses = 19

def binencode(ls):
    ret = np.zeros(nclasses * len(ls))
    for i in range(len(ls)):
        for j in range(len(ls[i])):
            ret[i * nclasses + ls[i][j]] = 1

    return np.reshape(ret, (len(ls), nclasses)) 

def getloc(recid):
    locmap = {1:0, 2:1, 4:2, 5:3, 7:4, 8:5, 10:6, 11:7,
              13:8, 15:9, 16:10, 17:11, 18:12}
    key =  int(recmap[recid].split('_')[0].split('PC')[1])
    return locmap[key]
    
def getclustloc(recid):
    clustmap = {1:0, 2:0, 4:0, 5:0, 7:1, 8:2, 10:3, 11:3,
                13:3, 15:4, 16:4, 17:5, 18:5}
    key =  int(recmap[recid].split('_')[0].split('PC')[1])
    return clustmap[key]

def maketarget(labels):
    y = tuple([])
    for i in range(len(labels)):
        labs = labels[i]
        y = y + tuple([labs[1:len(labs)]])

    y = binencode(y)
    return y

def formatdat(labels):
    for i in range(len(labels)):
        recid = labels[i][0]
        line = map(float, hist[recid])
        line = line[1:len(line)] # Omit recid
        line = np.array(line)
        nsegs = len(line[line != 0])
        line = np.hstack((line, nsegs))

        if i == 0:
            x = line
        else:
            x = np.vstack((x, line))
        
    return x

def appendloc(x, labels, locprobs):
    for i in range(len(labels)):
        recid = labels[i][0]
        loc = getloc(recid) 
        ex = np.array([loc, getclustloc(recid)])
        ex = np.hstack((ex, locprobs[loc]))

        if i == 0:
            locmat = ex
        else:
            locmat = np.vstack((locmat, ex))

    if (len(x) == 0):
        return locmat

    return np.hstack((x, locmat))
    
def getstats(imgdat, recid):
    nstrips = 16 
    thresh = 35 
    stats = np.zeros(nstrips)
    striplen = len(imgdat) / nstrips 
    for i in range(nstrips): 
        start = i * striplen
        end = start + striplen 
        imgstrip = imgdat[start:end].copy()
        imgstrip = imgstrip[imgstrip > thresh]
        if len(imgstrip) != 0:
            stats[i] = np.mean(imgstrip)

    return stats 

def appendstats(x, labels):
    for i in range(len(labels)):
        recid = labels[i][0]
        imgfile = 'mlsp_contest_dataset2/supplemental_data/filtered_spectrograms/'
        imgfile = imgfile + recmap[recid].strip() + '.bmp'
        imgdat = mpimg.imread(imgfile)
        stats = getstats(imgdat, recid) 
                
        if i == 0:
            statsmat = stats 
        else:
            statsmat = np.vstack((statsmat, stats)) 

    return np.hstack((statsmat, x))

def getprior(locprobs, labels, y):
    loccounts = dict() 
    for i in range(len(labels)):
        recid = labels[i][0]
        loc = getloc(recid) 
        if locprobs.has_key(loc) == False:
            locprobs[loc] = y[i].copy() 
            loccounts[loc] = 1
        else:
            locprobs[loc] += y[i]
            loccounts[loc] += 1

    for key in locprobs.iterkeys():
        locprobs[key] /= loccounts[key]

def readlabels(filename, labels, istest=False):
    file = open(filename, 'r')
    index = 0
    line = file.readline()
    line = file.readline()
    while line != '':
        if istest == True: 
            labels[index] = [int(line.split(',')[0])]
        else:
            labels[index] = map(int, line.split(','))
            
        index += 1
        line = file.readline()
    file.close()

def loaddata():
    readlabels('trainlabels.txt', trainlabels)
    readlabels('testlabels.txt', testlabels, True)

    # Load the mapping between rec ids and filenames
    recmapfile = open('mlsp_contest_dataset2/essential_data/rec_id2filename.txt',
                      'r')
    line = recmapfile.readline()
    line = recmapfile.readline()
    while line != '':
        tokens = line.split(',')
        index = int(tokens[0])
        recmap[index] = tokens[1].strip() 
        line = recmapfile.readline()
    recmapfile.close()

    # Load histogram of segment features
    histfile = open('myhist.txt', 'r')
    line = histfile.readline()
    line = histfile.readline()
    index = 0
    while line != '':
        hist[index] = line.split(',')
        index += 1
        line = histfile.readline()
    histfile.close()


def main():
    random.seed(0)
    loaddata()

    # Train
    ytrain = maketarget(trainlabels)
    xtrain = formatdat(trainlabels)
    locprobs = dict()
    getprior(locprobs, trainlabels, ytrain)
    xtrain = appendstats(xtrain, trainlabels)
    xtrain = appendloc(xtrain, trainlabels, locprobs)

    classif = RandomForestClassifier(n_estimators=500, criterion='entropy',
                                     random_state=np.random.RandomState(0)) 
    classif.fit(xtrain, ytrain)

    # Test
    xtest = formatdat(testlabels)
    xtest = appendstats(xtest, testlabels)
    xtest = appendloc(xtest, testlabels, locprobs)

    pred = classif.predict_proba(xtest)

    probvals = np.zeros((xtest.shape[0], nclasses))
    for i in range(probvals.shape[0]):
        for j in range(nclasses):
            probvals[i, j] = pred[j][i][1] 

    # Make submission file
    probvals = np.around(probvals, 4)
    submfilename = 'subm.csv'
    submfile = open(submfilename, 'w')
    submfile.write('Id,probability\n')
        
    for i in range(probvals.shape[0]):
        for j in range(nclasses):
            strline = str(testlabels[i][0]*100+j) + ',' + str(probvals[i, j]) + '\n'
            submfile.write(strline) 

    submfile.close()
    print 'wrote', submfilename

if __name__ == '__main__':
    hist = dict()
    recmap = dict()
    trainlabels = dict()
    testlabels = dict()
    main()
