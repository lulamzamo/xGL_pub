# coding=UTF-8
'''
    Any-language Graphical Lemmatiser -xGL
    Copyright (C) 2025  Lulamile Mzamo - lula[underscore]mzamo[at]yahoo.co.uk

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU AFFERO GENERAL PUBLIC LICENSE as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.


    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <http://www.gnu.org/licenses/>.
'''


import math
from sys import platform
from math import inf, log2
import cProfile, pstats, io
from collections import Counter
from numba import jit, cuda

from heapq import merge
from random import random

from _warnings import warn
# from cSegUtil import boundary

class dList(list):
    # def __init__(self, defValue=None):
    #     self.defValue = defValue
    #

    def __init__(self, iterable=[], defValue=None):
        self.defValue = defValue
        super().__init__(iterable)

        
    def __setitem__(self, index, item):
        try:
            super().__setitem__(index, item)
        except IndexError:
            lena = len(self) #Length of aLst
            if lena < index+1:
                super().extend([self.defValue.copy() for _ in range(index-lena+1)] if type(self.defValue)==list else [self.defValue]*(index-lena+1) )
            super().__setitem__(index, item)
             
    def __getitem__(self, __i):
        try:
            return list.__getitem__(self, __i)
        except IndexError:
            warn("Index Not Found - __getitem__ Returning default value", RuntimeWarning)
            return self.defValue

def generateRandomDirichlet(size):
    rd = [random() for _ in range(size)]
    return [rdi/sum(rd) for rdi in rd]
        




def cprofile(fnc):
    
    """A decorator that uses cProfile to profile a function"""
    
    def inner(*args, **kwargs):
        
        pr = cProfile.Profile()
        pr.enable()
        retval = fnc(*args, **kwargs)
        pr.disable()
        s = io.StringIO()
        sortby = 'cumulative'
        ps = pstats.Stats(pr, stream=s).sort_stats(sortby)
        ps.print_stats()
        print(s.getvalue())
        return retval

    return inner

def gramLen(sText,separator=''):
    return 0 if sText == '' else len(sText) if separator == '' else sText.count(separator)+1

# @jit(target_backend=cuda)
def genNGrams(sWord,minn=1,maxn=inf,separator=''):
#         nGrams = []
    K = gramLen(sWord,separator)
    if K < minn: return []
    
    if separator == '':
        
    #         lNGrams = [[sWord[j:i] for j in range(0 if maxn==inf or i-maxn < 0  else i-maxn,i-minn+1)] for i in range(minn,K+1)]
        lNGrams = [sWord[j:i] for i in range(minn,K+1) for j in range(0 if maxn==inf or i-maxn < 0  else i-maxn,i-minn+1)]
    #         for gram in lNGrams:
#             nGrams += gram
        
        return lNGrams
    else:
        aLocs = (-1,)+(i for i in  filter(lambda x: x >-1 , [i if sWord[i] == separator else -1 for i in range(0,len(sWord))])) + (len(sWord),)
        lNGrams = ()# [sWord[aLocs[j]:aLocs[i]] for i in range(minn,K+1) for j in range(0 if maxn==inf or i-maxn < 0  else i-maxn,i-minn+1)] 
        for i in range(minn,K+1):
            for j in range(0 if maxn==inf or i-maxn < 0  else i-maxn,i-minn+1):
                lNGrams+sWord[aLocs[j]+1:aLocs[i]]
#                 print(lNGrams)
        return lNGrams
 

# @jit(target_backend=cuda)   
def gGenNGrams(sWord,minn=1,maxn=inf,separator=''):
#         nGrams = []
    # test https://stackoverflow.com/questions/21883108/fast-optimize-n-gram-implementations-in-python
    K = gramLen(sWord,separator)
    if K < minn: return []
    
    if separator == '':
        
    #         lNGrams = [[sWord[j:i] for j in range(0 if maxn==inf or i-maxn < 0  else i-maxn,i-minn+1)] for i in range(minn,K+1)]
        lNGrams = (sWord[j:i] for i in range(minn,K+1) for j in range(0 if maxn==inf or i-maxn < 0  else i-maxn,i-minn+1)) 
    #         for gram in lNGrams:
#             nGrams += gram
        
        return lNGrams
    else:
        aLocs = [-1,]+[i for i in  filter(lambda x: x >-1 , [i if sWord[i] == separator else -1 for i in range(0,len(sWord))])] + [len(sWord),]
#         lNGrams = []# [sWord[aLocs[j]:aLocs[i]] for i in range(minn,K+1) for j in range(0 if maxn==inf or i-maxn < 0  else i-maxn,i-minn+1)] 
        lNGrams = (sWord[aLocs[j]+1:aLocs[i]] for i in range(minn,K+1) for j in range(0 if maxn==inf or i-maxn < 0  else i-maxn,i-minn+1))
#                 lNGrams.append()
#                 print(lNGrams)
        return lNGrams



def gramTrunc(sText,iLength,separator='',reverse=False):
    'Truncates a string to the desired length'
    if sText == '' or iLength == inf or gramLen(sText,separator) <= iLength:
        return sText
    elif separator == '':
        lenT = len(sText)
        sText = sText[lenT-iLength:] if reverse else sText[:iLength]
        return sText
    else:
        lenT = sText.count(separator) + 1 
        arh = sText.split(separator)
        sText = separator.join(arh[lenT-iLength:]) if reverse else separator.join(arh[:iLength ]) 
        return sText


@jit(target_backend='cuda')
def affixList(sAffix,separator='',reverse=False, minn=1, maxn=inf):
#         ret = {sAffix} if maxn==inf or  else set()
#         ndif = inf if maxn==inf else maxn-minn
        if separator == '':
            lens = len(sAffix)
            ret = {sAffix} if maxn==inf or lens < maxn   else set()
            ret.update([sAffix[:loc+1] for loc in range(minn-1,lens if maxn==inf else maxn+1)] if reverse else [sAffix[loc:] for loc in range(0 if maxn==inf or lens < maxn else lens-maxn,lens-minn+1)]) #0 if maxn==inf or loc-ndif < 1 else loc-ndif
            return ret
        else:
            aLocs = [i for i in  filter(lambda x: x >-1 , [i if sAffix[i] == separator else -1 for i in range(0,len(sAffix))])]
            lens = len(aLocs)
            ret = {sAffix} if maxn==inf or lens < maxn   else set()
            ret.update([sAffix[:loc] for loc in aLocs[minn-1:lens if maxn==inf else maxn]] if reverse else [sAffix[loc if loc ==0 else loc+1:] for loc in aLocs[0 if maxn==inf or lens<maxn  else lens-maxn:lens-minn+1]])
            return ret
        

        
        



def readTrainingFile(sTrainFile):
    inFile = open(sTrainFile, "r")
    TrainSet= []
    for line in inFile:
        if line.replace("\n","") != '' and not line.endswith(",\n"): #Ignore empty lines, full stops and blank lemmas
            if line.count(",") > 1:
                ar = line.replace("\n","").split(",")
                line = "\t".join([",".join(ar[:len(ar)-1]),ar[len(ar)-1]]).replace("\n","")
            else:
                TrainSet.append(line.replace(",","\t").replace("\n",""))
    inFile.close()
    return TrainSet

def boundaries(seg):
    i = 0
    bound = set()
    for c in seg:
        if c == '-':
            bound.add(i)
        else:
            i += 1
    return bound


# @jit(target_backend='cuda')
def boundary(seg):
    i = 0
    bound = []
    for c in seg:
        if c == '-':
            bound[i-1] = '1'
        else:
            bound.append('0')
            i += 1
    bound.pop()
    return "".join(bound)


def SegMatchCount(seg1,seg2):
    b1 = boundary(seg1)
    b2 = boundary(seg2)
    
    return sum(b1[i] == b2[i] for i in range(len(b2)))



def tp(s1,s2): #True Positive, #s1 in relation to s2
    
    b1 = boundary(s1)
    b2 = boundary(s2)
    
    return sum(b1[i] =='1' == b2[i] for i in range(len(b2)))



def fp(s1,s2): # False Positive, s1 in relation to s2
    b1 = boundary(s1)
    b2 = boundary(s2)
    
    return sum(b1[i]=='1' and b2[i]=='0' for i in range(len(b2)))



def tn(s1,s2): #True Negative, s1 in relation to s2
    b1 = boundary(s1)
    b2 = boundary(s2)
    
    return sum(b1[i] =='0' == b2[i] for i in range(len(b2)))



def fn(s1,s2): #False Negative, s1 in relation to s2
    b1 = boundary(s1)
    b2 = boundary(s2)
    
    return sum(b1[i]=='0' and b2[i]=='1' for i in range(len(b2)))


    



def sMetrics(s1,s2):
    res =  [0,0,0,0] # {'tp':0,'fp':0,'tn':0,'fn':0}
    b1 = boundary(s1)
    b2 = boundary(s2)
    
    for i in range(len(b2)):
        if b1[i]=='1' == b2[i]:
            res[0] += 1
        elif b1[i]=='1' and b2[i]=='0':
            res[1] += 1
        elif b1[i]=='0' == b2[i]:
            res[2] += 1
        elif b1[i]=='0' and b2[i]=='1' :
            res[3] += 1
        else:
            raise ValueError
    return res

def sAccuracy(s1,s2):
    metrics = sMetrics(s1,s2)
    return 1.*(metrics[0] + metrics[2])/sum(metrics)

   
def atp(s1,s2): #s1 in relation to s2
    
    return sum(tp(s1[i],s2[i]) for i in range(len(s2)))

def afp(s1,s2): #s1 in relation to s2
    
    return sum(fp(s1[i],s2[i]) for i in range(len(s2)))

def atn(s1,s2): #s1 in relation to s2
    
    return sum(tn(s1[i],s2[i]) for i in range(len(s2)))

def afn(s1,s2): #s1 in relation to s2

    return sum(fn(s1[i],s2[i]) for i in range(len(s2)))



def aMetrics(s1,s2):
#     res = [0,0,0,0] # {'tp':0,'fp':0,'tn':0,'fn':0}
    
    metrics = [sMetrics(s1[i],s2[i]) for i in range(len(s2))]
#     for i in range(len(s2)):
#         smetrics = sMetrics(s1[i],s2[i]) 
#         for j in [0,1,2,3]:
#             res[j] += smetrics[j]
    res = [x for x in map(sum,zip(*metrics))]        
    return res



def Accuracy(s1,s2=None):
    metrics = s1 if s2 == None else aMetrics(s1,s2) if type(s2) == list else sMetrics(s1, s2)
    print(metrics)
    return 0 if metrics == [] else 1.*(metrics[0] + metrics[2])/sum(metrics)

  
def Precision(s1,s2=None):
    metrics = s1 if s2 == None else aMetrics(s1,s2) if type(s2) == list else sMetrics(s1, s2)
    return 0 if metrics==[] or metrics[0]+metrics[1] ==0 else 1.*metrics[0]/(metrics[0]+metrics[1])

def Recall(s1,s2=None):
    metrics = s1 if s2 == None else aMetrics(s1,s2) if type(s2) == list else sMetrics(s1, s2)
    return 0 if metrics==[] or metrics[0]+metrics[3] ==0 else 1.*metrics[0]/(metrics[0]+metrics[3])

def f1Score(s1,s2=None):
    metrics = s1 if s2 == None else aMetrics(s1,s2) if type(s2) == list else sMetrics(s1, s2)
    precision = Precision(metrics)
    recall = Recall(metrics)
    return 0 if precision + recall == 0 else  2.*precision*recall/(precision + recall)

def getEnv():
    #Method for specifying locations etc based on platform
    if 'win32' in platform:
        fWorkLoc = "C:/Users/Lula/Documents/Working/" # Location to work with
        fDataLoc = "C:/Users/Lula/Dropbox/Personal/1.Ph.D/Evaluation/Data/" # Location to work with
        fExecLoc = "C:/Users/Lula/Documents/Working/".replace('/','\\')
        fResultsLoc = "C:/Users/Lula/Dropbox/Personal/1.Ph.D/Evaluation/Results/"
        resScript = "../scripts\timemem python"
        delScript = 'del /Q '
    elif 'linux' in platform:
        fWorkLoc = "/home/jupyter/Documents/Working/" # Location to work with
        fDataLoc = "/home/jupyter/Dropbox/Personal/1.Ph.D/Evaluation/Data/" # Location to work with
        fExecLoc = "/home/jupyter/Documents/Working/" #.replace('/','\\')
        fResultsLoc = "/home/jupyter/Dropbox/Personal/1.Ph.D/Evaluation/Results/"
        resScript = "/usr/bin/time -f '%M %S %U' ~/Jupyter/notebook/bin/python"
        delScript = 'rm -f '
        
    return fWorkLoc, fDataLoc, fExecLoc, fResultsLoc, resScript, delScript

def extResUsage(sRes):
    # Method for extracting resource usage statistics based on platform
    f = sRes.fields()
    if 'win32' in platform:
        lenf = len(f)
        duration = float(f[lenf-8][3])# + float(f[lenf-6][3]) # in seconds
        memory = float(f[lenf-4][3]) #in KB
    elif 'linux' in platform:
        memory = float(f[-1][0])
        duration = float(f[-1][1]) + float(f[-1][2])
    return duration,memory
    
def extResXMH(sRes):
    f = sRes.fields()
    lenf = len(f)
    duration = float(f[lenf-8][3])# + float(f[lenf-6][3]) # in seconds
    memory = float(f[lenf-4][3]) #in KB
    
    return duration,memory

def getResults(fSegResults, fTestResults): # fTestWords,  , fTrainSample
    
#     
#     Res['lemClass'] = [genTrans(resTokens[x],resLemLemma[x]) for x in range(len(resTokens))]
#     Res['testClass'] = [genTrans(resTokens[x],resTestLemma[x]) for x in range(len(resTokens))]
#     ResLemClass = [genTrans(resTokens[x],resLemLemma[x]) for x in range(len(resTokens))]
#     ResTestClass = [genTrans(resTokens[x],resTestLemma[x]) for x in range(len(resTokens))]
#     print Res
    
#     fSeg =  open(fSegResults, "r")
#     aSeg = [x.strip().replace(' ','-')  for x in open(fSegResults, "r").readlines()] if type(fSegResults) == str else fSegResults
# #     .strip().split('\n')
# #     fGold = open(fTestResults, "r")
#     aGold = [x.strip().replace(' ','-')  for x in open(fTestResults, "r").readlines()]  if type(fTestResults) == str else fTestResults
#     
    aSeg =  open(fSegResults, "r").readlines() if type(fSegResults) == str else fSegResults
    aSeg = [x.strip().replace(' ','-') for x in aSeg]
#     print(aSeg)
#     .strip().split('\n')
#     fGold = open(fTestResults, "r")
    aGold = open(fTestResults, "r").readlines()  if type(fTestResults) == str else fTestResults
    aGold = [x.strip().replace(' ','-') for x in aGold]
    
    metrics = aMetrics(aSeg, aGold)
    
    #FScore = f1_score(ResTestClass,ResLemClass,average='weighted')
#     print(metrics)
    
    Results = [Accuracy(metrics), Precision(metrics), Recall(metrics), f1Score(metrics)]
#     y_true =  list("".join([boundary(x) for x in aGold]))
#     y_pred =    list( "".join([boundary(x) for x in  aSeg ]))
#     accuracy = accuracy_score(y_true, y_pred)
#     precision, recall, fscore, support  = precision_recall_fscore_support(y_true, y_pred, pos_label='1', average='binary')
#         
    
    #2.2 Calculate Actual Lemma Classes
    #return blindAcc, unknownAcc, knownAcc, FScore
    return Results

def getResultSplits(fSegResults, fTestResults, fTrainSample):#,  fTestWords)
    aSeg =  open(fSegResults, "r").readlines() if type(fSegResults) == str else fSegResults
    aSeg = [x.strip().replace(' ','-') for x in aSeg]
    aSeg = [x.strip().replace('--','-') for x in aSeg]
    # print(aSeg[0])
    aGold = open(fTestResults, "r").readlines()  if type(fTestResults) == str else fTestResults
    aGold = [x.strip().replace(' ','-') for x in aGold]
    aGold = [x.strip().replace('--','-') for x in aGold]
    # print(aGold[0])
    aTrain = open(fTrainSample, "r").readlines()  if type(fTrainSample) == str else fTrainSample
    aTrain = set([x.strip() for x in aTrain])
    aKnownSeg = []
    aKnownGold = []
    aOoVSeg = []
    aOoVGold = []
    for i in range(len(aSeg)):
        if aSeg[i].replace("-","") in aTrain:
            aKnownSeg.append(aSeg[i])
            aKnownGold.append(aGold[i])
        else:
            aOoVSeg.append(aSeg[i])
            aOoVGold.append(aGold[i])
            
    knownMetrics = aMetrics(aKnownSeg, aKnownGold)
    OoVMetrics = aMetrics(aOoVSeg, aOoVGold)
    
    
    
    return len(aSeg), len(aKnownSeg), len(aOoVSeg), Accuracy(knownMetrics), Precision(knownMetrics), Recall(knownMetrics), f1Score(knownMetrics), Accuracy(OoVMetrics), Precision(OoVMetrics), Recall(OoVMetrics), f1Score(OoVMetrics)
    
def getSplitsDLPerp(fSegResults, fTestResults, fTrainSample):
    DL = getDL(fSegResults)
    Perp = getPerplexity(fSegResults) 
    return  getResultSplits(fSegResults, fTestResults, fTrainSample) + (DL, Perp) 
                 
def getMorfResults(fSegResults, fTestResults): # fTestWords,  , fTrainSample
    
#     
#     Res['lemClass'] = [genTrans(resTokens[x],resLemLemma[x]) for x in range(len(resTokens))]
#     Res['testClass'] = [genTrans(resTokens[x],resTestLemma[x]) for x in range(len(resTokens))]
#     ResLemClass = [genTrans(resTokens[x],resLemLemma[x]) for x in range(len(resTokens))]
#     ResTestClass = [genTrans(resTokens[x],resTestLemma[x]) for x in range(len(resTokens))]
#     print Res
    
#     fSeg =  open(fSegResults, "r")
    aSeg =  open(fSegResults, "r").readlines() if type(fSegResults) == str else fSegResults
    aSeg = [x.strip().replace(' ','-') for x in aSeg]
#     print(aSeg)
#     .strip().split('\n')
#     fGold = open(fTestResults, "r")
    aGold = open(fTestResults, "r").readlines()  if type(fTestResults) == str else fTestResults
    aGold = [x.strip().replace(' ','-') for x in aGold]
#     print(aGold)
    
    metrics = aMetrics(aSeg, aGold)
    
    #FScore = f1_score(ResTestClass,ResLemClass,average='weighted')
    print(metrics)
    
    Results = [Accuracy(metrics), Precision(metrics), Recall(metrics), f1Score(metrics)]
    
        
    
    #2.2 Calculate Actual Lemma Classes
    #return blindAcc, unknownAcc, knownAcc, FScore
    return Results


def getDist(m):
    '''
    m    : mapping of events to counts
    
    Return: distribution of events to probabilities based on the counts
    
    '''
    T = sum(m.values())
    pDist = m if T == 1 else Counter({seg:count/T for seg, count in m.items()})    
    return pDist

   
def getEntropy(fSegs, pDist=None, ):#,  fTestWords)
    '''
    fSegs : Either a filename string pointing to the segmentations or a list of segmentations
    pDist : A discrete probability distribution that maps strings that is then used to calculate the entropy
    
    Returns Entropy of fSeg given the distribution pDist
    '''
    
    morphCounts = Counter()
    if type(fSegs) == str:
        with open(fSegs, "r") as flSegs:
            fSegs = flSegs.readlines()
    
    wordCounts = Counter(fSegs)
    
#     with (open(fSegs, "r") if type(fSegs) == str else fSegs) as aSegs:
        
    for seg, count in wordCounts.items():
        morphCounts.update({morph:count*mCount for morph, mCount in Counter(seg.strip().replace(' ','-').replace('--','-').split('-')).items()})
    
    if not pDist:
        pDist = getDist(morphCounts)
            
    H = -sum(pDist[seg]*log2(pDist[seg]) for seg, count in morphCounts.items()   )

    return H

 
def getPerplexity(fSegs, pDist=None, ):#,  fTestWords)
    '''
    fSegs : Either a filename string pointing to the segmentations or a list of segmentations
    pDist : A discrete probability distribution that maps strings that is then used to calculate the entropy
    
    Returns Perplexity of fSeg given the distribution pDist
    '''
    
    morphCounts = Counter()
    if type(fSegs) == str:
        with open(fSegs, "r") as flSegs:
            fSegs = flSegs.readlines()
    
    wordCounts = Counter(fSegs)
    
#     with (open(fSegs, "r") if type(fSegs) == str else fSegs) as aSegs:
        
    for seg, count in wordCounts.items():
        morphCounts.update({morph:count*mCount for morph, mCount in Counter(seg.strip().replace(' ','-').replace('--','-').split('-')).items()})
    
    if not pDist:
        pDist = getDist(morphCounts)
            
    l = sum(log2(pDist[seg]) for seg, count in morphCounts.items())/len(morphCounts)

    return 2**(-l)


def getDLKit98(fSegs):
    morphCounts = Counter()
    if type(fSegs) == str:
        with open(fSegs, "r") as flSegs:
            fSegs = flSegs.readlines()
    wordCounts = Counter(fSegs)           
    for seg, count in wordCounts.items():
        morphCounts.update({morph:count*mCount for morph, mCount in Counter(seg.strip().replace(' ','-').replace('--','-').split('-')).items()})
    Total = sum(morphCounts.values())
    DL = -sum(count*log2(count/Total) for count in morphCounts.values()   ) # L(D|M)
  
    return DL

  
def getDL(fSegs, pDist=None, M=False ):#,  fTestWords)
    ''':
    fSegs : Either a filename string pointing to the segmentations or a list of segmentations
    pDist : A discrete probability distribution that maps strings to probabilities that is then used to calculate the entropy
    M     : M is a flag that informs the function whether the calculation is be for the description length of the data and the model (False) or the model alone (True)
    
    Returns Description length (DL) of fSeg given the distribution pDist based on Zhikhov's paper (Zhikov et al, 2013, An Efficient Algorithm for Unsupervised Word Segmentation with Branching Entropy and MDL)
    '''
    
    morphCounts = Counter()
    if type(fSegs) == str:
        with open(fSegs, "r") as flSegs:
            fSegs = flSegs.readlines()
    wordCounts = Counter(fSegs)           
    for seg, count in wordCounts.items():
        morphCounts.update({morph:count*mCount for morph, mCount in Counter(seg.strip().replace(' ','-').replace('--','-').split('-')).items()})
    
    if not pDist:
        pDist = getDist(morphCounts)
            
    L_DM = -sum(count*log2(pDist[seg]) for seg, count in morphCounts.items()   ) # L(D|M) Description length of the data alone given the model pDist
#     return L_DM
#     print(M)
    L_M = len("".join(morphCounts)) if M else getDL(Counter(tuple("".join(morphCounts))),None,True)#  L(M), Description Length of the Model
    L_ThetaM = (len(morphCounts)-1)/2*log2( sum(len(seg)*count for seg, count in morphCounts.items()  ) ) #L(Thetha|M) = (|M|-1)/2 *log2(len(D)) Complexity term defined in (Li, 1998)
    DL = L_M + L_DM + L_ThetaM
    return DL


def genSupervisedTrainingFile(fullSet,output,i,n,size):
    fullTrainSet= []
    trainFileSize=0
    #get Data
    if isinstance(fullSet,str): #load into memory
        fullTrainSet= readTrainingFile(fullSet)
    elif isinstance(fullSet,list):
        fullTrainSet = fullSet
        
    outFile = open(output, "w")
    
#     for line in fullTrainingSet[i*size:((i+1)*size-1)]:
# #         print line
#         outFile.write(line)
#     print fullTrainSet
    lset = len(fullTrainSet) #length of the available training set
    increment = lset*1./n
    rng = range(n)
    if size*(n+1.0)/n < lset:
        del rng[i]
        for x in rng:             
            for line in fullTrainSet[int(x*increment):int(x*increment+size/(n-1))]:
                outFile.write(line+"\n")
                trainFileSize=trainFileSize+1
    elif size < lset:
        for x in range(lset):
            if x < i*increment or x > i*increment+lset-size:
                outFile.write(fullTrainSet[x]+"\n")
                trainFileSize=trainFileSize+1
                
    else:
        import numpy as np
        for i in range(size):
            outFile.write(fullTrainSet[np.random.randint(lset-1)]+"\n")
            trainFileSize=trainFileSize+1
        
    outFile.close()
    return trainFileSize




punctuations ={',','.',';',':','"','?','-','|','(',')','@','Â©','Â','Â','“','”',"'",'/','!','#','$','%','^','&','*','_','-','+','=','{','}','[',']','\\','`','~','<','>' } #punctuation marks
def PuncTokens(sLine):
    sRetLine = sLine
#     lenLine = len(sLine)
    exPunc = set(sLine) & punctuations
    
    for punc in exPunc:
        sRetLine = sRetLine.replace(punc, ' ' + punc + ' ')
#     for i in range(lenLine):
#         if sLine[i] in punctuations:
#             if i > 0 and sLine[i-1] != ' ':
#                 sRetLine += ' '
#             sRetLine += sLine[i]
#             if i < lenLine -1 and sLine[i+1] != ' ':
#                 sRetLine += ' '    
#         else:
#             sRetLine += sLine[i]
    return sRetLine.replace('  ',' ')

vowels = {'a','e','i','o','u',"'"}

def segRoot(root,separator = ''):
    
    if separator == '':
        return tuple(segRoot(root,'|').split("|"))
    else:
        retRoot = root
        vSet = set(root) & vowels
        for vowel in vSet:
            retRoot = retRoot.replace(vowel,vowel+separator)
        return retRoot.strip(separator)
        

def RemovePuncs(sLine):
    sRetLine = ''
    lenLine = len(sLine)
    for i in range(lenLine):
        if sLine[i] in punctuations:
            pass    
        else:
            sRetLine += sLine[i]
    return sRetLine.replace('  ',' ').strip()



sLetters ='qwertyuioplkjhgfdsazxcvbnm QWERTYUIOPLKJHGFDSAZXCVBNM'
def getWordsOnly(s):
    s = s.replace("\n",' ').strip()
    return "".join(filter(lambda x:x in sLetters,s))


    
        
def genUnsupervisedTrainFile(fIn,fOut,i,n,size,separator ="\n"): #Generation n-fold validation sample words
    inF = open(fIn,'r', encoding="latin-1") #Open source file
#     print('inF Open')
    outF = open(fOut,'w', encoding="latin-1") #Open output file
    maxRange = int(n/(n-1) * size )#How long should the source training set be
    blockSize = int( size/(n-1))
    beforeRange = (-1, (i-1)*blockSize)
    afterRange = (i*blockSize+1,maxRange)
    ct = 0 # number of words encountered so far
    line = inF.readline()
    broken = False
    while not broken and line:
        line = RemovePuncs(line).replace('  ',' ') #Remove punctuation marks
        for word in line.split(' '):
            ct += 1
            if (ct >= beforeRange[0] and ct <= beforeRange[1]) or (ct >= afterRange[0] and ct <= afterRange[1]):
                outF.write(word  + separator)

            elif ct >= afterRange[1]:
                broken = True
                break
#         if line != '':
#             lct = line.count(' ')+1 #Number of words in the cleaned line
#             ctlct = ct+lct # ct if the entire line is added to the training file 
#             if ctlct <= beforeRange[1]: #Entire line still in before range
#                 outF.write(line.replace(' ','\n')  + '\n')
#                 ct = ctlct
#             elif  (ct+1 >= afterRange[0] and ctlct <= afterRange[1] ) : #Entire line in after rage
#                 outF.write(line.replace(' ','\n') + '\n')
#                 ct = ctlct
#             else: #Handling straddling
#                 if ct+1  <= beforeRange[1]   :# Before range ends in the line, after first line word   #(ct +1  >= afterRange[0] and ctlct  <= afterRange[1] )
#                     dRange = beforeRange[1] - ct # if ct < beforeRange[1] else beforeRange[1] - ct
#                     lineb = '\n'.join(line.split(' ')[:dRange])
#                     outF.write(lineb + '\n')
#                     ct = ct + dRange
#                     
#                 if ct+1 <= afterRange[0] and ctlct  >= afterRange[0] : # after range starts in this line
#                     dStart = afterRange[0] - ct
#                     if ctlct  >= afterRange[1]: #after range ends in line
#                         dEnd =  ctlct - afterRange[1] - ct
#                     else:
#                         dEnd = lct
#                     linea = '\n'.join(line.split(' ')[dStart-1:dEnd])
#                     outF.write(linea + '\n')    
# #                 dRange = beforeRange[1] - ct if ct < beforeRange[1] else beforeRange[1] - ct
#             ct = ctlct    
#             if ct >= afterRange[1]:
#                 break
        
        if separator != "\n": outF.write("\n")
        line = inF.readline()
        if line == '' and ct < afterRange[1] : #EOF
            inF.seek(0)
            line = inF.readline()                   
    inF.close()
    outF.close()
    
  

def gaussian(x, mu, sig):
    if sig == 0.0:
        if x == mu:
            return 1
        else:
            return 0
        

    return (math.exp(-((x - mu)**2.) / (2. * (sig**2.))))/(sig * math.sqrt(2.0 * math.pi))

def genProgs(n):
    ind = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100,110]
    mark =  [(int(n/100*i),i) for i in ind]
    return mark

# @jit(target_backend='cuda')
def sortKeys(keys, key=None, reverse=False):
    _Keys = []
    # print(len(LM))
    low = None 
    while keys:
        low = max(len(keys)-10000,0)
        # print(low)
        _Keys.append(sorted(keys[low:len(keys)],key=key,reverse=reverse))
        del keys[low:len(keys)]
    # LM =  sorted(LM,key=lambda x: sFM.format(str(len(x[0])),x[0][:-1]))
    keys = _Keys 
    keys = list(merge(*keys,key=key,reverse=reverse))
    return keys



# def tests():
#     # testGenerateRandomDirichlet()
# #     testGrams()
# #     testOrdString()
# #     testGenNGrams()
# #     testGGenNGrams()
# #     testRemovePuncs()
# #     testBoundaries()
# #     testBoundary()
# #     testtn()
# #     testfn()
# #     testtp()
# #     testfp()
# #     testSAccuracy()
# #     testAccuracy()
# #     testSegMatchCount()
# #     testAMetrics()
# #     testTrunc()
# #     testGramLen()
# #     testPuncTokens()
# #     testAffixList()
# #     testSegRoot()
# #     testGetResults()
# #     testResultsSplit()
# #     TestGenUnsupervisedTrainFile()
#     # testGetWordsOnly()
# #     testGetDist()
# #     testGetEntropy()
# #     testGetPerplexity()
# #     testGetDLKit98()
# #     testGetDL()
# #     testGetSplitsDLPerp()
#     pass
#
# if __name__ == "__main__": #Let do some tests
#     tests()
    