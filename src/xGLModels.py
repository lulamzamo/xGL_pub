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



import bz2
import pickle
import gc

from circTrie import circTrie

from statistics import mean, stdev
from xGLUtil import gaussian, punctuations

def LCS(s1, s2): #longest_common_substring
    # Inspired by https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Longest_common_substring#Python2
    m = [[0] * (1 + len(s2)) for _ in range (1 + len(s1))]
    longest, x_longest = 0, 0
    for x in range(1, 1 + len(s1)):
        for y in range(1, 1 + len(s2)):
            if s1[x - 1] == s2[y - 1]:
                m[x][y] = m[x - 1][y - 1] + 1
                if m[x][y] >= longest: #Bias to the last LCS found
                    longest = m[x][y]
                    x_longest = x
            else:
                m[x][y] = 0
    return s1[x_longest - longest: x_longest]

def test_LCS():
    assert LCS('ekuqinisekiseni', 'qina') == "qin"
    assert LCS('asiyi', 'siyaya') == "siy"
    assert LCS('asiyi', 'ya') == "y"

def genTClass(word, lemma, lcs=None):
    if lcs == None:
        lcs = LCS(word, lemma)
    
    wordSplits = word.split(lcs)
    lemmaSplits = lemma.split(lcs)
    lTrans = "L"+ wordSplits[0] +">"+ lemmaSplits[0]
    rTrans = "R"+ wordSplits[1] +">"+ lemmaSplits[1]
    
    tClass = (lTrans if lTrans != "L>" else "") + (rTrans if rTrans != "R>" else "")
    return tClass
    

   
def test_genTClass():
    assert genTClass('ekuqinisekiseni', 'qina') == 'Leku>Risekiseni>a'
    assert genTClass('esetyenziswayo', 'sebenza') == 'Lesety>sebRiswayo>a'
    assert genTClass('ixesha', 'xesha') == 'Li>'
    assert genTClass('elithatyathwayo', 'thabatha') == 'Lelithaty>thabRwayo>a'
    assert genTClass('azisiwe', 'azisa') == 'Riwe>a'
    
def genLemma(word, tClass):
    #find prefix and suffix transformations
    rIndex = tClass.find('R')
    if rIndex > -1:
        rTrans = tClass[rIndex+1:]
        tClass = tClass[:rIndex]
    else: 
        rTrans = ''
    

    if tClass and tClass[0] == 'L':
        lTrans = tClass[1:]
    else:
        lTrans = ''
    
    lemma = word
    if lTrans:
        alTrans = lTrans.split('>')
        lemma = alTrans[1] + lemma.removeprefix(alTrans[0])
        
    if rTrans:
        arTrans = rTrans.split('>')
        lemma = lemma.removesuffix(arTrans[0]) + arTrans[1]
        
    return lemma

def test_genLemma():
    assert genLemma('hamba' ,'') == 'hamba'
    assert genLemma('uhamba', 'Lu>') == 'hamba'
    assert genLemma('hambile', 'Rile>a') == 'hamba'
    assert genLemma('bahambile', 'Lba>Rile>a') == 'hamba'
    assert genLemma('kuhanjiwe', 'Lku>Rnjiwe>mba') == 'hamba'
    assert genLemma('ntlupheko', 'Lnt>hRo>a') == 'hlupheka'
    
               
class xGLModel():
    #     @profile
    # class 
    __slots__ = ('lexicon', 'tClasses', 'affixHierarchy', 'threshold', 'dbLocation')
    def __init__(self, dbLocation=None, threshold=0.975 ): #, cascade=False):
        self.lexicon={}
        self.tClasses = {}
        self.threshold = threshold
        self.affixHierarchy = circTrie()
        
        self.dbLocation = dbLocation
        
    
    def dump(self,dbLocation=None):
        if dbLocation: self.dbLocation = dbLocation
        
        print("Saving  Model: %s" % self.dbLocation)
        gc.disable()
        with bz2.open(self.dbLocation+'-xgl.pickle.bz2','wb') as modelFile:
#             self.grams = tuple(self.grams.items())
#             gc.collect()
            
            pickle.dump((self.lexicon,self.tClasses, self.affixHierarchy, self.threshold), modelFile, pickle.HIGHEST_PROTOCOL)#pickle.dump(self.grams,gramFile,pickle.HIGHEST_PROTOCOL)
#             self.grams = dict(self.grams) 
#             gc.collect()

       
        gc.enable()
        print('Finished Saving  Model')
         
    def load(self,dbLocation=None):
        print("Loading  Model: %s" % dbLocation)
        if dbLocation: self.dbLocation = dbLocation
        gc.disable()
        with bz2.open(self.dbLocation+'-xgl.pickle.bz2','rb') as modelFile:
            self.lexicon, self.tClasses, self.affixHierarchy, self.threshold = pickle.load(modelFile) #pickle.load(gramFile)
#             self.grams = dict(self.grams)
#             gc.collect() 
       
        gc.enable()
        print("Finished Loading  model")
    
    def findCircCandidates(self,word):
        testCirc = (word[:-1],word[1:])
        
        candCircs = list( (testCirc[0][:p] ,testCirc[1][s:]) for p in range(len(testCirc[0])) for s in range(len(testCirc[1])+1))
        
        candCircs = filter(lambda c : c in self.tClasses, candCircs)
        # for p in range(len(testCirc[0])):
        #     for s in range(len(testCirc[1])):
        #         if 
        
        return candCircs
    
    def predictLemma(self, word, candidates=False):
        
        #Find Possible Transform for word
        candCircs = self.findCircCandidates(word) #candidate circumfixes
        
        #Evaluate Probability of circumfix
        wordLen = len(word)
        
        circProbs = {gaussian(wordLen, *classStats): tClass  for candCirc in candCircs  for tClass, classStats in self.tClasses[candCirc].items()}
        if len(circProbs) == 0 : return word
        
        maxCircProb = max(circProbs) 
        bestTClass =  circProbs[maxCircProb]   
           
        #transform the word to lemma using most probable transform
        result = [ (genLemma(word, circProbs[classProb]), classProb) for classProb in reversed(circProbs)] if candidates else genLemma(word, bestTClass)
        
                 
        
        
        return result
    
    
    def updateLexicon(self, word, lemma):
        if word in self.lexicon:
            if lemma in self.lexicon[word]:
                self.lexicon[word][lemma] =self.lexicon[word][lemma] + 1
            else:
                self.lexicon[word][lemma] = 1
        else:
            self.lexicon[word] = {lemma:1}
    
    def updateAffixHierarchy(self, affixes):
        pass
            
    def updateTClasses(self, word, lemma):
        lcs = LCS(word, lemma)
        tClass = genTClass(word,lemma,lcs)
        affixes = tuple(word.split(lcs, maxsplit=1))
        
        if affixes in self.affixHierarchy:
            self.affixHierarchy[affixes] = self.affixHierarchy[affixes] + 1
        else:
            self.affixHierarchy[affixes] =1            
            
        if affixes in self.tClasses:
            if tClass in self.tClasses[affixes]:
                self.tClasses[affixes][tClass] = self.tClasses[affixes][tClass] + (len(word),) 
            else:
                self.tClasses[affixes][tClass] = (len(word),)
        else:
            self.tClasses[affixes] = {tClass: (len(word),)}
    
        
            
    
    def fit(self, X_train, Y_train) :#, cascade=False):
        # try:
        for word, lemma in zip(X_train, Y_train, strict=True): #WordLemma Pair
            # Update Lexicon
 
            if word in punctuations or word == '': continue
            self.updateLexicon(word, lemma)
            
            # Update tClasses
            self.updateTClasses(word, lemma)
            
        for circ, tClasses in self.tClasses.items():
            for tClass, classStats in tClasses.items():
                x = mean(classStats), stdev(classStats) if len(classStats)>1 else 0.
                self.tClasses[circ][tClass] = x
            
            
            
            
            
        # except ValueError :
        #     raise #ValueError("The length of X_train and Y_train is not equal")
        return self

    def predict(self,X, showCandidates=False ):
        results = (self.predictLemma(word, showCandidates) for word in X)
        return results


def test_xGLModel():
    #Persistence Test
    testLocation = "./test"
    xglModel = xGLModel(testLocation)
    xglModel.lexicon={'Lula': {'count':1,'mil':4}}
    xglModel.tClasses = {'Lizo':{'count':2,'lis':5}}
    xglModel.threshold = 0.975 
    xglModel.dump()
    xglModel = xGLModel()
    xglModel.load(testLocation)
    # print("Persistence Test=::dblocation: %s, lexicon: %s, tClasses: %s, threshold: %s" % (xglModel.dbLocation, xglModel.lexicon, xglModel.tClasses, xglModel.threshold))
    
    #fits test
    xglModel = xGLModel(testLocation)
    x_train = ('okunye','umhambi','inja','umhambi', 'umhambi')
    y_train = ('nye','hamba','nja','hambi','hambi')
    # y_train_ = y_train + ('lo',)
    # xglModel = xglModel.fit(x_train, y_train_ ) #Must results in a ValueError
    xglModel = xglModel.fit(x_train, y_train )
    xglModel.dump()
    xglModel = xGLModel()
    xglModel.load(testLocation)
    # print("fit Test=::dblocation: %s, lexicon: %s, tClasses: %s, circimfix Tree: %s threshold: %s" % (xglModel.dbLocation, xglModel.lexicon, xglModel.tClasses, str(xglModel.affixHierarchy),  xglModel.threshold))
    
    x_test = ('okuhle','umhambi','inja','umhambi', 'umhambi', 'umbhali', 'ihobe', 'bhla')
    y_test = xglModel.predict(x_test, showCandidates=True)
    print("y_test :" + str(x_test))
    print("y_test :" + str(tuple(y_test)))
    
    

def tests():
    test_LCS()
    test_genTClass()
    test_xGLModel()
    test_genLemma()
    
#     pass
#
if __name__ == "__main__":
    tests()

