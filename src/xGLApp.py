# coding=latin-1
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

import argparse

from xGLModels import xGLModel
from xGLUtil import PuncTokens, punctuations


class xGLApp():
    
    def __init__(self,dbLocation=None, threshold= 0.975 ):
        self.model = xGLModel(dbLocation, threshold)

    def fit(self, input_file, separator=',' ):#, smoothing = False ): # ""): #root, prefix, suffix, circumfix
        '''
            Assumption : input_file contain word, lemma pairs separated by separator
            input_file : open file object of text
            separator : string specifying the separating string between word and lemma
        '''

        
        
        words = list(y.strip().split(separator) for y in filter(lambda x: x !='' and x not in punctuations, (PuncTokens(line) for line in input_file.readlines())))
        x_train = [word[0].strip().lower() for word in words]
        y_train = [word[1].strip().lower() for word in words]
        model = self.model.fit(x_train,y_train)
        
        model.dump()
        
        return model    

     
    
    def predict(self,input_file,out_file, showCandidates=False):
        # self.model = xGLModel()
        
        
        words = (PuncTokens(line).strip().lower() for line in input_file.readlines())
        self.model.load()
        results = self.model.predict(words, showCandidates=showCandidates)
        for lemma in results:
            out_file.write (str(lemma) + "\n")
 
    
    def test(self):
        from xGLModels import tests
        tests()
    
    @classmethod   
    def argparse(cls):
        main_parser = argparse.ArgumentParser()
        main_parser.add_argument('command', default='predict', choices = ['test','fit','predict'], help='The mode to run %(prog)s in. ')
        main_parser.add_argument('-input', type=argparse.FileType('r', encoding='UTF-8' ), help='The input filename. When in fiting command mode this is the fiting filename. When in  predict command mode this is the source file to with the words to analyse ')
        main_parser.add_argument('-model', help='The prefix to the file names of the model files')
        main_parser.add_argument('-output', type=argparse.FileType('w'), help='The output file name that the predicted words must be stored in' )
        main_parser.add_argument('-showcandidates', default=False, type=boolType, help='show the candidates and their probabilities')
        main_args = main_parser.parse_args()
        
        if main_args.command == 'test':
            xglApp = xGLApp()
            xglApp.test()
            
        elif main_args.command == 'fit':
            
            fit_parser = argparse.ArgumentParser()
            fit_parser.add_argument('command',   help='The mode to run %(prog)s in. ')
            fit_parser.add_argument('-input', required=True, type=argparse.FileType('r', encoding='UTF-8' ), help='The input filename. When in fiting command mode this is the fiting filename. When in  predict command mode this is the source file to with the words to analyse ')
            fit_parser.add_argument('-model', help='The prefix to the file names of the model files')
            fit_parser.add_argument('-threshold', default=0.975, type= float, help='The minimum confidence threshold ')
            fit_args = fit_parser.parse_args()
            
            xglApp = xGLApp(dbLocation=fit_args.model, threshold=fit_args.threshold)
    #         from datetime import datetime
    #         t1 = datetime.now()
            
            xglApp.fit(fit_args.input)
            
            
    #         t2 = datetime.now()
    #         print(t2-t1)
            
            
        elif main_args.command == 'predict':
            predict_parser = argparse.ArgumentParser()
            predict_parser.add_argument('command',  help='The mode to run %(prog)s in. ')
            predict_parser.add_argument('-input', required=True, type=argparse.FileType('r', encoding='UTF-8' ), help='The input filename. When in fiting command mode this is the fiting filename. When in  predict command mode this is the source file to with the words to analyse ')
            predict_parser.add_argument('-model', required=True, help='The prefix to the file names of the model files')
            predict_parser.add_argument('-output', required=True,  type=argparse.FileType('w'), help='The output file name that the predicted words must be stored in' )
            predict_parser.add_argument('-threshold', default=0.975, type= float, help='The minimum confidence threshold ')
            predict_parser.add_argument('-showcandidates', default=False, type=boolType, help='show the candidates and their probabilities')
            predict_args = predict_parser.parse_args()
            
            xglApp = xGLApp(dbLocation=predict_args.model)
            xglApp.predict(predict_args.input, predict_args.output, predict_args.showcandidates)
                  
def boolType(x):         
    if x in ['True','False']:
        return x == 'True'
    elif set(x) <= {'0','1'}:
        return x
                
if __name__ == "__main__":
    xGLApp.argparse()
        
        
        