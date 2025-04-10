# -*- coding: utf-8 -*-
"""
Created on Thu Mar 13 17:14:25 2025

@author: u6942852
"""

import numpy as np
from csv import writer
from os import remove
from shutil import copyfile

class Fileprinter:
    
    def __init__(self, file_name:str, save_freq:int, header=None, resume=False):
        self.file_name=file_name
        self.temp_file_path = '-temp.csv'.join(self.file_name.split('.csv'))
        self.save_freq=save_freq
        self.callno = 0
        if header is not None and self.save_freq > 0: 
            if resume is False:
                self._createfile(header)
            if resume is True:
                try:
                    copyfile(self.file_name, self.temp_file_path)
                    remove(self.temp_file_path)
                except FileNotFoundError:
                    self._createfile(header)
                  
    def Terminate(self):
        self._commit()
        remove(self.temp_file_path)
                      
    def __call__(self, arr):
        if self.save_freq == 0:
            return
        self.callno+=1     
        self._print(arr)
        
        if self.callno % self.save_freq == 0:
            self._commit()
    
    def _print(self, arr):
        with open(self.temp_file_path, 'a', newline='') as file:
            writer(file).writerows(arr) 
            file.close()
    
    def _copyfile(self, forward=True):
        if forward is True:
            try:
                copyfile(self.file_name, self.temp_file_path)
            except FileNotFoundError as e:
                if self.callno == self.save_freq:
                    pass
                else: 
                    raise e 
        else:
           copyfile(self.temp_file_path, self.file_name)
           
    def _commit(self):
        print('\rWriting out to file. Do not interrupt', end='\r')
        self._copyfile(False)
        print('\r'+' '*40, end='\r')
    
    def _createfile(self, header):
        with open(self.file_name, 'w', newline='') as file:
            writer(file).writerow(header)
            file.close()
    

        