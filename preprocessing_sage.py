import pandas as pd
import scipy as sp
import numpy as np
from sklearn.preprocessing

class preprocess(object):
    ''' Cleans and processes DataFrame including turning into binary and scaling. '''
    def __init__(self, data):
        ''' Takes data as an argument. '''
        self.data = data

    def _clean_dataset(self, droptype="std", returntype="df"):
        ''' Takes in dataframe and removes nulls and inappropriate values.\n
            First keyword argument allows users to dictate exact null value cleaning they want.\n
            Second keyword argument allows users to dictate specific return data structure. '''
        assert isinstance(droptype, str)                # Checks that droptype input is a string
        assert isinstance(returntype, str)              # Checks that returntype input is a string

        droptype_set, returntype_set = set(["std", "all", "thresh2", "subs1"]), set(["df", "mat"])

        # Cleaning null values (multiple approaches from naïve to complex)
        if droptype in droptype_set:
            if droptype == "std":
                self.data = self.data.dropna()              # Drops every row with any occurrence of NaN
            elif droptype == "all":
                self.data = self.data.dropna(how="all")     # Drops every row where all values are NaNs
            elif droptype == "thresh2":
                self.data = self.data.dropna(thresh=2)      # Drops every row where at least 2 values are NaNs
            elif droptype == "subs1":
                self.data = self.data.dropna(subset=[1])    # Drops every row that has NaNs in column at index 1
        else:
            raise ValueError("Parameter '{}' is not a valid drop-type.".format(droptype))

        '''
        If data is numeric and there are zero-values where there should not be, then it is necessary
        to check the data distribution and see whether or not we can replace zero-values with dummy data
        near the data's mean or mode. If we cannot replace zero-values with means in fear of skewing or
        modifying our distribution, we can completely filter the rows out.
        '''
        # ----------------------------------------------------------------------------------------------------
        # --------------------------------------- ZERO-VALUE FILTERING ---------------------------------------
        # ----------------------------------------------------------------------------------------------------

        # Let N be the hypothetical index of the column we wish to perform zero-value filtering.
        # Let 'M' be the hypothetical title of the column we wish to perform zero-value filtering.

        # data_copy = self.data[:]                                    # Returns copy of dataset (Use for testing)
        #
        # # mean_value_N = data_copy.iloc[:, N].mean(skipna=True)     # Returns mean value of column N
        # mean_value_M = data_copy['M'].mean(skipna=True)
        #
        #                                                             # Filters zero-values into means for column N
        # # data_copy.iloc[:, N] = data_copy.iloc[:, N].mask(data_copy.iloc[:, N] == 0, mean_value_N)
        # data_copy['M'] = data_copy.M.mask(data_copy.M == 0, mean_value_M)

        # If zero-values cannot be replaced with means effectively, we can drop the rows entirely.

        # data_copy = data_copy[(data_copy != 0).all(1)]              # Returns dataframe copy with any zeros removed

        # ----------------------------------------------------------------------------------------------------
        # --------------------------------------- /ZERO-VALUE FILTERING --------------------------------------
        # ----------------------------------------------------------------------------------------------------

        # Returns formatted and cleaned dataset (multiple return data structures)
        if returntype in returntype_set:
            if returntype == "df":
                return self.data                            # Returns cleaned data as Pandas DataFrame
            elif returntype == "mat":
                return self.data.as_matrix()                # Returns cleaned data as NumPy array
        else:
            raise ValueError("Parameter '{}' is not a valid return-type.".format(returntype))

    def _setX_scaler(self,x1=None,x2=None,x3=None,x4=None):
        ''' sets the x axis scaler for the x axis in the _scalar method
            this method works by taking arguments for the clices themseves, for v of this, any value that you want to be open( going to the end), must be declared as None
        '''
        if x1==None:                            # slicing until x2, slicing from x3 to x4
            return self.data[:x2,x3:x4]
        elif x2== None:                         # slicing from x1 to the end, slcing from x3 to x4
            return self.data[x1:,x3:x4]
        elif x3== None:                         # slicing from x1 to x2. slicing from the begining to x4
            return self.data[x1:x2,:x4]
        elif x4 == None:                        # slicing from x1 to x2, slicing from x3 to end
            return self.data[x1:x2,x3:]
        elif x1 ==None and x2== None:           # takes whole, slicing from x3 to x4
            return self.data[:,x3:x4]
        elif x2== None and x3 == None:          # slicing from x1 to end, slicing from beggining to x4
            return self.data[x1:,:x4]
        elif x3== None and x4 == None:          # slicing x1 to x2 , slicing from begginign to end
            return self.data[x1:x2,]

        else raise ValueError("One of the parameters")

    def _setY_scalar(self,y1=None,y2=None,y3=None,y4=None):
        ''' sets the y axis scaler for the y axis in the _scalar method'''
        pass

    def _scaler(self):
        '''inputs for data frame slicing are as follows

           x1= starting x axis slice, could be a list? BINGO
           x2= ending x axis slice
           y1= staring y axis slice
           y2= ending y axis slice
           EXAMPLE
           X = data[:,0:8]
        '''
        dataset_values = self.data.values       #Takes the data fram of the object, takes the vales as the values for the x and y axis
