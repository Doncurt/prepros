import pandas as pd
import scipy as sp
import numpy as np
from sklearn.preprocessing import Binarizer
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScale

class preprocess(object):
    ''' Cleans and processes DataFrame including turning into binary and scaling. '''

    def __init__(self, data):
        ''' Takes data as an argument.'''
        self.data = data

    def _clean_dataset(self, droptype="std", returntype="df"):
        '''
        Takes in dataframe and removes nulls and inappropriate values.\n
        First keyword argument allows users to dictate exact null value cleaning they want.\n
        Second keyword argument allows users to dictate specific return data structure.
        '''
        assert isinstance(droptype, str)                # Checks that droptype input is a string
        assert isinstance(returntype, str)              # Checks that returntype input is a string

        droptype_set, returntype_set = set(["std", "all", "thresh2", "subs1"]), set(["df", "mat"])

        # Cleaning null values (multiple approaches from naïve to complex)
        if droptype in droptype_set:
            if droptype == "std":
                self.data = self.data.dropna()              # Drops every row with any occurrence of NaN

            elif droptype == "all":
                self.data = self.data.dropna(how="all")     # Drops every row where all values are NaNs

            # TODO: Use string formatting to accept last char of 'thresh{}' as integer for threshold value
            elif droptype == "thresh2":
                self.data = self.data.dropna(thresh=2)      # Drops every row where at least 2 values are NaNs

            # TODO: Use string formatting to accept last char of 'subs{}' as integer for subset value
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
        # TODO: Verify integrity of zero-value filtering script separately from rest of _clean_dataset()

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

    def _setXs(self,x1=None,x2=None,x3=None,x4=None):
        '''
        Sets the x axis scaler for the x axis in the _scalar method.
        This method works by taking arguments for the clices themseves, for v of this, any value that you want to be open( going to the end), must be declared as None
        '''
        if x1 is None:                              # slicing until x2, slicing from x3 to x4
            return self.data.values[:x2,x3:x4]

        elif x2 is None:                            # slicing from x1 to the end, slcing from x3 to x4
            return self.data.values[x1:,x3:x4]

        elif x3 is None:                            # slicing from x1 to x2. slicing from the begining to x4
            return self.data.values[x1:x2,:x4]

        elif x4 is None:                            # slicing from x1 to x2, slicing from x3 to end
            return self.data.values[x1:x2,x3:]

        elif x1 is None and x2 is None:             # takes whole, slicing from x3 to x4
            return self.data.values[:,x3:x4]

        elif x2 is None and x3  is None:            # slicing from x1 to end, slicing from beggining to x4
            return self.data.values[x1:,:x4]

        elif x3 is None and x4  is None:                # slicing x1 to x2 , slicing from beginning to end
            return self.data.values[x1:x2,:]

        else raise ValueError("One of the parameters")

    def _setYs(self, y1=None, y2=None, y3=None, y4=None):
        '''
        Sets the y axis scaler for the x axis in the _scalar method.
        This method works by taking arguments for the slices themseves, any value that you want to be open (going to the end), must be declared as None.
        '''
        if y1 isNone:                               # slicing until y2, slicing from y3 to y4
            return self.data.values[:y2,y3:y4]

        elif y2 is None:                            # slicing from y1 to the end, slcing from y3 to y4
            return self.data.values[y1:,y3:y4]

        elif y3 is None:                            # slicing from y1 to y2. slicing from the begining to x4
            return self.data.values[y1:y2,:y4]

        elif y4 is None:                            # slicing from y1 to y2, slicing from y3 to end
            return self.data.values[y1:y2,y3:]

        elif y1 is None and y2 is None:             # takes whole, slicing from y3 to y4
            return self.data.values[:,y3:y4]

        elif y2 is None and y3 is None:             # slicing from y1 to end, slicing from begining to y4
            return self.data.values[y1:,:y4]

        elif y3 is None and y4 is None:             # slicing y1 to y2 , slicing from beginign to end
            return self.data.values[y1:y2,:]

    def _rescaler(self):
        '''
        Rescales data using the Miniscaler for KNN and other machine learning algorithms.
        '''
        dataset_values = self.data.values       #Takes the dataframe of the object, takes the vales as the values for the x and y axis
