import pandas as pd
import numpy as pd
from loader_data_csv_rankings import data_csv_rankings

class Fairness_metrics_in_rankings:

    def __init__(self, path: str, EDr):
        self.model_url = path
        self.EDr = EDr

    def b(k):
        '''Function defining the position bias: the highest ranked candidates receive more attention from users than candidates at lower ranks, and here is adoptedwith algorithmic discount with smooth reduction and favorable theoretical properties (https://proceedings.mlr.press/v30/Wang13.html).'''
        return 1/np.log2(k+1)

    def Exposure_distance(self,path,model,sensitive,Protected_attirbute='Women',
                      Non_protected_attribute = 'Men'):
        '''Exposure distance to see where are the two groups located in the ranking'''
        
        dataset = data_csv_rankings(path)
        Dataframe_ranking = model(dataset,'Value')

        Rankings_per_attribute = {}
        for attribute_value in [Protected_attirbute,Non_protected_attribute]:
            Rankings_per_attribute[attribute_value] = list(Dataframe_ranking[Dataframe_ranking[attribute] == attribute_value].Ranking)

        self.EDr = np.round((sum([b(1/(r+1)) for r in Rankings_per_attribute[Protected_attirbute]])-sum([b(1/(r+1)) for r in Rankings_per_attribute[Non_protected_attribute]]))/2000,2)
        
        return (self.EDr)     
