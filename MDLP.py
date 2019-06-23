from operator import itemgetter
from collections import Counter
import bisect
from math import log, log2


class DiscretizationMDLP:
    
    def __init__(self, x, y):
        
        if len(x) != len(y):
            raise Exception("len(x)!=len(y)")
        
        x, y = zip(*sorted(zip(*[x, y]), key=itemgetter(0), reverse=False))
        self.x = list(x)
        self.y = list(y)
    
    
    # count a list of classes, y
    def get_count(self, y):
        
        ## count the number of each class in class_y
        counter_y = Counter(y)
        class_y, count_y = zip(*counter_y.items())
        class_y = list(class_y)
        count_y = list(count_y)
        ## the number of samples within y[low:upp]
        n_y = sum(count_y)
        ## the number of distinct classes within y[low:upp]
        n_class_y = len(class_y)
        
        return {'class': class_y, 'count': count_y, 'n_class': n_class_y, 'n': n_y}
    
    # calculate the entropy by giving the number of each class within y
    def get_entropy(self, count_y, n_y, n_class_y):
        
        return (-1)*(sum(map(lambda i: i*log2(i), count_y)) - n_y*log2(n_y))/n_y
    
    # calculate the cost hypothesis given a list of classes, y
    def cost_hypothesis(self, y):
        
        ## count the number of each class in y
        counter_y = self.get_count(y)
        ## calculate the entropy of y
        entropy_y = self.get_entropy(counter_y['count'], counter_y['n'], counter_y['n_class'])
        ## calculate the cost of partitioning
        cost = (counter_y['n']+counter_y['n_class'])*entropy_y
        
        return cost
        
        
    # get the index of cut point between index low and upp [low, upp]
    def get_cut(self, low, upp):
        
        # get a list of feature(x) values, which is a list of potential cut points
        check_list = sorted(set(self.x[low:upp]), reverse=False)
        ## count the number of each class in y
        counter_y = self.get_count(self.y[low:upp])
        # calculate cost(NT)    
        cost_nt = self.cost_hypothesis(self.y[low:upp])
        
        # calculate the adjusted term for cost(HT)
        adjusted_terms = log2((counter_y['n']-1)*(3**counter_y['n_class']-2))
        
        # go through the whole check list to calculate cost(HT) of accepting the partition
        cut_point_candidates = list(map(lambda i: (check_list[i]+check_list[i+1])/2, range(0, len(check_list)-1)))
        
        if len(cut_point_candidates)==0:
            return None, None, False
        
        # given a cut value from the list of cut-point candidates
        list_cost_ht = []
        list_cut_index = []
        for cut_value in cut_point_candidates:
            
            ## get the index of the current cut value
            cut_index = bisect.bisect_right(self.x[low:upp], cut_value) + low
            ## get lists of the current searched class list for right and left
            left_y = self.y[low:cut_index] 
            right_y = self.y[cut_index:upp] 
            ## get the costs for the left and right
            left_cost = self.cost_hypothesis(left_y)
            right_cost = self.cost_hypothesis(right_y)
            ## calculate cost(HT) = left_cost + right_cost and append it to list_cost_ht
            list_cost_ht.append(left_cost + right_cost)
            list_cut_index.append(cut_index)
        
        ## get the minimum of cost(HT) with what cost value, cut value, and cut index
        
        minimum_cost = min(list_cost_ht)
        minimum_cost_index = list_cost_ht.index(minimum_cost)
        cut_values = cut_point_candidates[minimum_cost_index]
        cut_index = list_cut_index[minimum_cost_index]
        
        ## if cost(HT) < cost(NT), accept this partition; otherwise, reject
        if (adjusted_terms+minimum_cost) < cost_nt:
            is_accepted = True
        else:
            is_accepted = False
        
        return cut_index, cut_values, is_accepted
        
    def get_partition_points(self):
        
        list_cut_index = []
        list_cut_value = []
        search_intervals = [(0, len(self.y))]
        
        while len(search_intervals) > 0:
            low, upp = search_intervals.pop()
            
            cut_index, cut_value, is_accepted = self.get_cut(low, upp)
            
            if not is_accepted:
                continue
            search_intervals.append((low, cut_index))
            search_intervals.append((cut_index, upp))
            
            list_cut_index.append(cut_index)
            list_cut_value.append(cut_value)
            
        return sorted(list_cut_index), sorted(list_cut_value)
