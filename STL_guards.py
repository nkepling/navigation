import rtamt
import sys


class STLGuard:
    """STLGuard class

    I refrence the STL gym implementaion: https://github.com/nphamilton/stl-gym/blob/main/stlgym/stlgym.py
    """
    def __init__(self,config_dict):

        self.stl_spec = rtamt.StlDiscreteTimeSpecification()
        

        # Add specifications, variables, and constants
        self.data = {}
        ###### Add constants ######
        if "constants" in config_dict.keys():
            constants = config_dict["constants"]
            for i in constants:
                self.stl_spec.declare_const(i['name'], i['type'], i['value'])
        ###### Add variables ######

        self.stl_variables = config_dict['variables']
        for i in self.stl_variables:
            self.stl_spec.declare_var(i['name'], i['type'])
            self.data[i['name']] = []
            if 'i/o' in i.keys():
                self.stl_spec.set_var_io_type(i['name'], i['i/o'])



        # Collect specifications
        self.specifications = config_dict['specifications']
        spec_str = "out = "
        for i in self.specifications:
            self.stl_spec.declare_var(i['name'], 'float')
            self.stl_spec.add_sub_spec(i['spec'])
            spec_str += i['name'] + ' and '
            if 'weight' not in i.keys():
                i['weight'] = 1.0
        spec_str = spec_str[:-5]
        self.stl_spec.declare_var('out', 'float')
        self.stl_spec.spec = spec_str


        # Parse the specification
        try:
            self.stl_spec.parse()
        except rtamt.STLParseException as err:
            print('STL Parse Exception: {}'.format(err))
            sys.exit()
        

    def calculate_robusness_reward(self, trace):
        """
        We have some base reward given by the environment that the VIN/MCTS agent is trying to maximize. 
        But if we add additional positive rewards for satisfying the STL specifications, the agent will be incentivized to satisfy the STL specifications as well.

        This forms a STL robustness reward function augmetns the stadard MCTS 'select' function. I will assume that the trace preprocessing happens before this function is called.

        Args:
            trace (list[Tuple(Str, float)]): A list of tuples where the first element is the variable name and the second element is the value of the variable. 
        Returns:
            float: The reward value for the trace.
        """
        reward = 0

        for a in trace:
            robustness = self.stl_spec.evaluate(a)
        # for i in self.specifications:
        #     val = self.stl_spec.get_value(['name'])[0]
        #     reward += val * float(i['weight'])

        return reward

    def prune_trace(self, trace):
        """Form an STL guard the prunes traces that we do not want to consider.

        This is different from the robustness reward function as for some traces in in that function its ok if we may have partial satisfaction.
        The specificaitons checked in this function are hard constraints that must be satisfied.. 

    
        """

        robustness = self.stl_spec.evaluate(trace)

        if any(val < 0 for val in robustness):
            return None
        
        return trace
    
def monitor():
    # # stl
    spec = rtamt.StlDiscreteTimeSpecification()
    # spec = rtamt.StlDiscreteTimeOnlineSpecificationCpp()
    spec.declare_var('a', 'float')
    spec.declare_var('b', 'float')
    spec.spec = 'eventually[0,1] (a >= b)'

    try:
        spec.parse()
        spec.pastify()
    except rtamt.RTAMTException as err:
        print('RTAMT Exception: {}'.format(err))
        sys.exit()

    rob = spec.update(0, [('a', 100.0), ('b', 20.0)])
    print('time=' + str(0) + ' rob=' + str(rob))

    print("quantitative semantics", rob)

    rob = spec.update(1, [('a', -1.0), ('b', 2.0)])
    print('time=' + str(0) + ' rob=' + str(rob))

    rob = spec.update(2, [('a', -2.0), ('b', -10.0)])
    print('time=' + str(0) + ' rob=' + str(rob))

    print('robustness', rob)


    print(spec.get_value("b"))


def monitor2():
    dataset = {
        'a' : [(x,x) for x in range(100)],
        'b' : [(x,x**2) for x in range(100)]
    }

    # dataset = {
    #      'a': [(0, 100.0), (1, -1.0), (2, -2.0), (3, 5.0)],
    #      'b': [(0, 20.0), (1, 2.0), (2, -10.0), (3, 5.0)]
    # }


    # dataset = {
    #      'a': [(0, 100.0), (1, -1.0), (2, -2.0), (3, 5.0)],
    #      'b': [(0, 100.0), (1, -1.0), (2, -2.0), (3, 5.0)]
    # }

    dataset = {
        "time": [0, 1, 2, 3, 4],
        "a": [x for x in range(4)],
        "b": [x**2 for x in range(4)]
    }

    spec = rtamt.StlDiscreteTimeSpecification()

    spec.declare_var('a', 'float')
    spec.declare_var('b', 'float')

    spec.spec = 'eventually(a >= b)'
    


    try:
        spec.parse()

    except rtamt.RTAMTException as err:
        print('RTAMT Exception: {}'.format(err))
        sys.exit()


    #       # # eval
    # aTraj = dataset['a']
    # bTraj = dataset['b']
    # for i in range(len(dataset['a'])):
    #     aData = aTraj[i]
    #     bData = bTraj[i]
    #     rob = spec.update(aData[0], [('a', aData[1]), ('b', bData[1])])
    #     val = spec.get_value('a')
    #     print("this is the value of a", val)
    #     print('time='+str(aData[0])+' rob='+str(rob))  

    out = spec.evaluate(dataset)
    print(out)








if __name__ == "__main__":
    # test configs
    monitor2()




    




    
