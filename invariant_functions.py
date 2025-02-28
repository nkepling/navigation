import numpy as np


class DistanceCalculator:
    def __init__(self,area_of_interest):
        if not isinstance(area_of_interest,np.ndarray):
            area_of_interest = np.array(area_of_interest)
        self.area_of_interest = area_of_interest

    def _compute_distance(self,current_location):
        if not isinstance(current_location,np.ndarray):
            current_location = np.array(current_location)
        return np.linalg.norm(current_location - self.area_of_interest,1)
    
    def __call__(self, current_location):
        return self._compute_distance(current_location)


class ObstacleDectector:
    def __init__(self,env):
        self.obstacles_map = env.unwrapped.obstacles
        self.map_size = self.obstacles_map.shape
    
    def _check_collision(self,current_location):
        x,y = current_location

        if x < 0 or x >= self.map_size[0] or y < 0 or y >= self.map_size[1]:
            return -np.inf
        elif self.obstacles_map[x,y] == 1:
            return -np.inf
        return 0
    
    def __call__(self,current_location):
        return self._check_collision(current_location)   
    
    
def always_moving_toward_goal(trace,calculator:DistanceCalculator):
    """To be called during the expansion step 

    Computes the robustnesss degree for the invariant: G_[t_0,t](d(t) - d(t-1) < 0)
    """
    if len(trace) < 2:
        return 0
    
    distances = [calculator(p) for p in trace]
    if distances == []:
        print("uh oh")
    robustness_values = [distances[t-1] - distances[t] for t in range(1, len(distances))]

    return min(robustness_values)

# def negative_distance(current_position, goal_position):
#     return -np.linalg.norm(current_position - goal_position, ord=1)

def negative_distance(trace, calculator):
    p = trace[-1]
    distances = calculator(p)
    return -distances 

def no_collision(trace,calculator):
    p = trace[-1]
    return calculator(p)

def penalize_staying_in_same_cell(trace):
    if len(trace) < 2:
        return 0
    if np.array_equal(trace[-1],trace[-2]):
        return -10
    return 0

def penalize_indecision(trace,calculators):
    if len(trace) < 2:
        return 0

    current_position = trace[-1]
    previous_position = trace[-2]

    # TODO: Come up with an stl conditon to penealize indicision

    # Calculate distances to each area of interest
    distances_current = [calculator(current_position) for calculator,w in calculators]
    distances_previous = [calculator(previous_position) for calculator,w in calculators]

    # Check if the agent is oscillating between two areas of interest
    if np.argmin(distances_current) != np.argmin(distances_previous):
        return -10  # Penalize oscillation

    return 0


def penalize_changing_direction(trace):
    if len(trace) < 3:
        return 0

    current_direction = np.array(trace[-1]) - np.array(trace[-2])
    previous_direction = np.array(trace[-2]) - np.array(trace[-3])

    # Check if the direction has changed
    if not np.array_equal(current_direction, previous_direction):
        return -10  # Penalize changing direction
    return 0


def manhattan_distance(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])


class VistCells:
    def __init__(self,target_cells):
        self.target_cells = target_cells

    def eventually_visit_cells(self,trace):
        """
        Quantitative robustness for:
        \phi = \Diamond (AgentIn(target_cells))
        
        At each step t, define:
        d_t = min_{c in target_cells} manhattan_distance(agent_pos(t), c)
        r_t = 1 - d_t
        Then the overall 'Eventually' measure is max_{t} r_t.
        
        Parameters
        ----------
        trace : list of (x, y) agent positions over time
        target_cells : list or set of (x, y) grid coordinates to 'eventually' visit
        
        Returns
        -------
        float
            Real-valued robustness. 
            > 0 implies the agent visits or is very close to some target cell at some time.
            The more positive, the deeper the satisfaction.
            Negative indicates it never got closer than distance=1 to any cell in target_cells.
        """

        target_cells = self.target_cells
        if not trace:
            return -float('inf')  # No movement => cannot visit => strongly violated

        # Convert target_cells to a list for iteration
        target_cells = list(target_cells)

        best_robustness = -float('inf')
        for pos in trace:
            # Minimum distance to any target cell
            d_t = min(manhattan_distance(pos, c) for c in target_cells)
            # r_t = 1 - distance
            r_t = 1.0 - d_t
            if r_t > best_robustness:
                best_robustness = r_t

        return best_robustness
    
    def __call__(self, *args, **kwds):
        return self.eventually_visit_cells(*args)


class AvoidCells:
    def __init__(self,forbidden_cells):
        self.forbidden_cells = forbidden_cells

    def always_keep_out_of_cells(self,trace):
        """
        A piecewise distance metric:
        If dist=0 => strong negative (violation)
        If dist=1 => near zone => mild negative or 0
        If dist>=2 => no penalty => +some margin
        Then take min over time for the Always operator.
        """
        forbidden_cells = self.forbidden_cells
        worst_r = float('inf')
        for pos in trace:
            d = min(manhattan_distance(pos, fc) for fc in forbidden_cells)
            if d == 0:
                r_t = -2.0  # standing on cell => strong violation
            else:
                r_t = +0.5  # safe zone => positive margin

            if r_t < worst_r:
                worst_r = r_t

        return worst_r
    
    def __call__(self, *args, **kwds):
        return self.always_keep_out_of_cells(*args)


class TemporalWindowSpec:
    def __init__(self,start_t,end_t,forbidden_cells):
        self.start_t = start_t
        self.end_t = end_t
        self.forbidden_cells = forbidden_cells

    def keep_out_of_cells_during_interval(self,trace):
        """
        Time-bound "Always keep out of cells" with piecewise distance measure:
        - If dist=0 => r_t = -2.0  (strong violation)
        - If dist=1 => r_t = -0.1 (mild negative)
        - If dist>=2 => r_t = +0.5 (safe margin)
        
        Then the overall 'Always' measure is min over t in [start_t, end_t].
        
        Parameters
        ----------
        trace : list of (x, y)
            The agent's trajectory. trace[t] = position at time t.
        forbidden_cells : list or set of (x, y)
            Cells the agent must not occupy.
        start_t : int
            Start of the time window (inclusive).
        end_t : int
            End of the time window (inclusive).
        
        Returns
        -------
        float
            The worst (minimum) robustness in the interval.
            > 0 => agent stays out with some margin,
            <= 0 => violation.
        """
        forbidden_cells = self.forbidden_cells
        start_t  = self.start_t
        end_t = self.end_t

        T = len(trace)
        if T == 0:
            # No steps => trivially satisfied or define how you want it:
            return float('inf')
        
        # Clamp the time window to [0, T-1]
        start_t = max(0, start_t)
        end_t = min(end_t, T - 1)
        
        # If the window is invalid or empty, interpret as trivially satisfied
        if start_t > end_t:
            return float('inf')
        
        # Convert forbidden_cells to a list (or set) if needed
        forbidden_cells = list(forbidden_cells)

        # We'll track the minimum robustness over time => "Always" operator
        worst_r = float('inf')
        
        # Check each time step in [start_t, end_t]
        for t in range(start_t, end_t + 1):
            pos = trace[t]
            
            # 1) Find distance to the closest forbidden cell
            min_dist = float('inf')
            for fc in forbidden_cells:
                d = manhattan_distance(pos, fc)
                if d < min_dist:
                    min_dist = d
                if min_dist == 0:
                    break  # No need to check more if we're on a forbidden cell
            
            # 2) Piecewise logic
            if min_dist == 0:
                r_t = -2.0   # strongly violated
            elif min_dist == 1:
                r_t = -0.1   # mild penalty
            else:
                r_t = 0.5    # safe margin
            
            # 3) Update worst_r (minimum over the interval)
            if r_t < worst_r:
                worst_r = r_t
            
            # Early exit if we find a strong violation
            if worst_r <= -2.0:
                return worst_r
        
        return worst_r
    
    def __call__(self, *args, **kwds):
        return self.keep_out_of_cells_during_interval(*args)

class VisitInOrder: 
    def __init__(self,priority_cell_list):
        self.priority_cell_list = priority_cell_list
    
    def always_visit_in_order(self,trace):
        priority_cell_list = self.priority_cell_list
        if not priority_cell_list:
        # No required cells => trivially satisfied
            return 0.0

        idx = 0
        total = len(priority_cell_list)

        for pos in trace:
            # Check if the agent is at the 'next' required cell
            if pos == priority_cell_list[idx]:
                idx += 1
                # If we've satisfied the entire sequence, break
                if idx == total:
                    break

        # Fraction of the order that was satisfied

        progress_frac = idx/total
        robustness = 2 * progress_frac - 1
        return robustness
    
    def __call__(self, *args, **kwds):
        self.always_visit_in_order(*args)


def conjunction_of_specs(trace,specs):
    h_list = [spec_func(trace) for spec_func in specs]
    return min(h_list)