import numpy as np
import math
from collections import defaultdict


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
        For each cell c in target_cells, compute \Diamond(AgentIn(c)),
        then take the conjunction => min of those eventually-measures.
        This enforces that *each* cell c is visited.
        """

        target_cells = self.target_cells

        # If no movement, can't visit anything => strong violation
        if not trace:
            return 0
        # We'll gather an 'eventually' measure for each cell
        eventually_list = []

        for c in target_cells:
            # Compute 'eventually' measure for cell c
            # That is: max_{t} [1 - d_t(c)], where d_t(c)=ManhattanDistance(pos(t), c)
            local_best = -float('inf')
            for pos in trace:
                d = manhattan_distance(pos, c)
                r_t = 1.0 - d
                if r_t > local_best:
                    local_best = r_t
            eventually_list.append(local_best)

    # Conjunction of eventually means min over all c
        return min(eventually_list)
    
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
                r_t = -1.0   # strongly violated
            elif min_dist == 1:
                #r_t = -0.1   # mild penalty
                r_t = 0.0
            else:
                r_t = 1.0    # safe margin
            
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

class DontStayInSameCell:
    def __init__(self,penalty_lambda=10.0):
        self.penalty_lambda=10.0

    def never_stay_in_same_coordinate_lambda(self, trace):
        """
        Quantitative STL robustness for G( p(t) != p(t+1) ), with a lambda penalty.
        
        - If p(t) = p(t+1) for any t, we assign a negative penalty: -penalty_lambda.
        - Otherwise (no duplicates in consecutive coords), we assign a positive margin (e.g. +1.0).
        - We then take the minimum over all time steps.
        
        Parameters
        ----------
        trace : list of positions, where each position is e.g. np.array([x,y]) or a tuple (x, y).
        penalty_lambda : float
            The magnitude of the penalty if a violation occurs at any time step.
        
        Returns
        -------
        float
            The real-valued robustness measure.
            - <= 0 indicates that the property was violated (the agent stayed in the same cell at least once).
            - > 0 indicates satisfaction with some positive margin.
        """

        penalty_lambda = self.penalty_lambda

        # If there's fewer than 2 positions, there's no step to compare => trivially satisfied
        if len(trace) < 2:
            return 1.0

        worst_robustness = float('inf')
        for t in range(len(trace) - 1):
            if trace[t] ==trace[t + 1]:
                # Violation: stayed in the same coordinate => negative penalty
                r_t = -penalty_lambda
            else:
                # No violation for this step => assign a positive margin
                r_t = 1.0

            # In STL "Always" is the min over time
            if r_t < worst_robustness:
                worst_robustness = r_t
            
            # Optional short-circuit: if we find a strong negative violation, we can return immediately
            if worst_robustness <= -penalty_lambda:
                return worst_robustness

        return worst_robustness
    
    def __call__(self, *args, **kwds):
        return self.never_stay_in_same_coordinate_lambda(*args)
    
class PenalizeRevisitation:

    def __init__(self,penalty=0.5):
        self.penalty = penalty

    def penalize_revisiting_cells(self,trace):
        """
        Quantitative STL-style robustness that penalizes revisiting any cell.
        
        - If at time t the agent is in a cell it has already occupied
        at any time < t, we incur a penalty (by default, -5).
        - Otherwise, 0 penalty.
        - We take the 'Always' aggregator => min over time steps.

        Parameters
        ----------
        trace : list
            List of positions, each position e.g. np.array([x,y]) or tuple(x, y).
        penalty : float
            Negative penalty applied each time a cell is revisited.

        Returns
        -------
        float
            The minimum local robustness across time (STL 'Always').
            <= 0 => a violation (some cell revisited).
            > 0  => no cell was revisited.
            Note: you can adjust how "strong" a violation is by using larger negative penalty.
        """

        if len(trace) < 2:
            return 0 
        
        scale = self.penalty

        visited_counts = defaultdict(int)
        worst_r = float('inf')

        for pos in trace:
            key = tuple(pos)
            visited_counts[key] += 1
            count = visited_counts[key]
        
            # Local penalty: 1 - (count * scale)
            r_t = 1.0 - (count * scale)
        
            # "Always" => take min over all time
            if r_t < worst_r:
                worst_r = r_t

        return worst_r
    
    def __call__(self, *args, **kwds):
        return self.penalize_revisiting_cells(*args)
    


def local_robustness_avoid(trace, start_t, end_t, forbidden_cells):
    """
    Returns a list of per-timestep robustness values for 
    "Avoid forbidden_cells from t=start_t..end_t."

    local_robustness[t] = piecewise measure at time t only.
    """
    T = len(trace)
    values = []

    for t in range(T):
        # If t outside [start_t, end_t], we treat it as not constrained => +1.0
        if t < start_t or t > end_t:
            values.append(1.0)
            continue

        pos = trace[t]
        # Compute min distance to any forbidden cell
        min_dist = float('inf')
        for fc in forbidden_cells:
            d = manhattan_distance(pos, fc)
            if d < min_dist:
                min_dist = d
            if min_dist == 0:
                break

        # Piecewise logic
        if min_dist == 0:
            # On a forbidden cell => negative
            r_t = -1.0
        elif min_dist == 1:
            r_t = 0.0
        else:
            r_t = 1.0

        values.append(r_t)

    return values


def local_robustness_visit_all(trace, target_cells):
    """
    local2[t] = +1.0 if by time t we've visited *all* target_cells,
                 or a negative if we have visited partial or none.
    For a continuous measure, you could scale by fraction visited.
    """
    T = len(trace)
    visited_set = set()
    target_cells = set(target_cells)

    values = []

    for t in range(T):
        pos = tuple(trace[t])
        visited_set.add(pos)

        # Check how many target cells are visited
        num_visited = sum(1 for c in target_cells if c in visited_set)
        fraction_visited = num_visited / len(target_cells) if target_cells else 1.0

        # If all visited => local2[t] = +1
        # else scale linearly: e.g. local2[t] = 2*fraction_visited - 1
        # That way 0% => -1, 100% => +1
        r_t = 2.0*fraction_visited - 1.0

        values.append(r_t)

    return values



def until_robustness(local1, local2):
    """
    Compute robustness for (phi1 U phi2) from time t=0..T-1,
    given local1[t], local2[t] for t in [0..T-1].
    """
    T = len(local1)
    assert len(local2) == T, "local1, local2 must have same length"

    if T == 0:
        return -math.inf  # no time to satisfy

    best_val = -math.inf

    for t_prime in range(T):
        # Evaluate phi2 at time t_prime
        val2 = local2[t_prime]

        # Evaluate phi1 from time 0 up to t_prime-1
        if t_prime == 0:
            # No times before 0
            val1_interm = math.inf
        else:
            val1_interm = min(local1[tau] for tau in range(t_prime))

        # Candidate aggregator for finishing at t_prime
        local_val = min(val2, val1_interm)

        # Take max over all possible t_prime
        if local_val > best_val:
            best_val = local_val

    return best_val





class StaticEnvExpSpec:
    """
    """
    def __init__(self,start_t,end_t,forbidden_cells,target_cells):
         self.start_t = start_t
         self.end_t = end_t
         self.forbidden_cells = forbidden_cells
         self.target_cells = target_cells
    
    def bounded_until_avoid_then_visit(self, trace):
        """
        STL Bounded Until for:
        (Avoid forbidden_cells in [start_t..end_t]) U (Eventually visit all target_cells).

        Returns a single scalar robustness.
        
        1) local1[t] = how well we avoid forbidden cells at time t if t in [start_t..end_t].
        2) local2[t] = how many target_cells have been visited by time t => eventually measure.
        3) final = Until aggregator over t=0..T-1.
        """


        # 1) local array for the "avoid" spec
        local1 = local_robustness_avoid(trace, self.start_t, self.end_t, self.forbidden_cells)

        # 2) local array for the "eventually visit" spec
        local2 = local_robustness_visit_all(trace, self.target_cells)

        # 3) Combine them with the standard "Until" aggregator
        return until_robustness(local1, local2)

    def __call__(self,*args,**kwargs):
        return self.bounded_until_avoid_then_visit(*args)








def conjunction_of_specs(trace,specs):
    h_list = [spec_func(trace) for spec_func in specs]
    return min(h_list)


