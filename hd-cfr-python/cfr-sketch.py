import numpy as np

average_policy = {
    "state_a" : np.array([0,0,0],dtype=float),
    "state_b" : np.array([0,0,0],dtype=float),
    "state_c" : np.array([0,0,0],dtype=float),
}
current_policy = {
    "state_a" : np.array([10,10,-10],dtype=float),
    "state_b" : np.array([10,10,-10],dtype=float),
    "state_c" : np.array([10,10,-10],dtype=float),
}
expected_values = {
    "state_a" : np.array([10,30,-10],dtype=float),
    "state_b" : np.array([30,100,-20],dtype=float),
    "state_c" : np.array([60,20,-30],dtype=float),
}
count = 2.

def relu(x):
    return np.maximum(0, x)

def get_prob_distr(x : np.ndarray) -> np.ndarray:
    x = relu(x)
    s = np.sum(x)
    return x / s if s != 0 else np.full_like(x, 1 / len(x))

def cfr_update(key : str, new_expected_values : np.ndarray, path_prob_self : float, path_prob_opponent : float, path_probability_sampling : float):
    global count
    alpha = 1 / count
    def get(d : dict):
        if key in d:
            return d[key]
        else:
            return np.array([0,0,0], dtype=float)
    a,c,e = get(average_policy), get(current_policy), get(expected_values)
    assert len(new_expected_values) == len(e)
    current_policy_probs = get_prob_distr(c)
    # print(f"current_policy_probs: {current_policy_probs}")
    mean_of_ev = np.dot(new_expected_values, current_policy_probs)
    # print(f"mean_of_ev: {mean_of_ev}")
    current_policy_update = (new_expected_values - mean_of_ev) * path_prob_opponent / path_probability_sampling
    # print(f"current_policy_update: {current_policy_update}")

    updated_current_policy = (1 - alpha) * c + alpha * current_policy_update
    average_policy_update = get_prob_distr(updated_current_policy)

    # Mutable updates
    current_policy[key] = updated_current_policy
    average_policy[key] = (1 - alpha) * a + alpha * average_policy_update
    expected_values[key] = e + new_expected_values
    count = count + 1.

# get_prob_distr(average_policy["state_a"])
print(get_prob_distr(current_policy["state_a"]))
cfr_update("state_a", np.array([100,150,-200],dtype=float),1,1,1)
print(get_prob_distr(current_policy["state_a"]))
cfr_update("state_a", np.array([170,100,-200],dtype=float),1,1,1)
print(get_prob_distr(current_policy["state_a"]))
# print((np.array([10,30,-10],dtype=float) + np.array([100,100,-200],dtype=float) + np.array([100,100,-200],dtype=float)) / 3)