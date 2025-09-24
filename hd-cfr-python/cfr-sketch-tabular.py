import numpy as np

average_policy = {
    "state_a" : np.array([0,0,0],dtype=float),
    "state_b" : np.array([0,0,0],dtype=float),
    "state_c" : np.array([0,0,0],dtype=float),
}
current_policy = {
    "state_a" : np.array([300,300,-10],dtype=float),
    "state_b" : np.array([300,300,-10],dtype=float),
    "state_c" : np.array([300,300,-10],dtype=float),
}
expected_values_nom = {
    "state_a" : np.array([0,0,0],dtype=float),
    "state_b" : np.array([0,0,0],dtype=float),
    "state_c" : np.array([0,0,0],dtype=float),
}
expected_values_den = {
    "state_a" : np.array([0,0,0],dtype=float),
    "state_b" : np.array([0,0,0],dtype=float),
    "state_c" : np.array([0,0,0],dtype=float),
}
def expected_values(key : str):
    a,b = expected_values_nom[key], expected_values_den[key]
    return np.divide(a, np.maximum(2 ** -30, b))

count = 2.

def relu(x):
    return np.maximum(0, x)

def get_prob_distr(x : np.ndarray) -> np.ndarray:
    x = relu(x)
    s = np.sum(x)
    return x / s if s != 0 else np.full_like(x, 1 / len(x))

def calculate_expected_values(expected_values : np.ndarray, sampling_prob : float, action_index : int, action_reward : float):
    e = np.zeros_like(expected_values)
    # print(f"sampling_prob: {sampling_prob}")
    # print(f"action_index: {action_index}")
    # print(f"action_reward: {action_reward}")
    for i in range(len(e)):
        ev = expected_values[i]
        if i == action_index:
            # R - Y + E[Y]
            e[i] = (action_reward - ev) / sampling_prob + ev
        else:
            e[i] = ev
    return e

def cfr_update(key : str, action_index : int, action_reward : float, path_prob_self : float, path_prob_opponent : float, path_probability_sampling : float):
    global count
    alpha = 1 / count
    epsilon = 0.0
    def get(d : dict):
        if key in d:
            return d[key]
        else:
            return np.array([0,0,0], dtype=float)
    a,c,e = get(average_policy), get(current_policy), expected_values(key)
    current_policy_probs = get_prob_distr(c)
    uniform_prob = 1 / len(current_policy_probs)
    action_prob : float = current_policy_probs[action_index]
    sampling_prob : float = (1 - epsilon) * action_prob + epsilon * uniform_prob
    new_expected_values = calculate_expected_values(e, sampling_prob, action_index, action_reward)
    # print(f"new_expected_values: {new_expected_values}")
    # print(f"current_policy_probs: {current_policy_probs}")
    mean_of_ev = np.dot(new_expected_values, current_policy_probs)
    # print(f"mean_of_ev: {mean_of_ev}")
    current_policy_update = (new_expected_values - mean_of_ev) * path_prob_opponent / path_probability_sampling
    # print(f"current_policy_update: {current_policy_update}")

    updated_current_policy = (1 - alpha) * c + alpha * current_policy_update
    average_policy_update = get_prob_distr(updated_current_policy)
    print(f"c: {c}")
    print(f"current_policy_update: {current_policy_update}")
    print(f"updated_current_policy: {updated_current_policy}")
    print(f"average_policy_update: {average_policy_update}")
    e_update_nom = np.zeros_like(e)
    e_update_den = np.zeros_like(e)
    e_update_nom[action_index] = action_reward
    e_update_den[action_index] = 1

    # Mutable updates
    current_policy[key] = updated_current_policy
    average_policy[key] = (1 - alpha) * a + alpha * average_policy_update
    expected_values_nom[key] = (1 - alpha) * expected_values_nom[key] + alpha * e_update_nom
    expected_values_den[key] = (1 - alpha) * expected_values_den[key] + alpha * e_update_den
    count = count + 1.

print(current_policy.get("state_a"))
print(average_policy.get("state_a"))
cfr_update("state_a",1,150,1,1,1)
print(current_policy.get("state_a"))
print(average_policy.get("state_a"))
cfr_update("state_a",0,225,1,1,1)
print(current_policy.get("state_a"))
print(average_policy.get("state_a"))
