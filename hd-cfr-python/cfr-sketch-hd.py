import numpy as np

def softmax(x):
    e_x = np.exp(x - np.max(x))  # subtract max for numerical stability
    return e_x / e_x.sum(axis=0)
def hopfield_dict(input : np.ndarray, keys : np.ndarray, values : np.ndarray, temperature = 0.01):
    return np.matmul(softmax(np.matmul(keys, input) / temperature), values)

average_policy_keys = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1],
],dtype=float)
average_policy_values = np.array([
    [0,0,0],
    [0,0,0],
    [0,0,0],
],dtype=float)

current_policy_keys = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1],
],dtype=float)
current_policy_values = np.array([
    [300,300,-10],
    [300,300,-10],
    [300,300,-10],
],dtype=float)

expected_values_nom_keys = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1],
],dtype=float)
expected_values_nom_values = np.array([
    [0,0,0],
    [0,0,0],
    [0,0,0],
],dtype=float)

expected_values_den_keys = np.array([
    [1,0,0],
    [0,1,0],
    [0,0,1],
],dtype=float)
expected_values_den_values = np.array([
    [0,0,0],
    [0,0,0],
    [0,0,0],
],dtype=float)

def expected_values(key : np.ndarray):
    a,b = hopfield_dict(key,expected_values_nom_keys, expected_values_nom_values), hopfield_dict(key,expected_values_den_keys, expected_values_den_values)
    return np.divide(a, b, where=(b != 0))

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

def cfr_update(key : np.ndarray, action_index : int, action_reward : float, path_prob_self : float, path_prob_opponent : float, path_probability_sampling : float):
    global count, current_policy_keys, current_policy_values, average_policy_keys, average_policy_values, expected_values_nom_keys, expected_values_nom_values, expected_values_den_keys, expected_values_den_values
    epsilon = 0.0
    def get(keys : np.ndarray, values : np.ndarray):
        return hopfield_dict(key, keys, values)
    a,c,e = get(average_policy_keys, average_policy_values), get(current_policy_keys,current_policy_values), expected_values(key)
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

    average_policy_update = get_prob_distr(c + current_policy_update)
    e_update_nom = np.zeros_like(e)
    e_update_den = np.zeros_like(e)
    e_update_nom[action_index] = action_reward
    e_update_den[action_index] = 1

    # Mutable updates
    current_policy_keys = np.vstack([current_policy_keys, key])
    current_policy_values = np.vstack([current_policy_values, current_policy_update])
    average_policy_keys = np.vstack([average_policy_keys, key])
    average_policy_values = np.vstack([average_policy_values, average_policy_update])
    expected_values_nom_keys = np.vstack([expected_values_nom_keys, key])
    expected_values_nom_values = np.vstack([expected_values_nom_values, e_update_nom])
    expected_values_den_keys = np.vstack([expected_values_den_keys, key])
    expected_values_den_values = np.vstack([expected_values_den_values, e_update_den])

# get_prob_distr(average_policy["state_a"])
input = np.array([1,0,0])
print(hopfield_dict(input, expected_values_nom_keys, expected_values_nom_values))
print(hopfield_dict(input, expected_values_den_keys, expected_values_den_values))
cfr_update(input,1,150,1,1,1)
print(hopfield_dict(input, expected_values_nom_keys, expected_values_nom_values))
print(hopfield_dict(input, expected_values_den_keys, expected_values_den_values))
cfr_update(input,0,225,1,1,1)
print(hopfield_dict(input, expected_values_nom_keys, expected_values_nom_values))
print(hopfield_dict(input, expected_values_den_keys, expected_values_den_values))
# print((expected_values_nom_keys, expected_values_nom_values))
# print((expected_values_den_keys, expected_values_den_values))