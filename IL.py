## Code for Imitation Learning
from sklearn.linear_model import LogisticRegression
import numpy as np

def collectData(expert_policy, dp, sample_size=500):
    """
    Uses expert policy pi to collect a dataset [(s_i, a_i)] of size sample_size.
    Assumes that all trajectories start from state 0.

    Input:
        expert_policy: expert policy (array of |S| integers)
        dp: an instance of the DynamicProgramming class
        sample_size: the number of samples to collect

    Output:
        states: An array of the form [s_1, s_2, ..., s_n], where s_i is a one-hot encoding of the state at timestep i,
                    and n=sample_size
        actions: An array of the form [a_1, a_2, ..., a_n], where a_i is the action taken at timestep i,
                    and n=sample_size

    Hint: You may need to make a deep copy of intermediate arrays (e.g. using np.copy())
    Hint: How do you get an action from your policy? How do you transition
    from one state to another?
    """
    #initialize dataset
    states = []
    actions = []
    states = np.zeroes((sample_size, dp.nStates,))
    actions = np.zeroes(sample_size)
    
    #initialize state
    s = 0
    
    # Main data collection loop
    for i in range(sample_size):
        # TODO: Fill in the data collection loop here.
        # Reset state to 0 if we reach the terminal state
        states[i][s] = 1
        actions[i] = expert_policy[s]
        s = np.random.choice(np.arange(17), p=dp.P[actions[i], s])

    return states, actions

def trainModel(states, actions):
    """
    Uses the dataset to train a policy pi using behavior cloning, using
    scikit-learn's logistic regression module as our policy class.

    Input:
        states: An array of the form [s_1, s_2, ..., s_n], where s_i is a one-hot encoding of the state at timestep i
                Note: n>=1
        actions: An array of the form [a_1, a_2, ..., a_n], where a_i is the action taken at timestep i
                Note: n>=1
    Output:
        pi: the learned policy (array of |S| integers)

    Hint: think about how to convert the output of the logistic regression model
          into a policy with the above specification.
    """
    # TODO: Replace the placeholders with the actual definitions of X and y here,
    # where X and y are the training inputs and outputs for the logistic regression model.
    X = states
    y = actions
    # Learn policy using logistic regression
    clf = LogisticRegression(random_state=0).fit(X, y)

    # Convert policy to vector form
    # Note that in collectData, we collected a dataset (s,a), so when we
    # train using logistic regression, we'll get a model phi that maps a one-hot vector s to
    # an integer a. However, our policy evaluation code requires a vector.
    # TODO: fill in the learned policy pi here using the logistic regression model.
    pi = clf.predict_proba(np.eye(17))
    return pi