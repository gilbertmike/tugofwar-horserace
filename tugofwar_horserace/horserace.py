import numpy as np

class HorseRace:
    def __init__(self, quality_1, quality_2, thres):
        """Creates a new simulation instance with two options with qualities
        `quality_1` and `quality_2`. Decision is made when threshold `thres`
        is reached.

        Every simulation time step, the sigmoid
            P = 1/(1 + exp(-quality_i))
        is used as probability that a stream S_i receives a reward.
        """
        self.quality_1 = quality_1
        self.reward_probs_1 = 1 / (1 + np.exp(-quality_1))
        self.quality_2 = quality_2
        self.reward_probs_2 = 1 / (1 + np.exp(-quality_2))
        self.thres = thres

    def simulate(self, num_sim):
        """Runs `num_sim` simulations and returns history and time of decision.

        Args:
            `num_sim`: number of simulations to be run.

        Returns:
            a list size `num_size` which contains tuples (timestep, decision) 
            where timestep is the time when decision is made and decision is
            the index of chosen option (1 or 2).
        """
        # Accumulator of rewards of option 1 and 2
        S_1 = np.zeros((num_sim,))
        S_2 = np.zeros((num_sim,))
        decision_time = np.zeros(num_sim)
        decision = np.zeros(num_sim)
        time = 1
        while True:
            # Generate rewards for all num_sim simulations for S_1 and S_2
            reward = np.random.rand(num_sim) < self.reward_probs_1
            reward = reward * (decision == 0)       # mask for undecided simulations
            S_1 += reward
            reward = np.random.rand(num_sim) < self.reward_probs_2
            reward = reward * (decision == 0)       # mask for undecided simulations
            S_2 += reward
            # Update decision_time and decision for simulations that just decided 2
            thres_2 = S_2 > self.thres
            decision_time = np.where(
                np.logical_and(thres_2, decision_time == 0),
                time,
                decision_time
            )
            decision = np.where(
                np.logical_and(thres_2, decision == 0),
                2,
                decision
            )
            # Update decision_time and decision for simulations that just decided 1
            thres_1 = S_1 > self.thres
            decision_time = np.where(
                np.logical_and(thres_1, decision_time == 0),
                time,
                decision_time
            )
            decision = np.where(
                np.logical_and(thres_1, decision == 0),
                1,
                decision
            )
            time += 1
            if np.all(decision > 0):
                break
        return list(zip(decision_time, decision))
