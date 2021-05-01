import numpy as np
import time
import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from testbedlib.cluster_env import LraClusterEnv
from testbedlib.PolicyGradient_CPO import PolicyGradient
import argparse
from testbedlib.simulator.simulator import Simulator
from itertools import chain

import ray
num_cpus = 20
ray.init(num_cpus = num_cpus) # Specify this system has 4 CPUs.
num_samplers = 20

"""
fish
'--batch_choice': 0, 1, 2, ``` 30
python3 PCPOWithoutSubScheduler.py --batch_choice 0
"""

hyper_parameter = {
        'batch_C_numbers': None
}
params = {
        'batch_size': 50,
        # 'epochs': 100000,
        'epochs': 50000,
        'path': "cpo_27_" + str(hyper_parameter['batch_C_numbers']),
        'rec_path': "cpo_separate_unified_replay_level_formal_new100",
        'recover': False,
        'learning rate': 0.01,
        'nodes per group': 3,
        'number of nodes in the cluster': 27,
        'replay size': 100,
        'container_limitation per node': 8
}

NUM_CONTAINERS = 100

app_node_set = np.array([
     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
     [2, 3, 5, 6, 7, 11, 12, 18, 20, 22, 23, 24, 25, 26],
     [0, 2, 8, 9, 19, 23, 24, 25, 26],
     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26]])
for idx in range(len(app_node_set)):
    print("[INFO] No. of nodes for App {}: {}".format(idx, len(app_node_set[idx]) ))


@ray.remote
class Network():
    def __init__(self, n_actions, n_features, params, idx):

        self.RL_1 = PolicyGradient(
            n_actions=n_actions,
            n_features=n_features,
            learning_rate=params['learning rate'],
            suffix=str(100) + '1a_sampler',
            safety_requirement=params['safety_requirement'],
            idx=idx)

        self.RL_2 = PolicyGradient(
            n_actions=n_actions,
            n_features=n_features,
            learning_rate=params['learning rate'],
            suffix=str(100) + '2a_sampler',
            safety_requirement=params['safety_requirement'],
            idx=idx)

        self.RL_3 = PolicyGradient(
            n_actions=n_actions,
            n_features=n_features,
            learning_rate=params['learning rate'],
            suffix=str(100) + '3a_sampler',
            safety_requirement=params['safety_requirement'],
            idx=idx)

        self.sampler_idx = idx
        print("{} th work initialized".format(idx))

    def set_weights(self, RL_1_weights, RL_2_weights, RL_3_weights):
        self.RL_1.set_weights(RL_1_weights)
        self.RL_2.set_weights(RL_2_weights)
        self.RL_3.set_weights(RL_3_weights)

    def rl_sample(self, observation, source_batch_, index_data, RL_1_weights, RL_2_weights, RL_3_weights, sim, env, params, idx):
        # one_start_time = time.time()

        self.set_weights(RL_1_weights, RL_2_weights, RL_3_weights)

        nodes_per_group = int(params['nodes per group'])
        NUM_NODES = params['number of nodes in the cluster']
        n_actions = nodes_per_group
        n_features = int(n_actions * (env.NUM_APPS + 1 + env.NUM_APPS )+ 1 + env.NUM_APPS)

        observation_episode_1 = []
        action_episode_1 = []
        observation_episode_2 = []
        action_episode_2 = []
        observation_episode_3 = []
        action_episode_3 = []

        def store_episode_1(observations, actions):
            observation_episode_1.append(observations)
            action_episode_1.append(actions)

        def store_episode_2(observations, actions):
            observation_episode_2.append(observations)
            action_episode_2.append(actions)

        def store_episode_3(observations, actions):
            observation_episode_3.append(observations)
            action_episode_3.append(actions)

        def batch_data_sub_parallel(rnd_array):
            rnd_array = rnd_array.copy()
            index_data = []
            for i in range(7):
                index_data.extend([i] * int(rnd_array[i]))

            return rnd_array, index_data


        """
        first layer
        """
        source_batch_first = source_batch_.copy()
        observation_first_layer = np.zeros([nodes_per_group, env.NUM_APPS], int)
        for inter_episode_index in range(NUM_CONTAINERS):
            appid = index_data[inter_episode_index]
            source_batch_first[appid] -= 1
            observation_first_layer_copy = observation_first_layer.copy()
            observation_first_layer_copy[:, appid] += 1
            observation_first_layer_copy = np.append(observation_first_layer_copy, observation_first_layer_copy > 9 * 2, axis=1)
            observation_first_layer_copy = np.append(observation_first_layer_copy, observation_first_layer_copy.sum(axis=1).reshape(nodes_per_group, 1), axis=1)
            observation_first_layer_copy = np.array(observation_first_layer_copy).reshape(1, -1)
            observation_first_layer_copy = np.append(observation_first_layer_copy, appid).reshape(1, -1)
            observation_first_layer_copy = np.append(observation_first_layer_copy, np.array(source_batch_first)).reshape(1, -1)
            action_1, prob_weights = self.RL_1.choose_action(observation_first_layer_copy.copy())
            observation_first_layer[action_1, appid] += 1
            store_episode_1(observation_first_layer_copy, action_1)
        """
        second layer
        """
        observation_second_layer_aggregation = np.empty([0, env.NUM_APPS], int)  # 9*20

        number_cont_second_layer = []

        for second_layer_index in range(nodes_per_group):

            rnd_array = observation_first_layer[second_layer_index].copy()
            source_batch_second, index_data = batch_data_sub_parallel(rnd_array)
            observation_second_layer = np.zeros([nodes_per_group, env.NUM_APPS], int)
            NUM_CONTAINERS_second = sum(source_batch_second)
            number_cont_second_layer.append(NUM_CONTAINERS_second)

            for inter_episode_index in range(NUM_CONTAINERS_second):

                appid = index_data[inter_episode_index]
                source_batch_second[appid] -= 1
                observation_second_layer_copy = observation_second_layer.copy()
                observation_second_layer_copy[:, appid] += 1
                observation_second_layer_copy = np.append(observation_second_layer_copy, observation_second_layer_copy > 3 * 2, axis=1)
                observation_second_layer_copy = np.append(observation_second_layer_copy, observation_second_layer_copy.sum(axis=1).reshape(nodes_per_group, 1), axis=1)
                observation_second_layer_copy = np.array(observation_second_layer_copy).reshape(1, -1)
                observation_second_layer_copy = np.append(observation_second_layer_copy, appid).reshape(1, -1)
                observation_second_layer_copy = np.append(observation_second_layer_copy, np.array(source_batch_second)).reshape(1, -1)

                action_2, prob_weights = self.RL_2.choose_action(observation_second_layer_copy.copy())
                observation_second_layer[action_2, appid] += 1
                store_episode_2(observation_second_layer_copy, action_2)

            observation_second_layer_aggregation = np.append(observation_second_layer_aggregation, observation_second_layer, 0)

        """
        third layer
        """
        observation_third_layer_aggregation = np.empty([0, env.NUM_APPS], int)  # 9*20
        number_cont_third_layer = []

        for third_layer_index in range(nodes_per_group * nodes_per_group):
            rnd_array = observation_second_layer_aggregation[third_layer_index].copy()
            source_batch_third, index_data = batch_data_sub_parallel(rnd_array)
            observation_third_layer = np.zeros([nodes_per_group, env.NUM_APPS], int)
            NUM_CONTAINERS_third = sum(source_batch_third)
            number_cont_third_layer.append(NUM_CONTAINERS_third)

            for inter_episode_index in range(NUM_CONTAINERS_third):
                appid = index_data[inter_episode_index]
                source_batch_third[appid] -= 1
                observation_third_layer_copy = observation_third_layer.copy()
                observation_third_layer_copy[:, appid] += 1

                observation_third_layer_copy = np.append(observation_third_layer_copy, observation_third_layer_copy > 1 * 2, axis=1)
                observation_third_layer_copy = np.append(observation_third_layer_copy, observation_third_layer_copy.sum(axis=1).reshape(nodes_per_group, 1), axis=1)
                observation_third_layer_copy = np.array(observation_third_layer_copy).reshape(1, -1)
                observation_third_layer_copy = np.append(observation_third_layer_copy, appid).reshape(1, -1)
                observation_third_layer_copy = np.append(observation_third_layer_copy, np.array(source_batch_third)).reshape(1, -1)

                action_3, prob_weights = self.RL_3.choose_action(observation_third_layer_copy.copy())
                observation_third_layer[action_3, appid] += 1
                store_episode_3(observation_third_layer_copy, action_3)

            observation_third_layer_aggregation = np.append(observation_third_layer_aggregation, observation_third_layer, 0)



        env.state = observation_third_layer_aggregation.copy()
        assert sum(sum(env.state)) == NUM_CONTAINERS
        assert (env.state.sum(0) == source_batch_).all()
        tput_state = env.state
        tput_breakdown = sim.predict(tput_state.reshape(-1, env.NUM_APPS))
        tput = (tput_breakdown * tput_state).sum() / NUM_CONTAINERS
        reward_ratio = (tput - 0)

        state = env.state
        # These three are not actually used in training, just for logging
        list_check_per_app = (env.state > 1).sum() + max((env.state - 1).max(), 0)
        list_check_sum = sum(env.state.sum(1) > params['container_limitation per node']) + max(max(env.state.sum(1) - params['container_limitation per node']), 0)
        list_check_coex = sum((env.state[:, 1] > 0) * (env.state[:, 2] > 0))

        # list_check = list_check_sum + list_check_coex + list_check_per_app
        list_check = 0
        # container limitation & deployment spread
        for node in range(NUM_NODES):
            for app in range(env.NUM_APPS):
                if env.state[node, :].sum() > params['container_limitation per node'] or env.state[node, app] > 1:
                    list_check += env.state[node, app]
        # print("[List Check INFO]: ", list_check)
        # hardware affinity & increamental deployment        
        for app in range(7):
            node_now = np.where(env.state[:,app]>0)[0]
            for node_ in node_now:
                if node_ not in app_node_set[app]:
                    list_check += env.state[node_,app]

        list_check_ratio = list_check / NUM_CONTAINERS

        sample_info = {}
        sample_info["tput"] = tput
        sample_info["reward_ratio"] = reward_ratio
        sample_info["list_check"] = list_check
        sample_info["list_check_ratio"] = list_check_ratio
        sample_info["list_check_per_app"] = list_check_per_app
        sample_info["list_check_sum"] = list_check_sum
        sample_info["list_check_coex"] = list_check_coex

        # print("{}th time {}, model time {}".format(idx, time.time() - one_start_time, one_model_time_end - one_model_time_start ))
        # print("{}th time {}".format(idx, time.time() - one_start_time))
        return observation_third_layer_aggregation, observation_episode_1, action_episode_1, observation_episode_2, action_episode_2, observation_episode_3, action_episode_3, sample_info


    def mega_sample(self, epoch_i, observation, source_batch_, index_data, RL_1_weights, RL_2_weights, RL_3_weights, sim, env, params, idx):
        if epoch_i < 300:
            mega_size = 1
        else:
            mega_size = 2
        # if epoch_i < 500:
        #     mega_size = 3
        # else:
        #     mega_size = 10

        res = []
        for i in range(mega_size):
            res.append(self.rl_sample(observation, source_batch_, index_data, RL_1_weights, RL_2_weights, RL_3_weights, sim, env, params, idx))
        return res



def train(params):

    """
    parameters set
    """
    print("Current params", params)
    
    NUM_NODES = params['number of nodes in the cluster']
    env = LraClusterEnv(num_nodes=NUM_NODES)
    batch_size = params['batch_size']
    ckpt_path_1 = "./checkpoint/" + params['path'] + "1/model.ckpt"
    ckpt_path_2 = "./checkpoint/" + params['path'] + "2/model.ckpt"
    ckpt_path_3 = "./checkpoint/" + params['path'] + "3/model.ckpt"

    ckpt_path_rec_1 = "./checkpoint/" + params['path'] + "1/model.ckpt"
    ckpt_path_rec_2 = "./checkpoint/" + params['path'] + "2/model.ckpt"
    ckpt_path_rec_3 = "./checkpoint/" + params['path'] + "3/model.ckpt"

    np_path = "./checkpoint/" + params['path'] + "/optimal_file_name.npz"
    Recover = params['recover']
    nodes_per_group = int(params['nodes per group'])
    replay_size = params['replay size']
    training_times_per_episode = 1  # TODO: if layers changes, training_times_per_episode should be modified
    # safety_requirement = 2.0 / 100.
    safety_requirement = params['safety_requirement']
    print("######## safety_requirement = {} ########".format(safety_requirement))
    """
    Build Network
    """
    n_actions = nodes_per_group  #: 3 nodes per group
    n_features = int(n_actions * (env.NUM_APPS + 1 + env.NUM_APPS )+ 1 + env.NUM_APPS)  #: 3*9+1 = 28
    
    model_time_start = time.time()
    RL_1 = PolicyGradient(
        n_actions=n_actions,
        n_features=n_features,
        learning_rate=params['learning rate'],
        suffix=str(100) + '1a',
        safety_requirement=safety_requirement)

    RL_2 = PolicyGradient(
        n_actions=n_actions,
        n_features=n_features,
        learning_rate=params['learning rate'],
        suffix=str(100) + '2a',
        safety_requirement=safety_requirement)

    RL_3 = PolicyGradient(
        n_actions=n_actions,
        n_features=n_features,
        learning_rate=params['learning rate'],
        suffix=str(100) + '3a',
        safety_requirement=safety_requirement)
    model_time_end = time.time()

    print("Init model time: {}".format(model_time_end - model_time_start))

    sim = Simulator()

    """
    Training
    """
    start_time = time.time()
    global_start_time = start_time
    number_optimal = []
    observation_episode_1, action_episode_1, reward_episode_1, safety_episode_1 = [], [], [], []
    observation_optimal_1, action_optimal_1, reward_optimal_1, safety_optimal_1 = [], [], [], []

    observation_episode_2, action_episode_2, reward_episode_2, safety_episode_2 = [], [], [], []
    observation_optimal_2, action_optimal_2, reward_optimal_2, safety_optimal_2 = [], [], [], []

    observation_episode_3, action_episode_3, reward_episode_3, safety_episode_3 = [], [], [], []
    observation_optimal_3, action_optimal_3, reward_optimal_3, safety_optimal_3 = [], [], [], []

    epoch_i = 0

    thre_entropy = 0.1
    # TODO: delete this range

    names = locals()
    for i in range(0, 10):
        names['highest_tput_' + str(i)] = 0
        names['observation_optimal_1_' + str(i)] = []
        names['action_optimal_1_' + str(i)] = []
        names['observation_optimal_2_' + str(i)] = []
        names['action_optimal_2_' + str(i)] = []
        names['observation_optimal_3_' + str(i)] = []
        names['action_optimal_3_' + str(i)] = []
        names['reward_optimal_1_' + str(i)] = []
        names['reward_optimal_2_' + str(i)] = []
        names['reward_optimal_3_' + str(i)] = []
        names['safety_optimal_1_' + str(i)] = []
        names['safety_optimal_2_' + str(i)] = []
        names['safety_optimal_3_' + str(i)] = []
        names['number_optimal_' + str(i)] = []
        names['optimal_range_' + str(i)] = 1.05
        names['lowest_vio_' + str(i)] = 500
        names['observation_optimal_1_vio_' + str(i)] = []
        names['action_optimal_1_vio_' + str(i)] = []
        names['observation_optimal_2_vio_' + str(i)] = []
        names['action_optimal_2_vio_' + str(i)] = []
        names['observation_optimal_3_vio_' + str(i)] = []
        names['action_optimal_3_vio_' + str(i)] = []
        names['reward_optimal_vio_1_' + str(i)] = []
        names['reward_optimal_vio_2_' + str(i)] = []
        names['reward_optimal_vio_3_' + str(i)] = []
        names['safety_optimal_vio_1_' + str(i)] = []
        names['safety_optimal_vio_2_' + str(i)] = []
        names['safety_optimal_vio_3_' + str(i)] = []
        names['number_optimal_vio_' + str(i)] = []
        names['optimal_range_vio_' + str(i)] = 1.1

    # def store_episode_1(observations, actions):
    #     observation_episode_1.append(observations)
    #     action_episode_1.append(actions)

    # def store_episode_2(observations, actions):
    #     observation_episode_2.append(observations)
    #     action_episode_2.append(actions)

    # def store_episode_3(observations, actions):
    #    observation_episode_3.append(observations)
    #    action_episode_3.append(actions)


    tput_origimal_class = 0
    source_batch_, index_data_ = batch_data(NUM_CONTAINERS, env.NUM_APPS)  # index_data = [0,1,2,0,1,2]


    Samplers = [ Network.remote(n_actions, n_features, params, _) for _ in range(num_samplers) ]
    num_sampled = 0

    while epoch_i < params['epochs']:
        time_ep_start = time.time()

        if Recover:
            print("Recover from {}".format(ckpt_path_rec_1))
            RL_1.restore_session(ckpt_path_rec_1)
            RL_2.restore_session(ckpt_path_rec_2)
            RL_3.restore_session(ckpt_path_rec_3)
            Recover = False

        # same in each epoch
        observation = env.reset().copy()  # (9,9)
        source_batch = source_batch_.copy()
        index_data = index_data_.copy()


        """
        Episode
        """
        time_al_start = time.time()
        batch_sample_results = []
        # for ep_idx in range(batch_size):
        #     batch_sample_results.append(rl_sample(observation, source_batch_, index_data, RL_1, RL_2, RL_3, sim, env, params))
        
        RL_1_weights = RL_1.get_weights()
        RL_2_weights = RL_2.get_weights()
        RL_3_weights = RL_3.get_weights()

        batch_sample_results = ray.get(
            [ sampler.mega_sample.remote(epoch_i, observation, source_batch_, index_data, RL_1_weights, RL_2_weights, RL_3_weights, sim, env, params, idx=x) for x, sampler in enumerate(Samplers) ]
            )

        batch_sample_results = list(chain.from_iterable(batch_sample_results))
        num_sampled += len(batch_sample_results)

        # batch_sample_results = ray.get([ rl_sample.remote(observation, source_batch_, index_data, RL_1_weights, RL_2_weights, RL_3_weights, sim, env, params, idx=x)
        #                                 for x in range(num_cpus) ])

        print("Length of batch_sample_results:", len(batch_sample_results))
        time_al_end = time.time()
        print("Allocate and process time", time_al_end - time_al_start)

        '''
        After sampling, process the results
        '''
        time_process_ac = 0.0
        
        # batch_sample_results[sample_idx][0]
        for sample_idx in range(len(batch_sample_results)):
            time_process_start = time.time()
            '''
            move to rl_sample()
            '''
            time_process_ac += time.time() - time_process_start

            '''
            extract information
            '''
            list_check_ratio = batch_sample_results[sample_idx][7]["list_check_ratio"]
            reward_ratio     = batch_sample_results[sample_idx][7]["reward_ratio"]
            tput             = batch_sample_results[sample_idx][7]["tput"]
            list_check       = batch_sample_results[sample_idx][7]["list_check"]

            list_check_per_app  = batch_sample_results[sample_idx][7]["list_check_per_app"]
            list_check_sum  = batch_sample_results[sample_idx][7]["list_check_sum"]
            list_check_coex = batch_sample_results[sample_idx][7]["list_check_coex"]


            safety_episode_1 = [list_check_ratio * 1.0] * len(batch_sample_results[sample_idx][1])
            reward_episode_1 = [reward_ratio * 1.0] * len(batch_sample_results[sample_idx][1])

            safety_episode_2 = [list_check_ratio * 1.0] * len(batch_sample_results[sample_idx][3])
            reward_episode_2 = [reward_ratio * 1.0] * len(batch_sample_results[sample_idx][3])

            safety_episode_3 = [list_check_ratio * 1.0] * len(batch_sample_results[sample_idx][5])
            reward_episode_3 = [reward_ratio * 1.0] * len(batch_sample_results[sample_idx][5])

            RL_1.store_tput_per_episode(tput, (epoch_i * batch_size) + sample_idx, list_check, list_check_per_app, list_check_coex, list_check_sum)
            RL_2.store_tput_per_episode(tput, (epoch_i * batch_size) + sample_idx, list_check, [],[],[])
            RL_3.store_tput_per_episode(tput, (epoch_i * batch_size) + sample_idx, list_check, [],[],[])

            RL_1.store_training_samples_per_episode(batch_sample_results[sample_idx][1], batch_sample_results[sample_idx][2], reward_episode_1, safety_episode_1)
            RL_2.store_training_samples_per_episode(batch_sample_results[sample_idx][3], batch_sample_results[sample_idx][4], reward_episode_2, safety_episode_2)
            RL_3.store_training_samples_per_episode(batch_sample_results[sample_idx][5], batch_sample_results[sample_idx][6], reward_episode_3, safety_episode_3)

            
            """
            check_tput_quality(tput)
            """
            if names['lowest_vio_' + str(tput_origimal_class)] > list_check:
                names['lowest_vio_' + str(tput_origimal_class)] = list_check
                names['observation_optimal_1_vio_' + str(tput_origimal_class)], names['action_optimal_1_vio_' + str(tput_origimal_class)], names['observation_optimal_2_vio_' + str(tput_origimal_class)], names['action_optimal_2_vio_' + str(tput_origimal_class)],  names['number_optimal_vio_' + str(tput_origimal_class)], names['safety_optimal_vio_1_' + str(tput_origimal_class)], names['safety_optimal_vio_2_' + str(tput_origimal_class)], names['safety_optimal_vio_3_' + str(tput_origimal_class)] = [], [], [], [], [], [], [], []
                names['observation_optimal_3_vio_' + str(tput_origimal_class)], names['action_optimal_3_vio_' + str(tput_origimal_class)] = [], []
                names['reward_optimal_vio_' + str(tput_origimal_class)] = []
                names['observation_optimal_1_vio_' + str(tput_origimal_class)].extend(batch_sample_results[sample_idx][1])
                names['action_optimal_1_vio_' + str(tput_origimal_class)].extend(batch_sample_results[sample_idx][2])
                names['observation_optimal_2_vio_' + str(tput_origimal_class)].extend(batch_sample_results[sample_idx][3])
                names['action_optimal_2_vio_' + str(tput_origimal_class)].extend(batch_sample_results[sample_idx][4])
                names['observation_optimal_3_vio_' + str(tput_origimal_class)].extend(batch_sample_results[sample_idx][5])
                names['action_optimal_3_vio_' + str(tput_origimal_class)].extend(batch_sample_results[sample_idx][6])
                names['number_optimal_vio_' + str(tput_origimal_class)].append(NUM_CONTAINERS)
                names['safety_optimal_vio_1_' + str(tput_origimal_class)].extend(safety_episode_1)
                names['safety_optimal_vio_2_' + str(tput_origimal_class)].extend(safety_episode_2)
                names['safety_optimal_vio_3_' + str(tput_origimal_class)].extend(safety_episode_3)
                names['reward_optimal_vio_' + str(tput_origimal_class)].extend(reward_episode_1)

                names['optimal_range_vio_' + str(tput_origimal_class)] = 1.1
            elif names['lowest_vio_' + str(tput_origimal_class)] >= list_check / names['optimal_range_vio_' + str(tput_origimal_class)]:
                names['observation_optimal_1_vio_' + str(tput_origimal_class)].extend(batch_sample_results[sample_idx][1])
                names['action_optimal_1_vio_' + str(tput_origimal_class)].extend(batch_sample_results[sample_idx][2])
                names['observation_optimal_2_vio_' + str(tput_origimal_class)].extend(batch_sample_results[sample_idx][3])
                names['action_optimal_2_vio_' + str(tput_origimal_class)].extend(batch_sample_results[sample_idx][4])
                names['observation_optimal_3_vio_' + str(tput_origimal_class)].extend(batch_sample_results[sample_idx][5])
                names['action_optimal_3_vio_' + str(tput_origimal_class)].extend(batch_sample_results[sample_idx][6])
                names['number_optimal_vio_' + str(tput_origimal_class)].append(NUM_CONTAINERS)
                names['safety_optimal_vio_1_' + str(tput_origimal_class)].extend(safety_episode_1)
                names['safety_optimal_vio_2_' + str(tput_origimal_class)].extend(safety_episode_2)
                names['safety_optimal_vio_3_' + str(tput_origimal_class)].extend(safety_episode_3)
                names['reward_optimal_vio_' + str(tput_origimal_class)].extend(reward_episode_1)

            # if list_check_ratio <= safety_requirement*0.5:
            if list_check_ratio <= safety_requirement:
                if names['highest_tput_' + str(tput_origimal_class)] < tput:
                    names['highest_tput_' + str(tput_origimal_class)] = tput

                    names['observation_optimal_1_' + str(tput_origimal_class)], names['action_optimal_1_' + str(tput_origimal_class)], names['observation_optimal_2_' + str(tput_origimal_class)], names['action_optimal_2_' + str(tput_origimal_class)],\
                    names['reward_optimal_1_' + str(tput_origimal_class)],names['reward_optimal_2_' + str(tput_origimal_class)],names['reward_optimal_3_' + str(tput_origimal_class)], \
                    names['number_optimal_' + str(tput_origimal_class)],\
                    names['safety_optimal_1_' + str(tput_origimal_class)],names['safety_optimal_2_' + str(tput_origimal_class)],names['safety_optimal_3_' + str(tput_origimal_class)]\
                        = [], [], [], [], [], [], [], [], [], [], []
                    names['observation_optimal_3_' + str(tput_origimal_class)], names['action_optimal_3_' + str(tput_origimal_class)] = [], []

                    names['observation_optimal_1_' + str(tput_origimal_class)].extend(batch_sample_results[sample_idx][1])
                    names['action_optimal_1_' + str(tput_origimal_class)].extend(batch_sample_results[sample_idx][2])
                    names['observation_optimal_2_' + str(tput_origimal_class)].extend(batch_sample_results[sample_idx][3])
                    names['action_optimal_2_' + str(tput_origimal_class)].extend(batch_sample_results[sample_idx][4])
                    names['observation_optimal_3_' + str(tput_origimal_class)].extend(batch_sample_results[sample_idx][5])
                    names['action_optimal_3_' + str(tput_origimal_class)].extend(batch_sample_results[sample_idx][6])

                    names['number_optimal_' + str(tput_origimal_class)].append(NUM_CONTAINERS)

                    names['safety_optimal_1_' + str(tput_origimal_class)].extend(safety_episode_1)
                    names['safety_optimal_2_' + str(tput_origimal_class)].extend(safety_episode_2)
                    names['safety_optimal_3_' + str(tput_origimal_class)].extend(safety_episode_3)
                    names['reward_optimal_1_' + str(tput_origimal_class)].extend(reward_episode_1)
                    names['reward_optimal_2_' + str(tput_origimal_class)].extend(reward_episode_2)
                    names['reward_optimal_3_' + str(tput_origimal_class)].extend(reward_episode_3)

                    names['optimal_range_' + str(tput_origimal_class)] = 1.05

                elif names['highest_tput_' + str(tput_origimal_class)] < tput * names['optimal_range_' + str(tput_origimal_class)]:
                    names['observation_optimal_1_' + str(tput_origimal_class)].extend(batch_sample_results[sample_idx][1])
                    names['action_optimal_1_' + str(tput_origimal_class)].extend(batch_sample_results[sample_idx][2])
                    names['observation_optimal_2_' + str(tput_origimal_class)].extend(batch_sample_results[sample_idx][3])
                    names['action_optimal_2_' + str(tput_origimal_class)].extend(batch_sample_results[sample_idx][4])
                    names['observation_optimal_3_' + str(tput_origimal_class)].extend(batch_sample_results[sample_idx][5])
                    names['action_optimal_3_' + str(tput_origimal_class)].extend(batch_sample_results[sample_idx][6])

                    names['number_optimal_' + str(tput_origimal_class)].append(NUM_CONTAINERS)

                    names['safety_optimal_1_' + str(tput_origimal_class)].extend(safety_episode_1)
                    names['safety_optimal_2_' + str(tput_origimal_class)].extend(safety_episode_2)
                    names['safety_optimal_3_' + str(tput_origimal_class)].extend(safety_episode_3)
                    names['reward_optimal_1_' + str(tput_origimal_class)].extend(reward_episode_1)
                    names['reward_optimal_2_' + str(tput_origimal_class)].extend(reward_episode_2)
                    names['reward_optimal_3_' + str(tput_origimal_class)].extend(reward_episode_3)

            # observation_episode_1, action_episode_1, reward_episode_1, safety_episode_1 = [], [], [], []
            # observation_episode_2, action_episode_2, reward_episode_2, safety_episode_2 = [], [], [], []
            # observation_episode_3, action_episode_3, reward_episode_3, safety_episode_3 = [], [], [], []

        """
        Each batch, RL.learn()
        """
        # if (epoch_i % batch_size == 0) & (epoch_i > 1):
        if True:
            for replay_class in range(0,1):

                number_optimal = names['number_optimal_' + str(replay_class)]

                reward_optimal_1 = names['reward_optimal_1_' + str(replay_class)]
                reward_optimal_2 = names['reward_optimal_2_' + str(replay_class)]
                reward_optimal_3 = names['reward_optimal_3_' + str(replay_class)]
                safety_optimal_1 = names['safety_optimal_1_' + str(replay_class)]
                safety_optimal_2 = names['safety_optimal_2_' + str(replay_class)]
                safety_optimal_3 = names['safety_optimal_3_' + str(replay_class)]

                observation_optimal_1 = names['observation_optimal_1_' + str(replay_class)]
                action_optimal_1 = names['action_optimal_1_' + str(replay_class)]
                observation_optimal_2 = names['observation_optimal_2_' + str(replay_class)]
                action_optimal_2 = names['action_optimal_2_' + str(replay_class)]
                observation_optimal_3 = names['observation_optimal_3_' + str(replay_class)]
                action_optimal_3 = names['action_optimal_3_' + str(replay_class)]


                buffer_size = int(len(number_optimal))

                if buffer_size < replay_size:
                    # TODO: if layers changes, training_times_per_episode should be modified
                    RL_1.ep_obs.extend(observation_optimal_1)
                    RL_1.ep_as.extend(action_optimal_1)
                    RL_1.ep_rs.extend(reward_optimal_1)
                    RL_1.ep_ss.extend(safety_optimal_1)

                    RL_2.ep_obs.extend(observation_optimal_2)
                    RL_2.ep_as.extend(action_optimal_2)
                    RL_2.ep_rs.extend(reward_optimal_2)
                    RL_2.ep_ss.extend(safety_optimal_2)

                    RL_3.ep_obs.extend(observation_optimal_3)
                    RL_3.ep_as.extend(action_optimal_3)
                    RL_3.ep_rs.extend(reward_optimal_3)
                    RL_3.ep_ss.extend(safety_optimal_3)

                else:
                    replay_index = np.random.choice(range(buffer_size), size=replay_size, replace=False)
                    for replay_id in range(replay_size):
                        replace_start = replay_index[replay_id]
                        start_location = sum(number_optimal[:replace_start])
                        stop_location = sum(number_optimal[:replace_start+1])
                        RL_1.ep_obs.extend(observation_optimal_1[start_location: stop_location])
                        RL_1.ep_as.extend(action_optimal_1[start_location: stop_location])
                        RL_1.ep_rs.extend(reward_optimal_1[start_location: stop_location])
                        RL_1.ep_ss.extend(safety_optimal_1[start_location: stop_location])

                        RL_2.ep_obs.extend(observation_optimal_2[start_location: stop_location])
                        RL_2.ep_as.extend(action_optimal_2[start_location: stop_location])
                        RL_2.ep_rs.extend(reward_optimal_2[start_location: stop_location])
                        RL_2.ep_ss.extend(safety_optimal_2[start_location: stop_location])

                        RL_3.ep_obs.extend(observation_optimal_3[start_location: stop_location])
                        RL_3.ep_as.extend(action_optimal_3[start_location: stop_location])
                        RL_3.ep_rs.extend(reward_optimal_3[start_location: stop_location])
                        RL_3.ep_ss.extend(safety_optimal_3[start_location: stop_location])



            # print("########")
            # print("RL_1.ep_obs: {}, RL_1.ep_as: {}, RL_1.ep_rs: {}, RL_1.ep_ss: {}".format(len(RL_1.ep_obs), len(RL_1.ep_as), len(RL_1.ep_rs), len(RL_1.ep_ss)))
            # print("RL_2.ep_obs: {}, RL_2.ep_as: {}, RL_2.ep_rs: {}, RL_2.ep_ss: {}".format(len(RL_2.ep_obs), len(RL_2.ep_as), len(RL_2.ep_rs), len(RL_2.ep_ss)))
            # print("RL_3.ep_obs: {}, RL_3.ep_as: {}, RL_3.ep_rs: {}, RL_3.ep_ss: {}".format(len(RL_3.ep_obs), len(RL_3.ep_as), len(RL_3.ep_rs), len(RL_3.ep_ss)))
            # print("########")

            if not RL_1.start_cpo:
                for replay_class in range(0,1):
                    number_optimal = names['number_optimal_vio_' + str(replay_class)]
                    safety_optimal_1 = names['safety_optimal_vio_1_' + str(replay_class)]
                    safety_optimal_2 = names['safety_optimal_vio_2_' + str(replay_class)]
                    safety_optimal_3 = names['safety_optimal_vio_3_' + str(replay_class)]
                    reward_optimal = names['reward_optimal_vio_' + str(replay_class)]

                    observation_optimal_1 = names['observation_optimal_1_vio_' + str(replay_class)]
                    action_optimal_1 = names['action_optimal_1_vio_' + str(replay_class)]
                    observation_optimal_2 = names['observation_optimal_2_vio_' + str(replay_class)]
                    action_optimal_2 = names['action_optimal_2_vio_' + str(replay_class)]
                    observation_optimal_3 = names['observation_optimal_3_vio_' + str(replay_class)]
                    action_optimal_3 = names['action_optimal_3_vio_' + str(replay_class)]

                    buffer_size = int(len(number_optimal))


                    # print("########")
                    # print("number_optimal", number_optimal)
                    # print("observation_optimal_1: {}, action_optimal_1: {}, reward_optimal: {}, safety_optimal_1: {}".format(len(observation_optimal_1), len(action_optimal_1), len(reward_optimal), len(safety_optimal_1)))
                    # print("observation_optimal_2: {}, action_optimal_2: {}, reward_optimal: {}, safety_optimal_2: {}".format(len(observation_optimal_2), len(action_optimal_2), len(reward_optimal), len(safety_optimal_2)))
                    # print("observation_optimal_3: {}, action_optimal_3: {}, reward_optimal: {}, safety_optimal_3: {}".format(len(observation_optimal_3), len(action_optimal_3), len(reward_optimal), len(safety_optimal_3)))
                    # print("########")

                    if buffer_size < replay_size:
                        # TODO: if layers changes, training_times_per_episode should be modified
                        RL_1.ep_obs.extend(observation_optimal_1)
                        RL_1.ep_as.extend(action_optimal_1)
                        RL_1.ep_ss.extend(safety_optimal_1)
                        RL_1.ep_rs.extend(reward_optimal)


                        RL_2.ep_obs.extend(observation_optimal_2)
                        RL_2.ep_as.extend(action_optimal_2)
                        RL_2.ep_rs.extend(reward_optimal)
                        RL_2.ep_ss.extend(safety_optimal_2)

                        RL_3.ep_obs.extend(observation_optimal_3)
                        RL_3.ep_as.extend(action_optimal_3)
                        RL_3.ep_rs.extend(reward_optimal)
                        RL_3.ep_ss.extend(safety_optimal_3)

                    else:
                        replay_index = np.random.choice(range(buffer_size), size=replay_size, replace=False)
                        for replay_id in range(replay_size):
                            replace_start = replay_index[replay_id]
                            start_location = sum(number_optimal[:replace_start])
                            stop_location = sum(number_optimal[:replace_start+1])
                            RL_1.ep_obs.extend(observation_optimal_1[start_location: stop_location])
                            RL_1.ep_as.extend(action_optimal_1[start_location: stop_location])
                            RL_1.ep_rs.extend(reward_optimal[start_location: stop_location])
                            RL_1.ep_ss.extend(safety_optimal_1[start_location: stop_location])

                            RL_2.ep_obs.extend(observation_optimal_2[start_location: stop_location])
                            RL_2.ep_as.extend(action_optimal_2[start_location: stop_location])
                            RL_2.ep_rs.extend(reward_optimal[start_location: stop_location])
                            RL_2.ep_ss.extend(safety_optimal_2[start_location: stop_location])

                            RL_3.ep_obs.extend(observation_optimal_3[start_location: stop_location])
                            RL_3.ep_as.extend(action_optimal_3[start_location: stop_location])
                            RL_3.ep_rs.extend(reward_optimal[start_location: stop_location])
                            RL_3.ep_ss.extend(safety_optimal_3[start_location: stop_location])
            
            
            # print("########")
            # print("RL_1.ep_obs: {}, RL_1.ep_as: {}, RL_1.ep_rs: {}, RL_1.ep_ss: {}".format(len(RL_1.ep_obs), len(RL_1.ep_as), len(RL_1.ep_rs), len(RL_1.ep_ss)))
            # print("RL_2.ep_obs: {}, RL_2.ep_as: {}, RL_2.ep_rs: {}, RL_2.ep_ss: {}".format(len(RL_2.ep_obs), len(RL_2.ep_as), len(RL_2.ep_rs), len(RL_2.ep_ss)))
            # print("RL_3.ep_obs: {}, RL_3.ep_as: {}, RL_3.ep_rs: {}, RL_3.ep_ss: {}".format(len(RL_3.ep_obs), len(RL_3.ep_as), len(RL_3.ep_rs), len(RL_3.ep_ss)))
            # print("########")


            time_s = time.time()
            RL_1.learn(epoch_i, thre_entropy, Ifprint=True)
            RL_2.learn(epoch_i, thre_entropy)
            optim_case = RL_3.learn(epoch_i, thre_entropy)
            time_e = time.time()

            
            

        '''
        save checkpoint
        '''
        if (epoch_i % 1000 == 0) & (epoch_i > 1):

            RL_1.save_session(ckpt_path_1)
            RL_2.save_session(ckpt_path_2)
            RL_3.save_session(ckpt_path_3)
            np.savez(np_path, tputs=np.array(RL_1.tput_persisit), candidate=np.array(RL_1.episode), vi_perapp=np.array(RL_1.ss_perapp_persisit), vi_coex=np.array(RL_1.ss_coex_persisit), vi_sum=np.array(RL_1.ss_sum_persisit))


        # if (epoch_i % 30 == 10):
        if (num_sampled % 3000 == 0 and num_sampled > 1):
            print("Number of sampled ep: {}".format(num_sampled))
            """
            optimal range adaptively change
            """
            for class_replay in range(0, 1):
                number_optimal = names['number_optimal_' + str(class_replay)]
                count_size = int(len(number_optimal))

                if (count_size > 300):
                    names['optimal_range_' + str(class_replay)] *= 0.99
                    names['optimal_range_' + str(class_replay)] = max(names['optimal_range_' + str(class_replay)], 1.01)
                    start_location = sum(names['number_optimal_' + str(class_replay)][:-50]) * training_times_per_episode
                    names['observation_optimal_1_' + str(class_replay)] = names['observation_optimal_1_' + str(class_replay)][start_location:]
                    names['action_optimal_1_' + str(class_replay)] = names['action_optimal_1_' + str(class_replay)][start_location:]
                    names['observation_optimal_2_' + str(class_replay)] = names['observation_optimal_2_' + str(class_replay)][start_location:]
                    names['action_optimal_2_' + str(class_replay)] = names['action_optimal_2_' + str(class_replay)][start_location:]
                    names['observation_optimal_3_' + str(class_replay)] = names['observation_optimal_3_' + str(class_replay)][start_location:]
                    names['action_optimal_3_' + str(class_replay)] = names['action_optimal_3_' + str(class_replay)][start_location:]
                    names['number_optimal_' + str(class_replay)] = names['number_optimal_' + str(class_replay)][-50:]
                    names['safety_optimal_1_' + str(class_replay)] = names['safety_optimal_1_' + str(class_replay)][start_location:]
                    names['safety_optimal_2_' + str(class_replay)] = names['safety_optimal_2_' + str(class_replay)][start_location:]
                    names['safety_optimal_3_' + str(class_replay)] = names['safety_optimal_3_' + str(class_replay)][start_location:]
                    names['reward_optimal_1_' + str(class_replay)] = names['reward_optimal_1_' + str(class_replay)][start_location:]
                    names['reward_optimal_2_' + str(class_replay)] = names['reward_optimal_2_' + str(class_replay)][start_location:]
                    names['reward_optimal_3_' + str(class_replay)] = names['reward_optimal_3_' + str(class_replay)][start_location:]

                print("optimal_range:", names['optimal_range_' + str(class_replay)])

            thre_entropy *= 0.5
            thre_entropy = max(thre_entropy, 0.0001)


        print("End2End time epoch_i", epoch_i, time.time() - time_ep_start)
        print("Allocate time epoch_i", epoch_i, time_al_end - time_al_start)
        # print("Process time epoch_i", epoch_i, time_process_ac)
        print("learning time epoch_i:", epoch_i, time_e - time_s)

        epoch_i += 1
        if epoch_i > 500:
            batch_size = 100


def batch_data(NUM_CONTAINERS, NUM_NODES):

    npzfile = np.load("./data/batch_set_cpo_27node_" + str(100) + '.npz')
    # batch_set = npzfile['batch_set']
    batch_set = npzfile['arr_0']
    rnd_array = batch_set[hyper_parameter['batch_C_numbers'], :]
    index_data = []
    for i in range(7):
        index_data.extend([i] * rnd_array[i])

    print(hyper_parameter['batch_C_numbers'])
    print(rnd_array)
    return rnd_array, index_data


def batch_data_sub(rnd_array):

    rnd_array = rnd_array.copy()
    index_data = []
    for i in range(7):
        index_data.extend([i] * int(rnd_array[i]))

    return rnd_array, index_data


def make_path(dirname):

    if not os.path.exists("./checkpoint/" + dirname):
        os.mkdir("./checkpoint/"+ dirname)
        print("Directory ", "./checkpoint/" + dirname, " Created ")
    else:
        print("Directory ", "./checkpoint/" + dirname, " already exists")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_choice', type=int)
    parser.add_argument('--epochs', type=int, default=50000)
    parser.add_argument('--batch_size_tunning', type=int, default=20)
    parser.add_argument('--safety_requirement', type=float, default=0.05)
    parser.add_argument('--recover', action='store_true')
    args = parser.parse_args()

    hyper_parameter['batch_C_numbers'] = args.batch_choice
    params['path'] = "cpo_27_" + str(hyper_parameter['batch_C_numbers'])

    params['batch_size'] = args.batch_size_tunning
    # params['epochs'] = params['epochs'] * (params['batch_size'] / 50)
    params['epochs'] = args.epochs
    params['safety_requirement'] = args.safety_requirement
    params['recover'] = args.recover

    make_path(params['path'])
    train(params)

if __name__ == "__main__":
    main()
