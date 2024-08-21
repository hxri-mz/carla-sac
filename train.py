import os
import sys
import time
import random
import numpy as np
import argparse
import logging
import pickle
import torch
from distutils.util import strtobool
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from algos.ppo.agent import PPOAgent
from algos.sac.agent import SACAgent
from sim.connection import ClientConnection
from sim.environment import CarlaGymEnv
from parameters import *
from utils.transform import TransformObservation
from encoder.zoo import EncoderZoo
import pygame
from encoder.zoo import EncoderZoo
from utils.transform import TransformObservation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp-name', type=str, help='name of the experiment')
    parser.add_argument('--env-name', type=str, default='carla', help='name of the simulation environment')
    parser.add_argument('--learning-rate', type=float, default=PPO_LEARNING_RATE, help='learning rate of the optimizer')
    parser.add_argument('--seed', type=int, default=SEED, help='seed of the experiment')
    parser.add_argument('--chkpt', type=int, default=CHKPT, help='Checkpoint number to continue from')
    parser.add_argument('--total-timesteps', type=int, default=TOTAL_TIMESTEPS, help='total timesteps of the experiment')
    parser.add_argument('--action-std-init', type=float, default=ACTION_STD_INIT, help='initial exploration noise')
    parser.add_argument('--test-timesteps', type=int, default=TEST_TIMESTEPS, help='timesteps to test our model')
    parser.add_argument('--episode-length', type=int, default=EPISODE_LENGTH, help='max timesteps in an episode')
    parser.add_argument('--train', default=True, type=boolean_string, help='is it training?')
    parser.add_argument('--town', type=str, default="Town07", help='which town do you like?')
    parser.add_argument('--load-checkpoint', type=bool, default=MODEL_LOAD, help='resume training?')
    parser.add_argument('--torch-deterministic', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, help='if toggled, `torch.backends.cudnn.deterministic=False`')
    parser.add_argument('--cuda', type=lambda x:bool(strtobool(x)), default=True, nargs='?', const=True, help='if toggled, cuda will not be enabled by deafult')
    args = parser.parse_args()
    
    return args

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'



def runner():

    #========================================================================
    #                           BASIC PARAMETER & LOGGING SETUP
    #========================================================================
    
    args = parse_args()
    exp_name = args.exp_name
    train = args.train
    town = args.town
    checkpoint_load = args.load_checkpoint
    total_timesteps = args.total_timesteps
    action_std_init = args.action_std_init

    try:
        if exp_name == 'ppo':
            run_name = "PPO"
        elif exp_name == 'sac':
            run_name = "SAC"
        else:
            """
            
            Here the functionality can be extended to different algorithms.

            """ 
            sys.exit() 
    except Exception as e:
        print(e.message)
        sys.exit()
    
    if train == True:
        writer = SummaryWriter(f"runs/{run_name}_{action_std_init}_{int(total_timesteps)}/{town}")
    else:
        writer = SummaryWriter(f"runs/{run_name}_{action_std_init}_{int(total_timesteps)}_TEST/{town}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}" for key, value in vars(args).items()])))


    #Seeding to reproduce the results 
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    action_std_decay_rate = 0.05
    min_action_std = 0.05   
    action_std_decay_freq = 5e5
    timestep = 0
    episode = 0
    cumulative_score = 0
    episodic_length = list()
    scores = list()
    deviation_from_center = 0
    distance_covered = 0
    

    #========================================================================
    #                           CREATING THE SIMULATION
    #========================================================================

    try:
        print("Trying to connect")
        client, world = ClientConnection(town).setup()
        logging.info("Connection has been setup successfully.")
    except Exception as e:
        print(e)
        logging.error("Connection has been refused by the server.")
        ConnectionRefusedError
    if train:
        env = CarlaGymEnv(client, world,town)
    else:
        env = CarlaGymEnv(client, world,town, ckpt_freq=None)

    device = torch.device("cpu")
    tfm = TransformObservation(device)
    encoder_zoo = EncoderZoo(CODE_SIZE)
    en_model = encoder_zoo.get_model(ENCODER_MODEL)


    #========================================================================
    #                           ALGORITHM
    #========================================================================
    try:
        time.sleep(0.5)
        if checkpoint_load:
            if args.chkpt:
                chkt_file_nums = args.chkpt
            else:
                chkt_file_nums = len(next(os.walk(f'checkpoints/{run_name}/{town}'))[2]) - 1
            print(chkt_file_nums)
            chkpt_file = f'checkpoints/{run_name}/{town}/checkpoint_{exp_name}_'+str(chkt_file_nums)+'.pickle'
            with open(chkpt_file, 'rb') as f:
                data = pickle.load(f)
                episode = data['episode']
                timestep = data['timestep']
                cumulative_score = data['cumulative_score']
                action_std_init = data['action_std_init']
            agent = PPOAgent(town, action_std_init)
            agent.load()
        else:
            if exp_name == 'ppo':
                if train == False:
                    agent = PPOAgent(town, action_std_init)
                    agent.load()
                    for params in agent.old_policy.actor.parameters():
                        params.requires_grad = False
                else:
                    agent = PPOAgent(town, action_std_init)
            elif exp_name == 'sac':
                print('SAC agent')
                agent = SACAgent(input_dims=env.observation_space,
                                 env=env,
                                 n_actions=env.continous_action_space[0])
                print('Initialized SAC agent')
            else:
                print('No algo found')
        if train:
            #Training
            while timestep < total_timesteps:
            
                observation = env.reset()
                obs = tfm.transform(observation)
                # observation = encoder.transform(observation)
                current_ep_reward = 0
                t1 = datetime.now()

                for t in range(args.episode_length):
                    # select action with policy
                    # action = agent.get_action(observation, train=True)
                    action = agent.action_selection(obs)
                    # print(action)
                    n_observation, reward, done, info = env.step(action)
                    n_obs = tfm.transform(n_observation)
                    if observation is None:
                        break
                    # observation = encoder.transform(observation)
                    # agent.memory.rewards.append(reward)
                    # agent.memory.dones.append(done)
                    # import pdb; pdb.set_trace()
                    agent.memorize(obs, action, reward, n_obs, done)
                    
                    timestep +=1
                    current_ep_reward += reward
                    
                    # if timestep % action_std_decay_freq == 0:
                    #     action_std_init =  agent.decay_action_std(action_std_decay_rate, min_action_std)

                    if timestep == total_timesteps -1:
                        agent.save_models()

                    # break; if the episode is over
                    if done:
                        episode += 1

                        t2 = datetime.now()
                        t3 = t2-t1
                        
                        episodic_length.append(abs(t3.total_seconds()))
                        break
                    
                    obs = n_obs
                
                deviation_from_center += info[1]
                distance_covered += info[0]
                
                scores.append(current_ep_reward)
                
                if checkpoint_load:
                    cumulative_score = ((cumulative_score * (episode - 1)) + current_ep_reward) / (episode)
                else:
                    cumulative_score = np.mean(scores)


                print('Episode: {}'.format(episode),', Timestep: {}'.format(timestep),', Reward:  {:.2f}'.format(current_ep_reward),', Average Reward:  {:.2f}'.format(cumulative_score))
                if episode % 10 == 0:
                    print("\n")
                    agent.learn()
                    agent.save_models()
                    chkt_file_nums = len(next(os.walk(f'checkpoints/{run_name}/{town}'))[2])
                    if chkt_file_nums != 0:
                        chkt_file_nums -=1
                    chkpt_file = f'checkpoints/{run_name}/{town}/checkpoint_{exp_name}_'+str(chkt_file_nums)+'.pickle'
                    data_obj = {'cumulative_score': cumulative_score, 'episode': episode, 'timestep': timestep, 'action_std_init': action_std_init}
                    with open(chkpt_file, 'wb') as handle:
                        pickle.dump(data_obj, handle)
                    
                
                if episode % 5 == 0:

                    writer.add_scalar("Episodic Reward/episode", scores[-1], episode)
                    writer.add_scalar("Cumulative Reward/info", cumulative_score, episode)
                    writer.add_scalar("Cumulative Reward/(t)", cumulative_score, timestep)
                    writer.add_scalar("Average Episodic Reward/info", np.mean(scores[-5]), episode)
                    writer.add_scalar("Average Reward/(t)", np.mean(scores[-5]), timestep)
                    writer.add_scalar("Episode Length (s)/info", np.mean(episodic_length), episode)
                    writer.add_scalar("Reward/(t)", current_ep_reward, timestep)
                    writer.add_scalar("Average Deviation from Center/episode", deviation_from_center/5, episode)
                    writer.add_scalar("Average Deviation from Center/(t)", deviation_from_center/5, timestep)
                    writer.add_scalar("Average Distance Covered (m)/episode", distance_covered/5, episode)
                    writer.add_scalar("Average Distance Covered (m)/(t)", distance_covered/5, timestep)

                    episodic_length = list()
                    deviation_from_center = 0
                    distance_covered = 0

                if episode % 100 == 0:
                    
                    agent.save_models()
                    chkt_file_nums = len(next(os.walk(f'checkpoints/{run_name}/{town}'))[2])
                    chkpt_file = f'checkpoints/{run_name}/{town}/checkpoint_{exp_name}_'+str(chkt_file_nums)+'.pickle'
                    data_obj = {'cumulative_score': cumulative_score, 'episode': episode, 'timestep': timestep, 'action_std_init': action_std_init}
                    with open(chkpt_file, 'wb') as handle:
                        pickle.dump(data_obj, handle)
                        
            print("Terminating the run.")
            sys.exit()
        else:
            #Testing
            while timestep < args.test_timesteps:
                observation = env.reset()
                # observation = encoder.transform(observation)

                current_ep_reward = 0
                t1 = datetime.now()
                for t in range(args.episode_length):
                    # select action with policy
                    action = agent.get_action(observation, train=False)
                    observation, reward, done, info = env.step(action)
                    if observation is None:
                        break
                    # observation = encoder.transform(observation)
                    
                    timestep +=1
                    current_ep_reward += reward
                    # break; if the episode is over
                    if done:
                        episode += 1

                        t2 = datetime.now()
                        t3 = t2-t1
                        
                        episodic_length.append(abs(t3.total_seconds()))
                        break
                deviation_from_center += info[1]
                distance_covered += info[0]
                
                scores.append(current_ep_reward)
                cumulative_score = np.mean(scores)

                print('Episode: {}'.format(episode),', Timestep: {}'.format(timestep),', Reward:  {:.2f}'.format(current_ep_reward),', Average Reward:  {:.2f}'.format(cumulative_score))
                
                writer.add_scalar("TEST: Episodic Reward/episode", scores[-1], episode)
                writer.add_scalar("TEST: Cumulative Reward/info", cumulative_score, episode)
                writer.add_scalar("TEST: Cumulative Reward/(t)", cumulative_score, timestep)
                writer.add_scalar("TEST: Episode Length (s)/info", np.mean(episodic_length), episode)
                writer.add_scalar("TEST: Reward/(t)", current_ep_reward, timestep)
                writer.add_scalar("TEST: Deviation from Center/episode", deviation_from_center, episode)
                writer.add_scalar("TEST: Deviation from Center/(t)", deviation_from_center, timestep)
                writer.add_scalar("TEST: Distance Covered (m)/episode", distance_covered, episode)
                writer.add_scalar("TEST: Distance Covered (m)/(t)", distance_covered, timestep)

                episodic_length = list()
                deviation_from_center = 0
                distance_covered = 0

            print("Terminating the run.")
            sys.exit()
    except Exception as e:
        print(e)
    finally:
        sys.exit()


if __name__ == "__main__":
    try:        
        runner()
    except KeyboardInterrupt:
        sys.exit()
    finally:
        print('\nExit')
