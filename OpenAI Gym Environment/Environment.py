from gym import Env
from gym.spaces import Box
from gym import spaces
import numpy as np
import random
import torch
from pv import PV_generator
from wind import WindTurbineGenerator
from load import load_generator
class Microgrid(Env):
    def __init__(self):

        self.action_space = spaces.MultiDiscrete([3, 3, 3])

        # SoC1, SoC2, SoC3, BC1, BC2, BC3, PV1, PV2, PV3, WT1, WT2, WT3, Load, FLoad, Price, Fprice, mg1, mg2, mg3, ac
        self.observation_space = Box(low=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
                                     high=np.array(
                                         [1, 1, 1, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf,
                                          np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf, np.inf]))
        self.state = np.zeros(20)
        self.current_step = 0 
        self.steps = 72
        self.pv1, self.pv2, self.pv3 = PV_generator.pv()
        self.wt1, self.wt2, self.wt3 = WindTurbineGenerator.wind_turbine()
        self.load = load_generator.load()
        

    def step(self, action):
        self.current_step += 1
        # Unpack the state
        steps = 72
        s = steps*0.75
        soc1, soc2, soc3 = self.state[0], self.state[1], self.state[2]
        bc1, bc2, bc3 = self.state[3], self.state[4], self.state[5]

        # PV
        pv1, pv2, pv3 = (self.pv1[self.current_step % steps], self.pv2[self.current_step % steps],
                         self.pv3[self.current_step % steps])  
        self.state[6], self.state[7], self.state[8] = (pv1/100), (pv2/100), (pv3/100)

        # Wind 
        wt1, wt2, wt3 = (self.wt1[self.current_step % steps], self.wt2[self.current_step % steps],
                         self.wt3[self.current_step % steps])
        self.state[9], self.state[10], self.state[11] = (wt1/100), (wt2/100), (wt3/100)

        # Load
        load = (self.load[self.current_step % steps])/100
        fload = load + ((np.random.choice([-4, 4], 1)+ np.random.normal(0, 2))/100)
        self.state[12], self.state[13] = load, fload

        # Price
        price = (np.random.uniform(7, 45)/100) #price
        fprice = price + ((np.random.choice([-4, 4], 1)+ np.random.normal(0, 2))/100)
        self.state[14], self.state[15] = price, fprice
        
        #microgrid
        mg1 = (pv1 + wt1)/100
        mg2 = (pv2 + wt2)/100
        mg3 = (pv3 + wt3)/100

        # Apply actions
        for i, act in enumerate(action):
            if act == 0:  # Discharge
                if i == 0:
                    soc1 -= 0.06
                    tc1 = bc1 - (soc1 * bc1)
                elif i == 1:
                    soc2 -= 0.02
                    tc2 = bc2 - (soc2 * bc2)
                    mg2 = mg2 - tc2
                elif i == 2:
                    soc3 -= 0.03
                    tc3 = bc3 - (soc3 * bc3)
                    mg3 = mg3 - tc3
            elif act == 1:  # Charge
                if i == 0:
                    soc1 += 0.06
                    tc1 = bc1 + (soc1 * bc1)
                    mg1 = mg1 + tc1
                elif i == 1:
                    soc2 += 0.02
                    tc2 = bc2 + (soc2 * bc2)
                    mg2 = mg2 + tc2
                elif i == 2:
                    soc3 += 0.03
                    tc3 = bc3 + (soc3 * bc3)
                    mg3 = mg3 + tc3
            elif act == 2:  # Do nothing (idle)
                pass

        # Update state
        soc1 = max(0, min(1, soc1))
        soc2 = max(0, min(1, soc2))
        soc3 = max(0, min(1, soc3))
        mg1 = max(0, min(1, mg1))
        mg2 = max(0, min(1, mg2))
        mg3 = max(0, min(1, mg3))

        self.state[0], self.state[1], self.state[2] = soc1, soc2, soc3
        self.state[16], self.state[17], self.state[18] = mg1, mg2, mg3

        reward = self.calculate_reward(self.state, action,s)

        done = self.current_step >= self.steps
        info = {"TimeLimit.truncated": done}
        return  self.state, reward, np.array([done]), info

    def calculate_reward(self, state, action, s):
        soc1, soc2, soc3 = state[0], state[1], state[2]
        mg1, mg2, mg3 = state[16], state[17], state[18]
        load = state[12]
        renewable = mg1 + mg2 + mg3
        price = state[14]
        fprice = state[15]
        fload = state[13]
        self.state[19] = load - renewable
        if self.state[19] <=0:
            self.state[19] = 0
        else:
            self.state[19]=self.state[19]
        ac = self.state[19]

        # Calculate reward
        reward = 0
        mr = 0.22
        lr = 0.11
        hr = 0.33

        # based on time
        if (s < 3 or s > 22) and any(action == 0 for action in action):
            reward += lr
        
        # based on soc
        for soc, act in zip([soc1, soc2, soc3], action):
            if (act == 1 and soc >= 0.9) or (act == 0 and soc <= 0.2):
                reward -= hr * 3
            elif (act == 0 and soc >= 0.9) or (act == 1 and soc <= 0.2):
                reward += hr*3

        # based on mg
        for mg, act in zip([mg1, mg2, mg3], action):
            if (act == 0 and (mg <= 0.05)):
                reward += hr
        
        # based on renewable energy
        if any(act == 1 for act in action) and renewable > 0.4 * load:
            reward += mr
        elif any(act == 0 for act in action) and renewable > 0.4 * load:
            reward -= lr

        # based on non-renewable energy

        if any(act == 0 for act in action) and ac > 0.75 * load:
            reward += mr

        # based on price
        for act in action:
            if act == 0 and price > fprice:
                reward += lr / 3
            elif act == 1 and price < fprice:
                reward += lr / 3
            elif act == 1 and price > fprice:
                reward -= lr / 4
            elif act == 0 and price < fprice:
                reward -= lr / 4

        # base on load demand
        for act in action:
            if act == 0 and load > fload and renewable < 0.4 * load:
                reward += mr
            elif act == 1 and load < fload and renewable > 0.4 * load:
                reward += mr
        return reward

    def reset(self):
        self.pv1, self.pv2, self.pv3 = PV_generator.pv()
        self.wt1, self.wt2, self.wt3 = WindTurbineGenerator.wind_turbine()
        self.load = load_generator.load()
        self.state[0], self.state[1], self.state[2] = ((random.uniform(0.1, 0.9)), (random.uniform(0.1, 0.9)), (random.uniform(0.1, 0.9)))
        self.state[3], self.state[4], self.state[5] = ((random.uniform(1, 20) / 100), (random.uniform(1, 20) / 100), (random.uniform(1, 20)) / 100)
        self.state[6], self.state[7], self.state[8] = ((random.uniform(0, 7)/100), (random.uniform(0, 9)/100),
                                                       (random.uniform(0, 6))/100) # pv1, pv2, pv3
        self.state[9], self.state[10], self.state[11] = ((random.uniform(0.1, 7)/100), (random.uniform(0.1, 13)/100),
                                                         (random.uniform(0.1, 12))/100) # wt1, wt2, wt3
        self.state[12] = random.uniform(40, 100)/100 #load
        self.state[13] = self.state[12] + (random.randint(-2, 2)/100) #fload
        self.state[14] = random.uniform(7, 45)/100 #price
        self.state[15] = self.state[14] + (random.randint(-2, 2)/100) #fprice
        self.state[16] = self.state[6] + self.state[9] # mg1
        self.state[17] = self.state[7] + self.state[10] # mg2
        self.state[18] = self.state[8] + self.state[11] # mg3
        self.state[19] = self.state[12] - (self.state[16]+self.state[17]+self.state[18])
        if self.state[19] <=0:
            self.state[19] = 0
        else:
            self.state[19]=self.state[19]
        self.current_step = 0
        return torch.tensor(self.state)