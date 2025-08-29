import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt

policy = {0: 1, 1: 2, 2: 1, 3: 0, 4: 1, 6: 1,
          8: 2, 9: 1, 10: 1, 13: 2, 14: 2}

with_policy = True # 간단한 결정론적 정책 사용 or not

#SLIPPERY = False  # 결정론적 환경
SLIPPERY = True  # 확률적 환경


env = gym.make('FrozenLake-v1', desc=None, map_name="4x4", is_slippery=SLIPPERY)

n_games = 100
win_pct = []
scores = []

# 게임 진행
for i in range(n_games):
    
    terminated, truncated = False, False  
    obs, info = env.reset()
    score = 0
    
    while not terminated and not truncated:

        if with_policy:
            action = policy[obs]
        else: 
            action = env.action_space.sample()  # randomly

        obs, reward, terminated, truncated, info = env.step(action)
        score += reward 

    scores.append(score)

    if i % 10:  
        average = np.mean(scores[-10:]) 
        win_pct.append(average)  

env.close()

# graph
plt.plot(win_pct)
plt.xlabel('episode')
plt.ylabel('success ratio')
plt.title('With Policy: average success ratio of last 10 games\n - {}'
          .format('Stochastic Env' if SLIPPERY else 'Deterministic Env'))
plt.show()
       
    

        



# 그래프 그리기

