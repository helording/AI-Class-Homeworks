from agent import Agent
import time

maze = '2'

if maze == '1':
    from maze_env1 import Maze
elif maze == '2':
    from maze_env2 import Maze


if __name__ == "__main__":
    ### START CODE HERE ###
    # This is an agent with random policy. You can learn how to interact with the environment through the code below.
    # Then you can delete it and write your own code.

    env = Maze()
    agent = Agent(actions=list(range(env.n_actions)))
    for episode in range(400):
        s = env.reset()
        s.append(False)
        #print("Envrionment reset = : " + str(s))
        episode_reward = 0
        while True:
            env.render()                 # You can comment all render() to turn off the graphical interface in training process to accelerate your code.
            a = agent.choose_action(s, episode)
            s_, r, done = env.step(a)
            #print("AFTER STEP S and S_ = " + str(s) + " " + str(s_))
            agent.UpdateM(s, a, s_, r, episode)
            agent.UpdateQ(s, a, s_, r)
            agent.reduce_noise(episode)
            episode_reward += r
            s = s_

            #time.sleep(0.5)

            #print("\n")
            if done:
                #print(str(s_) + " " + str(s))
                env.render()
                time.sleep(0.5)
                break

        agent.Hallucinate()
        print('episode:', episode, 'episode_reward:', episode_reward)

    ### END CODE HERE ###

    print('\ntraining over\n')
