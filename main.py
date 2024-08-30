import server
from ppo import PPO, PolicyNetwork, ValueNetwork, Environment
import numpy as np
import logging
from time import sleep
from server import RecvMessageType
from typing import List

logging.basicConfig(level=logging.INFO)

def measure_fn(game: server.TasClient, script):
    # This function will be used by ClientHandlerThread to evaluate the TAS script with given parameters (point).
    # Implement the logic to run the game with the parameters and return a score (reward).
    game.receive()
    game.start_content_playback(script)
    game.receive_until(RecvMessageType.PROCESSED_SCRIPT)
    game.entity_info()
    info: List[server.EntityInfo] = game.receive_until(RecvMessageType.ENTITY_INFO)
    print(info[0])
    return abs(info[0].pitch)

def main():
    with open("template_script.p2tas", "r") as f:
        script_template = f.read()
    param_ranges = np.array([[-89,89], [-180, 180] ,[-180, 180]])

    with open("initscript.p2tas", "r") as f:
        init_script = f.read()

    client_pool = server.ClientPool(
        measure_fn=measure_fn,
        script_template=script_template,
        init_script=init_script
    )

    client_pool.listen(port=12345)

    env = Environment(client_pool, param_ranges)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]

    policy = PolicyNetwork(state_dim, action_dim)
    value = ValueNetwork(state_dim)
    ppo = PPO(
        policy=policy,
        value=value,
        env=env
    )

    try:
        ppo.train(max_episodes=1000)
    except KeyboardInterrupt:
        logging.info("Training interrupted by user.")
    finally:
        client_pool.shutdown()
if __name__ == "__main__":
    main()