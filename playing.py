"""
Once a model is learned, use this to play it.
"""

import carmunk
import numpy as np
from nn import neural_net

NUM_SENSORS = 7
lastState = np.array([14,14,14])
lastaction = 0
def play(model):
    global lastState
    global lastaction
    car_distance = 0
    game_state = carmunk.GameState()

    # Do nothing to get initial.
    _, state = game_state.frame_step((2))
    train_state = np.append(lastState, state[0])
    train_state = np.append(train_state, lastaction)
    train_state = np.expand_dims(train_state, axis=0)
    # Move.
    while True:
        car_distance += 1

        # Choose action.
        action = (np.argmax(model.predict(train_state, batch_size=1)))
        print(action)

        # Take action.
        _, state = game_state.frame_step(action)
        train_state = np.append(lastState, state[0])

        train_state = np.append(train_state, action)
        train_state = np.expand_dims(train_state, axis=0)
        lastState = state[0]
        # Tell us something.
        if car_distance % 1000 == 0:
            print("Current distance: %d frames." % car_distance)


if __name__ == "__main__":
    saved_model = 'saved-models/128-128-400-100000-70000.h5'
    model = neural_net(NUM_SENSORS, [128, 128], saved_model, dropout = True)
    play(model)