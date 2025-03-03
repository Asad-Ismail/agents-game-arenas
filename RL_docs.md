
### Q(action value) update

def calculate_loss(q_network, state, action, next_state, reward, done):
    q_values = q_network(state)
    print(f'Q-values: {q_values}')
    # Obtain the current state Q-value
    current_state_q_value = q_values[action]
    print(f'Current state Q-value: {current_state_q_value:.2f}')
    # Obtain the next state Q-value
    next_state_q_value = q_network(next_state).argmax()  
    print(f'Next state Q-value: {next_state_q_value:.2f}')
    # Calculate the target Q-value
    target_q_value = reward + gamma * next_state_q_value * (1-done)
    print(f'Target Q-value: {target_q_value:.2f}')
    # Obtain the loss
    loss = nn.MSELoss()(target_q_value, current_state_q_value)
    print(f'Loss: {loss:.2f}')
    return loss


## Main training loop for DQL
for episode in range(10):
    state, info = env.reset()
    done = False
    step = 0
    episode_reward = 0
    while not done:
        step += 1     
        # Select the action
        action = select_action(q_network, state)
        next_state, reward, terminated, truncated, _ = (env.step(action))
        done = terminated or truncated
        # Calculate the loss
        loss = calculate_loss(q_network, state, action, next_state, reward, done)
        optimizer.zero_grad()
        # Perform a gradient descent step
        loss.backward()
        optimizer.step()
        state = next_state
        episode_reward += reward
    describe_episode(episode, reward, episode_reward, step)
