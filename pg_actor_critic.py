import random
import numpy as np
import tensorflow as tf

from brain import Brain

MEMORY_SIZE = 1000000
BATCH_SIZE = 32
INIT_EPSILON = 1.0
FIN_EPSILON = 0.1
OBSERVE = 100000
EXPLORE = 200000

class PolicyGradientActorCritic(Brain):
  def __init__(self, brain_category):
    self.brain_category = brain_category

    self.session        = tf.Session()
    self.optimizer      = tf.train.RMSPropOptimizer(learning_rate=0.0001, decay=0.9)
    self.summary_writer = tf.summary.FileWriter('/tmp/flower-card/ddpg', self.session.graph, flush_secs=120)

    # training parameters
    self.num_actions     = 96
    self.discount_factor = 0.99
    self.max_gradient    = 5.0
    self.reg_param       = 0.001

    # exploration parameters
    self.epsilon  = INIT_EPSILON

    # counters
    self.train_iteration = 0

    # rollout buffer
    self.state_buffer  = list()
    self.reward_buffer = list()
    self.action_buffer = list()

    # create and initialize variables
    self.states = tf.placeholder(tf.float32, [None, 7, 12, 5], name="states")
    self.create_variables()
    var_lists = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    self.session.run(tf.variables_initializer(var_lists))

    # make sure all variables are initialized
    self.session.run(tf.assert_variables_initialized())

    if self.summary_writer is not None:
      # graph was not available when journalist was created
      self.summary_writer.add_graph(self.session.graph)
      self.summary_every = 150

  def weight_variable(self, shape):
    initial = tf.truncated_normal(shape, stddev = 0.01)
    return tf.Variable(initial)

  def bias_variable(self, shape):
    initial = tf.constant(0.0, shape=shape)
    return tf.Variable(initial)

  def actor_network(self):
    # Convolution
    # Frist
    w_conv1 = tf.get_variable("w_conv1", [5, 5, 5, 128], initializer=tf.random_normal_initializer())
    b_fc1 = tf.get_variable("b_fc1", [128], initializer=tf.constant_initializer(0))
    h_conv1 = tf.nn.relu(tf.nn.conv2d(self.states, w_conv1, [1, 1, 1, 1], padding = "SAME") + b_fc1)

    # 2, 3, 4, 5, 6
    w_conv2 = tf.get_variable("w_conv2", [3, 3, 128, 128], initializer=tf.random_normal_initializer())
    b_fc2 = tf.get_variable("b_fc2", [128], initializer=tf.constant_initializer(0))
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, w_conv2, [1, 1, 1, 1], padding = "SAME") + b_fc2)

    w_conv3 = tf.get_variable("w_conv3", [3, 3, 128, 128], initializer=tf.random_normal_initializer())
    b_fc3 = tf.get_variable("b_fc3", [128], initializer=tf.constant_initializer(0))
    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, w_conv3, [1, 1, 1, 1], padding = "SAME") + b_fc3)

    w_conv4 = tf.get_variable("w_conv4", [3, 3, 128, 128], initializer=tf.random_normal_initializer())
    b_fc4 = tf.get_variable("b_fc4", [128], initializer=tf.constant_initializer(0))
    h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, w_conv4, [1, 1, 1, 1], padding = "SAME") + b_fc4)

    w_conv5 = tf.get_variable("w_conv5", [3, 3, 128, 128], initializer=tf.random_normal_initializer())
    b_fc5 = tf.get_variable("b_fc5", [128], initializer=tf.constant_initializer(0))
    h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, w_conv5, [1, 1, 1, 1], padding = "SAME") + b_fc5)

    w_conv6 = tf.get_variable("w_conv6", [3, 3, 128, 128], initializer=tf.random_normal_initializer())
    b_fc6 = tf.get_variable("b_fc6", [128], initializer=tf.constant_initializer(0))
    h_conv6 = tf.nn.relu(tf.nn.conv2d(h_conv5, w_conv6, [1, 1, 1, 1], padding = "SAME") + b_fc6)

    # Fully connected 
    flat_input = tf.reshape(h_conv6, [-1, 7 * 12 * 128])
    w_fc = tf.get_variable("w_fc", [7 * 12 * 128, 96], initializer=tf.random_normal_initializer())
    b_fc = tf.get_variable("b_fc", [96], initializer=tf.constant_initializer(0))
    output = tf.nn.tanh(tf.matmul(flat_input, w_fc) + b_fc)

    return output

  def critic_network(self):
    # Convolution
    # Frist
    w_conv1 = tf.get_variable("w_conv1", [5, 5, 5, 128], initializer=tf.random_normal_initializer())
    b_fc1 = tf.get_variable("b_fc1", [128], initializer=tf.constant_initializer(0))
    h_conv1 = tf.nn.relu(tf.nn.conv2d(self.states, w_conv1, [1, 1, 1, 1], padding = "SAME") + b_fc1)

    # 2, 3, 4, 5, 6
    w_conv2 = tf.get_variable("w_conv2", [3, 3, 128, 128], initializer=tf.random_normal_initializer())
    b_fc2 = tf.get_variable("b_fc2", [128], initializer=tf.constant_initializer(0))
    h_conv2 = tf.nn.relu(tf.nn.conv2d(h_conv1, w_conv2, [1, 1, 1, 1], padding = "SAME") + b_fc2)

    w_conv3 = tf.get_variable("w_conv3", [3, 3, 128, 128], initializer=tf.random_normal_initializer())
    b_fc3 = tf.get_variable("b_fc3", [128], initializer=tf.constant_initializer(0))
    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_conv2, w_conv3, [1, 1, 1, 1], padding = "SAME") + b_fc3)

    w_conv4 = tf.get_variable("w_conv4", [3, 3, 128, 128], initializer=tf.random_normal_initializer())
    b_fc4 = tf.get_variable("b_fc4", [128], initializer=tf.constant_initializer(0))
    h_conv4 = tf.nn.relu(tf.nn.conv2d(h_conv3, w_conv4, [1, 1, 1, 1], padding = "SAME") + b_fc4)

    w_conv5 = tf.get_variable("w_conv5", [3, 3, 128, 128], initializer=tf.random_normal_initializer())
    b_fc5 = tf.get_variable("b_fc5", [128], initializer=tf.constant_initializer(0))
    h_conv5 = tf.nn.relu(tf.nn.conv2d(h_conv4, w_conv5, [1, 1, 1, 1], padding = "SAME") + b_fc5)

    w_conv6 = tf.get_variable("w_conv6", [3, 3, 128, 128], initializer=tf.random_normal_initializer())
    b_fc6 = tf.get_variable("b_fc6", [128], initializer=tf.constant_initializer(0))
    h_conv6 = tf.nn.relu(tf.nn.conv2d(h_conv5, w_conv6, [1, 1, 1, 1], padding = "SAME") + b_fc6)

    # Fully connected 
    flat_input = tf.reshape(h_conv6, [-1, 7 * 12 * 128])
    w_fc = tf.get_variable("w_fc", [7 * 12 * 128, 1], initializer=tf.random_normal_initializer())
    b_fc = tf.get_variable("b_fc", [1], initializer=tf.constant_initializer(0))
    output = tf.nn.tanh(tf.matmul(flat_input, w_fc) + b_fc)

    return output

  def create_variables(self):
    self.states = tf.placeholder(tf.float32, [None, 7, 12, 5], name="states")

    # rollout action based on current policy
    with tf.name_scope("predict_actions"):
      # initialize actor-critic network
      with tf.variable_scope("actor_network"):
        self.policy_outputs = self.actor_network()
      with tf.variable_scope("critic_network"):
        self.value_outputs = self.critic_network()

      # predict actions from policy network
      self.action_scores = tf.identity(self.policy_outputs, name="action_scores")
      # Note 1: tf.multinomial is not good enough to use yet
      # so we don't use self.predicted_actions for now
      self.predicted_actions = tf.multinomial(self.action_scores, 1)

    # get variable list
    actor_network_variables  = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="actor_network")
    critic_network_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="critic_network")

    # compute loss and gradients
    with tf.name_scope("compute_pg_gradients"):
      # gradients for selecting action from policy network
      self.taken_actions = tf.placeholder(tf.int32, (None,), name="taken_actions")
      self.discounted_rewards = tf.placeholder(tf.float32, (None,), name="discounted_rewards")

      with tf.variable_scope("actor_network", reuse=True):
        self.logprobs = self.actor_network()

      with tf.variable_scope("critic_network", reuse=True):
        self.estimated_values = self.critic_network()

      # compute policy loss and regularization loss
      self.cross_entropy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self.logprobs, labels=self.taken_actions)
      self.pg_loss            = tf.reduce_mean(self.cross_entropy_loss)
      self.actor_reg_loss     = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in actor_network_variables])
      self.actor_loss         = self.pg_loss + self.reg_param * self.actor_reg_loss

      # compute actor gradients
      self.actor_gradients = self.optimizer.compute_gradients(self.actor_loss, actor_network_variables)
      # compute advantages A(s) = R - V(s)
      self.advantages = tf.reduce_sum(self.discounted_rewards - self.estimated_values)
      # compute policy gradients
      for i, (grad, var) in enumerate(self.actor_gradients):
        if grad is not None:
          self.actor_gradients[i] = (grad * self.advantages, var)

      # compute critic gradients
      self.mean_square_loss = tf.reduce_mean(tf.square(self.discounted_rewards - self.estimated_values))
      self.critic_reg_loss  = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in critic_network_variables])
      self.critic_loss      = self.mean_square_loss + self.reg_param * self.critic_reg_loss
      self.critic_gradients = self.optimizer.compute_gradients(self.critic_loss, critic_network_variables)

      # collect all gradients
      self.gradients = self.actor_gradients + self.critic_gradients

      # clip gradients
      for i, (grad, var) in enumerate(self.gradients):
        # clip gradients by norm
        if grad is not None:
          self.gradients[i] = (tf.clip_by_norm(grad, self.max_gradient), var)

      # emit summaries
      tf.summary.histogram("estimated_values", self.estimated_values)
      tf.summary.scalar("actor_loss", self.actor_loss)
      tf.summary.scalar("critic_loss", self.critic_loss)
      tf.summary.scalar("reg_loss", self.actor_reg_loss + self.critic_reg_loss)

    # training update
    with tf.name_scope("train_actor_critic"):
      # apply gradients to update actor network
      self.train_op = self.optimizer.apply_gradients(self.gradients)

  def get_action(self, current_state, state):
    def softmax(y):
      """ simple helper function here that takes unnormalized logprobs """
      maxy = np.amax(y)
      e = np.exp(y - maxy)
      return e / np.sum(e)

    if self.epsilon > FIN_EPSILON and state.t > OBSERVE:
      self.epsilon -= (INIT_EPSILON - FIN_EPSILON) / EXPLORE

    # epsilon-greedy exploration strategy
    if np.random.rand(1) < self.epsilon:
      action = random.randrange(len(state.current_player.cards))
    else:
      action_dist = self.session.run(self.action_scores, feed_dict={self.states: [current_state]})
      action_probs  = softmax(action_dist) - 1e-5
      
      # Select best action probability among my cards
      i = 0
      max_prob = 0.0
      action = 0 # action index for the next step. Action range is from 0 to length(cards) - 1

      for card in state.current_player.cards:
        index = card.get_index()

        if max_prob < action_probs[0][index]:
          action = i
          max_prob = action_probs[0][index]

        i += 1

    return action

  def store_experience(self, state, action, reward):
    self.action_buffer.append(action)
    self.reward_buffer.append(reward)
    self.state_buffer.append(state)

    if len(self.action_buffer) > MEMORY_SIZE:
      self.action_buffer.pop(0)
      self.reward_buffer.pop(0)
      self.state_buffer.pop(0)

  def train(self, state, current_state, action_to_store, reward, next_state, done):
    # Store first
    self.store_experience(current_state, action_to_store, reward)

    if state.t < OBSERVE:
      return

    if done:
      if len(self.action_buffer) < BATCH_SIZE:
        N = len(self.action_buffer)
      else:
        N = BATCH_SIZE

      r = 0 # use discounted reward to approximate Q value

      # compute discounted future rewards
      discounted_rewards = np.zeros(N)
      for t in reversed(xrange(N)):
        # future discounted reward from now on
        r = self.reward_buffer[t] + self.discount_factor * r
        discounted_rewards[t] = r

      # update policy network with the rollout in batches
      for t in xrange(N-1):

        # prepare inputs
        states  = self.state_buffer[t][np.newaxis, :]
        actions = np.array([self.action_buffer[t]])
        rewards = np.array([discounted_rewards[t]])

        # perform one update of training
        self.session.run(
          self.train_op, {
          self.states:             states,
          self.taken_actions:      actions,
          self.discounted_rewards: rewards
        })

      self.train_iteration += 1

  def write_summary(self, win_ratio, episode):
    pass
