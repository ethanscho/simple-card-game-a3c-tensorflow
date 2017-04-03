# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import random
import time
import sys
import cocos

from game_ac_network import GameACFFNetwork, GameACLSTMNetwork

from simple_game_state import CardGameState

from constants import GAMMA
from constants import LOCAL_T_MAX
from constants import ENTROPY_BETA
from constants import USE_LSTM

ACTION_SIZE = 3
GAME_HISTORY_SIZE = 200.0

class A3CTrainingThread(object):
  def __init__(self,
               thread_index,
               global_network,
               initial_learning_rate,
               learning_rate_input,
               grad_applier,
               max_global_time_step,
               device):

    self.thread_index = thread_index
    self.learning_rate_input = learning_rate_input
    self.max_global_time_step = max_global_time_step

    if USE_LSTM:
      self.local_network = GameACLSTMNetwork(ACTION_SIZE, thread_index, device)
    else:
      self.local_network = GameACFFNetwork(ACTION_SIZE, thread_index, device)

    self.local_network.prepare_loss(ENTROPY_BETA)

    with tf.device(device):
      var_refs = [v._ref() for v in self.local_network.get_vars()]
      self.gradients = tf.gradients(
        self.local_network.total_loss, var_refs,
        gate_gradients=False,
        aggregation_method=None,
        colocate_gradients_with_ops=False)

    self.apply_gradients = grad_applier.apply_gradients(
      global_network.get_vars(),
      self.gradients )
      
    self.sync = self.local_network.sync_from(global_network)
    
    cocos.director.director.init(width=1, height=1)
    self.game_state = CardGameState(113 * thread_index)
    
    self.local_e = 0.0
    self.win_counter = 0.0

    self.local_t = 0

    self.initial_learning_rate = initial_learning_rate

    self.episode_reward = 0

    # variable controling log output
    self.prev_local_t = 0

    self.game_history = list()

  def _anneal_learning_rate(self, global_time_step):
    learning_rate = self.initial_learning_rate * (self.max_global_time_step - global_time_step) / self.max_global_time_step
    if learning_rate < 0.0:
      learning_rate = 0.0
    return learning_rate

  def choose_action(self, pi_values):
    card_prob = np.zeros([len(self.game_state.my_cards[0])])

    # Select best action probability among my cards
    prob_sum = 0.0
    for i in range(0, len(self.game_state.my_cards[0])):
      if self.game_state.my_cards[0][i][0] == 1:
        card_prob[i] = pi_values[i]
        prob_sum += card_prob[i]
      else:
        card_prob[i] = 0.0
    
    card_prob /= prob_sum

    return np.random.choice(range(len(card_prob)), p=card_prob)

  def _record_score(self, sess, summary_writer, summary_op, score_input, score, global_t):
    summary_str = sess.run(summary_op, feed_dict={
      score_input: score
    })
    summary_writer.add_summary(summary_str, global_t)
    summary_writer.flush()
    
  def set_start_time(self, start_time):
    self.start_time = start_time

  def process(self, sess, global_t, summary_writer, summary_op, score_input):
    states = []
    actions = []
    rewards = []
    values = []

    terminal_end = False

    # copy weights from shared to local
    sess.run( self.sync )

    start_local_t = self.local_t

    if USE_LSTM:
      start_lstm_state = self.local_network.lstm_state_out
    
    # t_max times loop
    for i in range(LOCAL_T_MAX):
      pi_, value_ = self.local_network.run_policy_and_value(sess, self.game_state.s_t)

      action = self.choose_action(pi_)

      states.append(self.game_state.s_t)
      actions.append(action)
      values.append(value_)

      # process game
      self.game_state.process(action)
      # receive game result
      reward = self.game_state.reward
      terminal = self.game_state.terminal

      self.episode_reward += reward

      # clip reward
      rewards.append( np.clip(reward, -1, 1) )

      self.local_t += 1

      # s_t1 -> s_t
      self.game_state.update()
      
      if terminal:
        terminal_end = True

        self.local_e += 1.0

        if self.thread_index == 0:
          win_rate = 0.0

          if self.episode_reward == 1:
            self.game_history.append(1)
          else:
            self.game_history.append(0)

          if len(self.game_history) < GAME_HISTORY_SIZE:
            win_rate = np.sum(self.game_history) / float(len(self.game_history)) * 100.0
          else:
            self.game_history.pop(0)
            win_rate = np.sum(self.game_history) / GAME_HISTORY_SIZE * 100.0

          print("Episode {} | Win Rate = {}".format(self.local_e, win_rate))

          self._record_score(sess, summary_writer, summary_op, score_input, win_rate, self.local_e)
          
        self.episode_reward = 0
        self.game_state.reset()

        if USE_LSTM:
          self.local_network.reset_state()
        break
    
    R = 0.0
    if not terminal_end:
      R = self.local_network.run_value(sess, self.game_state.s_t)

    actions.reverse()
    states.reverse()
    rewards.reverse()
    values.reverse()

    batch_si = []
    batch_a = []
    batch_td = []
    batch_R = []

    # compute and accmulate gradients
    for(ai, ri, si, Vi) in zip(actions, rewards, states, values):
      R = ri + GAMMA * R
      td = R - Vi
      a = np.zeros([ACTION_SIZE])
      a[ai] = 1

      batch_si.append(si)
      batch_a.append(a)
      batch_td.append(td)
      batch_R.append(R)

    cur_learning_rate = self._anneal_learning_rate(global_t)

    if USE_LSTM:
      batch_si.reverse()
      batch_a.reverse()
      batch_td.reverse()
      batch_R.reverse()

      sess.run( self.apply_gradients,
                feed_dict = {
                  self.local_network.s: batch_si,
                  self.local_network.a: batch_a,
                  self.local_network.td: batch_td,
                  self.local_network.r: batch_R,
                  self.local_network.initial_lstm_state: start_lstm_state,
                  self.local_network.step_size : [len(batch_a)],
                  self.learning_rate_input: cur_learning_rate } )
    else:
      _, policy_loss = sess.run( [self.apply_gradients, self.local_network.policy_loss] ,
                feed_dict = {
                  self.local_network.s: batch_si,
                  self.local_network.a: batch_a,
                  self.local_network.td: batch_td,
                  self.local_network.r: batch_R,
                  self.learning_rate_input: cur_learning_rate} )

    # return advanced local step size
    diff_local_t = self.local_t - start_local_t
    return diff_local_t
    
