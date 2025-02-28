# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 14:14:51 2018

@author: mengxiaomao
"""
import scipy
import numpy as np
import tensorflow as tf
reuse=tf.AUTO_REUSE
   
class DNN:
    def __init__(self, env, weight_file, max_episode = 5000, INITIAL_EPSILON = 0.2, FINAL_EPSILON = 0.0001):
        self.state_num = env.state_num   #numb of features (50) (refer paper)
        self.action_num = env.power_num          #power levels  (10)
        self.min_p = 5 #dBm          
        self.power_set = env.get_power_set(self.min_p)     #quantisation      
        self.M = env.M   #number of samples
        self.max_episode = max_episode    #num of episodes
        self.weight_file = weight_file    #weights       
        self.INITIAL_EPSILON = INITIAL_EPSILON   #exploration/exploitation
        self.FINAL_EPSILON = FINAL_EPSILON

        self.s = tf.placeholder(tf.float32, [None, self.state_num], name ='s')   #placeholder for state  shape=[500,50]   none is for batch size  (for input state)
        self.a = tf.placeholder(tf.float32, [None, self.action_num], name ='a')     #placeholder for action shape = [500,10]
        self.dqn = self.create_dqn(self.s, 'dqn')  #neural network
        self.dqn_params = self.get_params('dqn')    #gets parameters for dqn
        self.load_dqn_params = self.load_params('dqn')   
        
        self.s_target = tf.placeholder(tf.float32, [None, self.state_num], name ='s_target')   #for target state (fixed for us) (our use?)
        self.dqn_target = self.create_dqn(self.s_target, 'dqn_tar')
        self.dqn_target_params = self.get_params('dqn_tar')
        self.load_dqn_target_params = self.load_params('dqn', True)

    
    def get_dqn_in(self, is_target=False):
        if is_target:
            return self.s_target
        else:
            return self.s      #return state placeholder
            
    def get_action(self, is_target=False):
        if is_target:
            return self.a_target
        else:
            return self.a     #returns action placeholder
            
    def get_dqn_out(self, is_target=False):
        if is_target:
            return self.dqn_target      
        else:
            return self.dqn      #returns created neural network      
        
    def get_dqn_params(self, is_target=False):
        if is_target:
            return self.dqn_target_params
        else:
            return self.dqn_params    
        
    def get_params(self, para_name):
        sets=[]
        for var in tf.trainable_variables():   #w and b
            if not var.name.find(para_name):   #this return true if paraname is present in var name. here paraname is dqn and we create dqn variables with name dqn + numb later in the neural network hence it will return true for w and b in neural networks.
                sets.append(var)
        return sets   
        
    def variable_w(self, shape, name = 'w'):
        w = tf.get_variable(name, shape = shape, initializer = tf.truncated_normal_initializer(stddev=0.1))     #truncated normal dist, cut from either sides
        return w
        
    def variable_b(self, shape, initial = 0.01):
        b = tf.get_variable('b', shape = shape, initializer = tf.constant_initializer(initial))     #constant value  
        return b
        
    def create_dqn(self, s, name):           #neural network creation
        with tf.variable_scope(name + '.0', reuse = reuse):   #here we name all variables as 'dqn' + and numb.
            w = self.variable_w([self.state_num, 128])        #input number for this w = state.num
            b = self.variable_b([128])
            l = tf.nn.relu(tf.matmul(s, w)+b)
        with tf.variable_scope(name + '.1', reuse = reuse):
            w = self.variable_w([128, 64])             
            b = self.variable_b([64])
            l = tf.nn.relu(tf.matmul(l, w) + b)
        with tf.variable_scope(name + '.2', reuse = reuse):
            w = self.variable_w([64, self.action_num])
            b = self.variable_b([self.action_num])
            q_hat = tf.matmul(l, w) + b
        return q_hat
        
    def save_params(self):
        dict_name={}
        for var in tf.trainable_variables():  #from the neural network
            dict_name[var.name]=var.eval()     #dict with variable name and its value
        scipy.io.savemat(self.weight_file, dict_name)  

        
    def load_params(self, name, is_target = False):
        if name == 'dqn':
            if is_target:
                var_list = self.dqn_target_params
            else:
                var_list = self.dqn_params
        try:

            theta = scipy.io.loadmat(self.weight_file)       #load the saved parameters file
            update=[]
            for var in var_list:
#                print(var.name, var.shape)
                print(theta[var.name].shape)
                update.append(tf.assign(tf.get_default_graph().get_tensor_by_name(var.name),tf.constant(np.reshape(theta[var.name],var.shape))))  #updating tensor with new values
        except:
            print('fail dqn')
            update=[]
        return update
        
        
class DQN:
    def __init__(self, sess, dnn, learning_rate = 1e-3):
        self.sess = sess
        self.learning_rate = learning_rate
        self.action_num = dnn.action_num
        self.power_set = dnn.power_set
        self.M = dnn.M
        self.max_episode = dnn.max_episode
        self.INITIAL_EPSILON = dnn.INITIAL_EPSILON
        self.FINAL_EPSILON = dnn.FINAL_EPSILON
        
        self.y = tf.placeholder(tf.float32, [None])
        self.s = dnn.get_dqn_in(is_target=False)   #input tensor for s
        self.a = dnn.get_action(is_target=False)   #action tensor
        self.q_hat = dnn.get_dqn_out(is_target=False)  #output tensor of q values
        self.a_hat = tf.argmax(self.q_hat, 1)        #action with highest q value
        self.params = dnn.get_dqn_params(is_target=False)
        self.load = dnn.load_dqn_params

        self.r = tf.reduce_sum(tf.multiply(self.q_hat, self.a), reduction_indices = 1)      #getting r from q and action
        self.loss = tf.nn.l2_loss(self.y - self.r)          #loss
        with tf.variable_scope('opt_dqn', reuse = reuse):
            self.optimize = tf.train.AdamOptimizer(self.learning_rate).minimize(self.loss, var_list = self.params)

    def train(self, s, a, y):
        return self.sess.run(self.optimize, feed_dict={
            self.s: s, self.a: a, self.y: y})

    def predict_a(self, s):
        q = self.predict_q(s)
        return np.argmax(q, axis = 1)
        
    def predict_p(self, s):
        return self.power_set[self.predict_a(s)] 
        
    def predict_q(self, s):
        return self.sess.run(self.q_hat, feed_dict={self.s: s})
        
    def load_params(self):
        return self.sess.run(self.load)

    def select_action(self, a_hat, episode):
        epsilon = self.INITIAL_EPSILON - episode * (self.INITIAL_EPSILON - self.FINAL_EPSILON) / self.max_episode            # epsilon value dec uniformly
        random_index = np.array(np.random.uniform(size = (self.M)) < epsilon, dtype = np.int32)     #binary array with epsilon probability of 1 and 1-epsilon for 0 (uniform dist bw 0 and 1 so prob of less than epsilon is epsilon)
        random_action = np.random.randint(0, high = self.action_num, size = (self.M))            #choosing random action, M random powers for M users from action_num levels.
        action_set = np.vstack([a_hat, random_action])         #ahat is storing max action for each user, random_action has random action for each user.
        power_index = action_set[random_index, range(self.M)] #if ith element of random_index=1 then power_index will be random power else max (0 pe ahat is set and 1 pe random is set)
        p = self.power_set[power_index] #contains powers for all users in ith action.
        a = np.zeros((self.M, self.action_num), dtype = np.float32)  #[100,10], all set to 0
        a[range(self.M), power_index] = 1.        #chosen user-power pairs set to 1.
        return p, a

    
class DQN_target:
    def __init__(self, sess, dnn, tau = 0.001):
        self.sess = sess
        self.tau = tau    #learning rate?

        self.s = dnn.get_dqn_in(is_target=True)   
        self.out = dnn.get_dqn_out(is_target=True)
        self.params = dnn.get_dqn_params(is_target=True)
        self.params_other = dnn.get_dqn_params(is_target=False)
        self.load = dnn.load_dqn_target_params
        self.update_params = \
            [self.params[i].assign(tf.multiply(self.params_other[i], self.tau) + tf.multiply(self.params[i], 1. - self.tau))
                for i in range(len(self.params))]   #doubt
            
    def train(self):
        self.sess.run(self.update_params)

    def predict_a(self, s):
        q = self.predict_q(s)
        return np.argmax(q, axis = 1)
        
    def predict_q(self, s):
        return self.sess.run(self.out, feed_dict={self.s: s})
        
    def load_params(self):
        return self.sess.run(self.load)
