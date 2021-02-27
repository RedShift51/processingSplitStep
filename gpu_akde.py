import tensorflow as tf
import numpy as np
import os
import pickle as pkl

def local(datnpy, grid):

    datnpy = datnpy.astype(np.float32)
    MIN, MAX = [np.min(datnpy)][:][0], [np.max(datnpy)][:][0]
    scaling = MAX - MIN
    MIN -= scaling / 10.
    MAX += scaling / 10.
    datnpy = (datnpy - MIN) / scaling

    gam = 1900
    batch_size = 200000
    det_ = [(0.2 / (len(datnpy)))**(1. / (1. + 4.))][:][0]
    mu0 = datnpy[np.random.randint(0, len(datnpy), size=(gam,))].reshape(-1, 1)
    sig0 = (det_**2) * np.random.uniform(0., 0.5, size=(gam,1))
    w0 = np.random.uniform(0, 0.5, size=(gam,1))
    #w0 /= np.sum(w0)

    tf.compat.v1.reset_default_graph()
    tf.compat.v1.disable_eager_execution()

    global_step = tf.Variable(0, trainable=False)
    learning_rate = tf.compat.v1.train.exponential_decay(0.0002, global_step, \
                                               100, 0.1, staircase=True)
    global_step1 = tf.Variable(0, trainable=False)
    learning_rate1 = tf.compat.v1.train.exponential_decay(0.00001, global_step1, \
                                               2000, 0.1, staircase=True)
    global_step2 = tf.Variable(0, trainable=False)
    learning_rate2 = tf.compat.v1.train.exponential_decay(0.000001, global_step2, \
                                               2000, 0.1, staircase=True)

    def pdf_measure(i, x, mu, std, w, t):
        #(x-tf.reduce_min(x))*
        std1 = tf.compat.v1.clip_by_value(std[i]+t, 0.00001, 1000.)
        #return tf.reduce_sum((1.+0.25*tf.math.abs(x-tf.reduce_mean(x)))*(tf.math.exp(w[i]) / tf.reduce_sum(tf.math.exp(w))) \
        #                     * tf.exp(-tf.math.square(x-mu[i]) / 2. / tf.math.square(std1))\
        #                     / tf.abs(std1) / np.sqrt(2. * np.pi))
        return tf.reduce_sum((0.5+tf.exp(x))*(w[i] / tf.reduce_sum(w)) \
                             * tf.exp(-tf.math.square(x-mu[i]) / 2. / tf.math.square(std1))\
                             / tf.abs(std1) / np.sqrt(2. * np.pi))

        
    def pdf_estimate(mesh, mu, std, w, t):
        ans = tf.zeros(shape=(len(grid),))
        std1 = lambda i1 : tf.compat.v1.clip_by_value(std[i1]+t, 0.00001, 1000.)
        #summarize = lambda a, i: a + (tf.math.exp(w[i]) / tf.reduce_sum(tf.math.exp(w))) \
        #                     * tf.exp(-tf.math.square(mesh-mu[i]) / 2. / tf.math.square(std1(i)))\
        #                     / tf.abs(std1(i)) / np.sqrt(2. * np.pi)
        summarize = lambda a, i: a + (w[i] / tf.reduce_sum(w)) \
                             * tf.exp(-tf.math.square(mesh-mu[i]) / 2. / tf.math.square(std1(i)))\
                             / tf.abs(std1(i)) / np.sqrt(2. * np.pi)
        ans1 = tf.compat.v1.scan(fn=summarize, elems=tf.range(gam), initializer=ans)
        return ans1
            
    src_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(batch_size,1))
    grid_ = tf.compat.v1.placeholder(dtype=tf.float32, shape=(len(grid),))
    mu_ = tf.compat.v1.Variable(initial_value=mu0, dtype=tf.float32)
    sig_ = tf.compat.v1.Variable(initial_value=sig0, dtype=tf.float32)

    # here we will try FF NN for weights
    #w_ = tf.compat.v1.Variable(initial_value=w0, dtype=tf.float32)
    out = tf.compat.v1.layers.Dense(units=gam*2, activation=tf.compat.v1.nn.relu)(\
                tf.compat.v1.concat([tf.reshape(mu_, [1, gam]), tf.reshape(sig_, [1, gam])], axis=-1))
    out = tf.compat.v1.layers.Dense(units=int(gam*1.7), activation=tf.compat.v1.nn.relu)(out)
    #out = tf.compat.v1.layers.Dense(units=int(gam*1.7), activation=tf.compat.v1.nn.relu)(out) 
    out = tf.compat.v1.layers.Dense(units=int(gam*1.5), activation=tf.compat.v1.nn.relu)(out)
    out = tf.compat.v1.layers.Dense(units=int(gam*1.3), activation=tf.compat.v1.nn.relu)(out)
    w_ = tf.compat.v1.layers.Dense(units=gam, activation=tf.compat.v1.nn.softmax)(out)
    w_ = tf.reshape(w_, [gam, 1])

    t_ = tf.compat.v1.Variable(initial_value=0.1, dtype=tf.float32)

    pdfs = lambda x: -tf.math.log(tf.compat.v1.clip_by_value(pdf_measure(x, src_, mu_, sig_, w_, t_), \
                                                            0.0000001, 100000.))
    pdf_ans = pdf_estimate(grid_, mu_, sig_, w_, t_)
    #cut_w = tf.compat.v1.assign(w_, tf.clip_by_value(w_, tf.reduce_mean(w_)-10., tf.reduce_mean(w_)+3.))

    Z = tf.compat.v1.map_fn(pdfs, tf.range(gam), dtype=tf.float32)
    loss = tf.reduce_mean(Z) + 0.01*tf.reduce_sum(w_ * w_)# + \
                #0.01*tf.reduce_sum(tf.math.square(tf.reduce_mean(mu_) - mu_)) + \
                #0.01*tf.reduce_sum(tf.math.square(tf.reduce_mean(sig_) - sig_))
    #tf.reduce_sum(tf.exp(w_)*tf.exp(w_))
    optimize_procedure = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    optimize_procedure1 = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate1).minimize(loss)
    optimize_procedure2 = tf.compat.v1.train.AdamOptimizer(learning_rate=learning_rate2).minimize(loss)

    #optimize_procedure = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(loss)
    #w_renorm = tf.compat.v1.assign(w_, tf.compat.v1.abs(w_)+0.000000001)
    sig_renorm = tf.compat.v1.assign(sig_, tf.compat.v1.abs(sig_)+0.000000001)

    #tf.math.divide(tf.math.exp(w_), tf.compat.v1.reduce_sum(tf.math.exp(w_)))

    with tf.compat.v1.Session() as sess:
        sess.run(tf.compat.v1.global_variables_initializer())
        
        n_iter = 2000
        for k in range(n_iter):
            curr_data = [datnpy[np.random.randint(0, len(datnpy), \
                    size=(batch_size,))].reshape(-1, 1)][:][0]
            #if k == 0:
            #    sess.run(w_renorm)

            l, m, s, w = sess.run([loss, mu_, sig_, w_], feed_dict={src_: curr_data})
            if k < 100:
                sess.run(optimize_procedure, feed_dict={src_: curr_data})
            elif k >= 100 and k < 800:
                sess.run(optimize_procedure1, feed_dict={src_: curr_data})
            else:
                sess.run(optimize_procedure2, feed_dict={src_: curr_data})
            #if k > 3:
                #sess.run(w_renorm)
            #    sess.run(sig_renorm)
            #if k == 1:
            #    1/0
            #sess.run(w_renorm)
            print(k, l)
            pdf_calc = sess.run(pdf_ans, feed_dict={grid_: (grid - MIN) / scaling})

            if k % 300 == 0:
                with open("pdf_gpulog0_" + str(k) + ".pkl", "wb") as f:
                    pkl.dump(pdf_calc / scaling, f)

    with open("pdf_gpulog0.pkl", "wb") as f:
        pkl.dump(pdf_calc / scaling, f)
