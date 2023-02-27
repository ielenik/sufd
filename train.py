import tensorflow as tf
import numpy as np
from dataset import getFaceDatasets
from facemodel import createFaceModel
from displaydata import displayData
import time
import wandb
fine_tune = False
if __debug__:
    wandb.init(mode="disabled")
    print('*'*20, "DEGUG MODE", '*'*20)
else:
    if fine_tune:
        wandb.init(project="FaceFindRev_ft")
    else:
        wandb.init(project="FaceFindRev")

def set_batchnorm_momentum(model, m):
    def _set_batchnorm_momentum(layer, m):
        if isinstance(layer, tf.keras.Model):
            for l in layer.layers:
                _set_batchnorm_momentum(l, m)
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.momentum = m
    _set_batchnorm_momentum(model, m)
    return

# strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
# with strategy.scope():

def tf_decorator(func):
    if __debug__:
        return func
    else:
        return tf.function(func)
if True:
    bContinue = False
    # if __debug__:
    #     bContinue = True
    if fine_tune or bContinue:
        model = tf.keras.models.load_model('best.h5')
    else:
        model = createFaceModel()
    train, valid = getFaceDatasets()
    displayData(train)
    # train = strategy.experimental_distribute_dataset(train)
    # valid = strategy.experimental_distribute_dataset(valid)

    @tf_decorator
    def full_loss(pr, tr):
        face_loc = tf.not_equal(tr[:,:,:,1], 0)
        pr_faces = tf.boolean_mask(pr, face_loc)
        tr_faces = tf.boolean_mask(tr, face_loc)

        face_count = tf.shape(tr_faces)[0]

        error_false_neg = tf.reduce_sum(1-tf.math.sigmoid(pr_faces[:,0]))
        #error_false_neg = tf.reduce_sum(tf.nn.relu(1 - pr_faces[:,0]))
        proc_error_false_neg = tf.reduce_sum(tf.cast(tf.greater_equal(0.5,tf.math.sigmoid(pr_faces[:,0])),tf.float32))
        error_false_pos = tf.reduce_sum(tr[:,:,:,0] * tf.math.sigmoid(pr[:,:,:,0]))
        #error_false_pos = tf.reduce_sum(tr[:,:,:,0] * tf.nn.relu(1 + pr[:,:,:,0]))
        proc_error_false_pos = tf.reduce_sum(tf.cast(tf.greater_equal(tr[:,:,:,0]*tf.math.sigmoid(pr[:,:,:,0]),0.5),tf.float32))

        error_size      = tf.reduce_sum(tf.square(3.5 + pr_faces[:,3] - tf.math.log(tr_faces[:,4])))
        error_angle     = tf.reduce_sum(tf.square(pr_faces[:,4] - tr_faces[:,5]))

        def atr_err(tr, pr):
            wh1 = tf.not_equal(tr,-1)
            attr1_p = tf.boolean_mask(pr,wh1)
            attr1_t = tf.boolean_mask(tr,wh1)
            return tf.reduce_sum(tf.square(tf.math.sigmoid(attr1_p) - attr1_t)), tf.shape(attr1_t)[0]

        error_spoof, cnt_spoof     = atr_err(tr_faces[:,6], pr_faces[:,5]) 
        error_mask, cnt_mask      = atr_err(tr_faces[:,7], pr_faces[:,6]) 

        def atr_err_proc(tr, pr):
            wh1 = tf.not_equal(tr,-1)
            attr1_p = tf.boolean_mask(pr,wh1)
            attr1_t = tf.boolean_mask(tr,wh1)
            return tf.reduce_sum(tf.cast(tf.greater_equal(tf.abs(tf.math.sigmoid(attr1_p) - attr1_t),0.5),tf.float32))
        proc_error_spoof  = atr_err_proc(tr_faces[:,6], pr_faces[:,5]) 
        proc_error_mask   = atr_err_proc(tr_faces[:,7], pr_faces[:,6]) 

        error_shift     = tf.reduce_sum(tf.square(pr_faces[:,1] - tr_faces[:,2])) + tf.reduce_sum(tf.square(pr_faces[:,2] - tr_faces[:,3]))
        total_error = (error_size + error_angle*5  + error_shift) + \
            error_false_neg + error_false_pos + \
            error_mask + error_spoof
        # total_error = error_false_neg*10 + error_false_pos
        return total_error, \
            proc_error_false_neg, proc_error_false_pos, error_size, error_angle, proc_error_mask, proc_error_spoof, error_shift, \
            face_count, cnt_mask, cnt_spoof


    @tf_decorator
    def train_on_batch(optimizer, batch):
        with tf.GradientTape() as tape:
            pred = model(batch[0])
            total_loss, fn,fp,sz,ang,ms,sp,sh, cnt, cnt_m, cnt_s = full_loss(pred, batch[1])

            s2_loss = 0
            for w in model.trainable_weights:
                s2_loss += tf.reduce_sum(tf.square(w))
            total_loss += s2_loss*1e-5
            #total_loss = tf.reduce_sum([fn,fp,sz,ang,ms,sp,sh])
        
        grads = tape.gradient(total_loss, model.trainable_variables)
        grads, glnorm = tf.clip_by_global_norm(grads, 5.0)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        return total_loss, fn,fp,sz,ang,ms,sp,sh, cnt, cnt_m, cnt_s

    if fine_tune:
        optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum = 0.5, nesterov = True)
    #optimizer = tf.keras.optimizers.SGD(learning_rate=1e-6, momentum = 0.7, nesterov = True)
    model.compile()

    best_err = 1e6
    t_total_loss = 1e7
    for epoch in range(256):
        total_cnt  = 1e10
        total_step = 257
        sub_step = -1

        for image_batch in train:
            if total_step >= 64:
                # if sub_step >= 1 and sub_step <= 2:
                #     print("Set learning rate to", 1e-6*(10**(sub_step)))
                #     optimizer.learning_rate.assign(1e-6*(10**(sub_step)))
                if sub_step > 1:
                    #t_fp/(total_step*64), t_fn/t_cnt, t_sz/t_cnt, t_ang/t_cnt, t_ms/t_cnt_m, t_sp/t_cnt_s, t_sh/t_cnt), end = '\r')
                    wandb.log(
                        {
                            'total_loss': t_total_loss/(total_step*64), 
                            'fp per image': t_fp/(total_step*64), 
                            'fn per sample': t_fn/t_cnt,
                            'log sz mse': t_sz/t_cnt,
                            'angle mse': t_ang/t_cnt,
                            'shift mse': t_sh/t_cnt,
                            'mask': t_ms/t_cnt_m,
                            'spoof': t_sp/t_cnt_s,
                            }
                        )


                print()
                if best_err > t_total_loss:
                    model.save('best.h5')
                    best_err = t_total_loss
                    print('*** best model')

                momentum = 1 - 10**(-sub_step-2)
                if momentum <= 0.9999:
                    set_batchnorm_momentum(model, momentum)
                    print("Set momentum to", momentum)

                t_total_loss, t_fn, t_fp, t_sz, t_ang, t_ms, t_sp, t_sh, t_cnt, t_cnt_m, t_cnt_s = 0,0,0,0,0,0,0,0,0,0,0
                total_step = 0
                st = time.time()
                sub_step += 1
                model.save('last_model.h5')

            # res = strategy.run(train_on_batch, args=(optimizer, image_batch))
            # total_loss, fn, fp, sz, ang, ms, sp, sh, cnt, cnt_m, cnt_s = [ strategy.reduce(tf.distribute.ReduceOp.SUM, i, axis=None).numpy() for i in res]
            res = train_on_batch(optimizer, image_batch)
            total_loss, fn, fp, sz, ang, ms, sp, sh, cnt, cnt_m, cnt_s = [ i.numpy() for i in res]

            t_total_loss += total_loss
            t_fn += fn
            t_fp += fp
            t_sz += sz
            t_ang += ang
            t_ms += ms
            t_sp += sp
            t_sh += sh
            t_cnt += cnt
            t_cnt_m += cnt_m
            t_cnt_s += cnt_s
            total_step += 1
            print(sub_step, sub_step*64*64+total_step*64,t_cnt, 
                f"%.3fms loss:%.2f fp:%.2f fn:%.2f sz:%.2f ang:%.2f msk:%.2f spf:%.2f shft:%.2f            "%((time.time() - st)/(total_step*64), t_total_loss/(total_step*64),
                t_fp/(total_step*64), t_fn/t_cnt, t_sz/t_cnt, t_ang/t_cnt, t_ms/t_cnt_m, t_sp/t_cnt_s, t_sh/t_cnt), 
                end = '\r'
                )

        print()


