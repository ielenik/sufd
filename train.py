import tensorflow as tf
import numpy as np
import time
import wandb
import importlib

from src.dataset import getFaceDatasets
from src.facemodel import createFaceModel
from src.displaydata import displayData
import src.config as conf


if __debug__:
    wandb.init(mode="disabled")
    print('*'*20, "DEGUG MODE", '*'*20)
else:
    wandb.init(project="FaceFindRev", resume = conf.RESUME_TRAIN)

def set_batchnorm_momentum(model, m):
    def _set_batchnorm_momentum(layer, m):
        if isinstance(layer, tf.keras.Model):
            for l in layer.layers:
                _set_batchnorm_momentum(l, m)
        elif isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.momentum = m
    _set_batchnorm_momentum(model, m)
    return


def tf_decorator(func):
    if __debug__:
        return func
    else:
        return tf.function(func)

# strategy = tf.distribute.MirroredStrategy(cross_device_ops=tf.distribute.HierarchicalCopyAllReduce())
# with strategy.scope():
if True:
    bContinue = False

    if conf.RESUME_TRAIN:
        model = tf.keras.models.load_model(conf.BEST_MODEL_NAME)
    else:
        model = createFaceModel()

    train, valid = getFaceDatasets()

    if conf.OUTPUT_PREVIEW:
        displayData(train)
    # train = strategy.experimental_distribute_dataset(train)
    # valid = strategy.experimental_distribute_dataset(valid)

    @tf_decorator
    def full_loss(pr, tr):
        face_loc = tf.not_equal(tr[:,:,:,1], 0)
        pr_faces = tf.boolean_mask(pr, face_loc)
        tr_faces = tf.boolean_mask(tr, face_loc)

        face_count = tf.shape(tr_faces)[0]

        error_false_neg = tf.reduce_sum(-tf.math.log(tf.math.sigmoid(pr_faces[:,0]) + 1e-4))
        #error_false_neg = tf.reduce_sum(tf.nn.relu(1 - pr_faces[:,0]))
        proc_error_false_neg = tf.reduce_sum(tf.cast(tf.greater_equal(0.5,tf.math.sigmoid(pr_faces[:,0])),tf.float32))
        error_false_pos = tf.reduce_sum(tr[:,:,:,0] * -tf.math.log(1-tf.math.sigmoid(pr[:,:,:,0]) + 1e-4))
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

    if conf.USE_ADAM_OPTIMIZER:
        optimizer = tf.keras.optimizers.Adam(learning_rate=conf.LEARNING_RATE)
    else:
        optimizer = tf.keras.optimizers.SGD(learning_rate=conf.LEARNING_RATE, momentum = conf.MOMENTUM, nesterov = conf.NESTEROV)
    
    model.compile()
    
    best_err = 1e6
    t_total_loss = 1e7
    step_at_epoch = conf.STEPS_PER_EPOCH
    total_steps = -1

    for image_batch in train:
        if step_at_epoch >= conf.STEPS_PER_EPOCH:
            # if sub_step >= 1 and sub_step <= 2:
            #     print("Set learning rate to", 1e-6*(10**(sub_step)))
            #     optimizer.learning_rate.assign(1e-6*(10**(sub_step)))
            if total_steps > 1:
                #t_fp/(total_step*64), t_fn/t_cnt, t_sz/t_cnt, t_ang/t_cnt, t_ms/t_cnt_m, t_sp/t_cnt_s, t_sh/t_cnt), end = '\r')
                wandb.log(
                    {
                        'total_loss': t_total_loss/(step_at_epoch*conf.BATCH_SIZE), 
                        'fp per image': t_fp/(step_at_epoch*conf.BATCH_SIZE), 
                        'fn per sample': t_fn/t_cnt,
                        'log sz mse': t_sz/t_cnt,
                        'angle mse': t_ang/t_cnt,
                        'shift mse': t_sh/t_cnt,
                        'mask': t_ms/t_cnt_m,
                        'spoof': t_sp/t_cnt_s,
                        }
                    )

            print()
            prev_lr = conf.LEARNING_RATE
            importlib.reload(conf)
            if prev_lr != conf.LEARNING_RATE:
                print("Set learning rate to", conf.LEARNING_RATE)
                optimizer.learning_rate.assign(conf.LEARNING_RATE)

            if best_err > t_total_loss:
                model.save('best.h5')
                best_err = t_total_loss
                print('*** best model')

            momentum = 1 - 10**(-total_steps-2)
            if momentum <= 0.9999:
                set_batchnorm_momentum(model, momentum)
                print("Set momentum to", momentum)

            t_total_loss, t_fn, t_fp, t_sz, t_ang, t_ms, t_sp, t_sh, t_cnt, t_cnt_m, t_cnt_s = 0,0,0,0,0,0,0,0,0,0,0
            step_at_epoch = 0
            st = time.time()
            total_steps += 1
            model.save(conf.LAST_MODEL_NAME)

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
        step_at_epoch += 1
        print(total_steps, total_steps*conf.BATCH_SIZE*conf.STEPS_PER_EPOCH+step_at_epoch*conf.BATCH_SIZE,t_cnt, 
            f"%.3fms loss:%.2f fp:%.2f fn:%.2f sz:%.2f ang:%.2f msk:%.2f spf:%.2f shft:%.2f            "%((time.time() - st)/(step_at_epoch*conf.BATCH_SIZE), t_total_loss/(step_at_epoch*conf.BATCH_SIZE),
            t_fp/(step_at_epoch*conf.BATCH_SIZE), t_fn/t_cnt, t_sz/t_cnt, t_ang/t_cnt, t_ms/t_cnt_m, t_sp/t_cnt_s, t_sh/t_cnt), 
            end = '\r'
            )

