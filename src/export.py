import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"]="1"
model_name = 'models/best.h5'

model = tf.keras.models.load_model(model_name, compile = False)

inp = tf.keras.layers.Input(shape = (None,None,3))
x = tf.dtypes.cast(inp, tf.float32)
x = x/127.5-1
pr = model(x)

centers = pr[:,:,:,0:1]
centers_mp = tf.nn.max_pool2d(centers, ksize=(3, 3), strides=(1, 1), padding="SAME")


NMS = True
if NMS:
    faces = tf.logical_and(tf.greater(centers, 0.1),tf.equal(centers, centers_mp))
    centers = tf.where(faces)
    print(centers)
    faces = tf.gather_nd(pr, centers[:,:3])
    print(faces)

    centf = tf.cast(centers,tf.float32)
    centx = (centf[:,2] + faces[:,1])*16 
    centy = (centf[:,1] + faces[:,2])*16 
    size = tf.exp(faces[:,3] + 3.5)
    angl = faces[:,4]
    prob = tf.nn.sigmoid(faces[:,0])
    spof = tf.nn.sigmoid(faces[:,5])
    mask = tf.nn.sigmoid(faces[:,6])

    bx = tf.stack([centf[:,0], centx, centy, size, angl, prob, spof, mask, faces[:,0]], axis = 1)
    # boxes = tf.stack([bx[:,1] - bx[:,3],bx[:,2] - bx[:,3],bx[:,1] + bx[:,3],bx[:,2] + bx[:,3]], axis = 1)
    # boxes += bx[:,0:1]*4096
    # nms = tf.image.non_max_suppression(boxes, bx[:,8], 32, iou_threshold=0.2)
    # bx = tf.gather(bx, nms)
    m = tf.keras.Model(inp,bx)
else:
    faces = tf.cast(tf.greater(tf.nn.sigmoid(centers), 0.5), tf.float32) * tf.cast(tf.equal(centers, centers_mp), tf.float32)
    m = tf.keras.Model(inp,pr*faces)

def frozen_keras_graph(model):
  from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2_as_graph
  from tensorflow.lite.python.util import run_graph_optimizations, get_grappler_config

  real_model = tf.function(model).get_concrete_function(tf.TensorSpec(model.inputs[0].shape, model.inputs[0].dtype))
  frozen_func, graph_def = convert_variables_to_constants_v2_as_graph(real_model)

  input_tensors = [
      tensor for tensor in frozen_func.inputs
      if tensor.dtype != tf.resource
  ]
  output_tensors = frozen_func.outputs

  graph_def = run_graph_optimizations(
      graph_def,
      input_tensors,
      output_tensors,
      config=get_grappler_config(["constfold", "function"]),
      graph=frozen_func.graph)
  
  return graph_def
  
m.save(model_name+'.h5')
graph_def = frozen_keras_graph(m)
tf.io.write_graph(graph_def, '.', model_name + '.pb', as_text=False)
