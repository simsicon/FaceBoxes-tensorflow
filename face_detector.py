import tensorflow as tf
import numpy as np
import base64
from tensorflow.saved_model import simple_save
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model.signature_def_utils\
    import predict_signature_def

from tensorflow.python.saved_model.tag_constants import SERVING
from tensorflow.python.saved_model.signature_constants\
    import DEFAULT_SERVING_SIGNATURE_DEF_KEY



class FaceDetector:
    def __init__(self, model_path, gpu_memory_fraction=0.25, visible_device_list='0'):
        """
        Arguments:
            model_path: a string, path to a pb file.
            gpu_memory_fraction: a float number.
            visible_device_list: a string.
        """
        with tf.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        nodes = [n.name + ' => ' +  n.op for n in graph_def.node if n.op in ('Placeholder')]
        print(nodes)

        print([node for node in graph_def.node if node.name == 'image_tensor'])

        self.input_image = tf.placeholder(tf.string, shape=(None,), name="input_image")
        input_image_tensor = self.load_base64_tensor(self.input_image)

        tf.import_graph_def(graph_def, {'image_tensor': input_image_tensor})

        print([node for node in graph_def.node if node.name == 'image_tensor'])

        graph = tf.Graph()
        with graph.as_default():
            tf.import_graph_def(graph_def, name='import')

        self.output_ops = {
            "boxes": graph.get_tensor_by_name('import/boxes:0'),
            "scores": graph.get_tensor_by_name('import/scores:0'),
            "num_boxes": graph.get_tensor_by_name('import/num_boxes:0')
        }

        gpu_options = tf.GPUOptions(
            per_process_gpu_memory_fraction=gpu_memory_fraction,
            visible_device_list=visible_device_list
        )
        config_proto = tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False)
        self.sess = tf.Session(graph=graph, config=config_proto)
        self.graph = graph

    def __tf_jpeg_process(self, data):

        # The whole jpeg encode/decode dance is neccessary to generate a result
        # that matches the original model's (caffe) preprocessing
        # (as good as possible)
        image = tf.image.decode_jpeg(data, channels=3,
                                     fancy_upscaling=True,
                                     dct_method="INTEGER_FAST")

        image = tf.image.convert_image_dtype(image, tf.uint8, saturate=True)

        return image

    def load_base64_tensor(self, _input):

        def decode_and_process(base64):
            _bytes = tf.decode_base64(base64)
            _image = self.__tf_jpeg_process(_bytes)

            return _image

        # we have to do some preprocessing with map_fn, since functions like
        # decode_*, resize_images and crop_to_bounding_box do not support
        # processing of batches
        image = tf.map_fn(decode_and_process, _input,
                          back_prop=False, dtype=tf.uint8)

        return image

    def export(self):
        import os
        inputs = {'b64_image': self.input_image}

        export_path = os.path.join(tf.compat.as_bytes("models"), tf.compat.as_bytes(str("1")))

        builder = saved_model_builder.SavedModelBuilder(export_path)

        builder.add_meta_graph_and_variables(
            self.sess, [SERVING],
            signature_def_map={
                DEFAULT_SERVING_SIGNATURE_DEF_KEY: predict_signature_def(
                    inputs=inputs,
                    outputs=self.output_ops
                )
            }
        )

        builder.save()

    def detect(self, b64_image, score_threshold=0.5):
        boxes, scores, num_boxes = self.sess.run(
            self.output_ops, feed_dict={self.input_image: b64_image}
        )

        num_boxes = num_boxes[0]
        boxes = boxes[0][:num_boxes]
        scores = scores[0][:num_boxes]

        to_keep = scores > score_threshold
        boxes = boxes[to_keep]
        scores = scores[to_keep]

        scaler = np.array([h, w, h, w], dtype='float32')
        boxes = boxes * scaler

        return boxes, scores