import numpy as np

import tensorflow as tf
from tensorflow.python.platform import gfile



class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


args = Namespace(
    pb_path = r"/protobuf/model/path.pb",
    input_x = r"X",
    input_y = r"Y",
)

class TFModel:

    def __init__(self, pb_path = None, input_shape = [150, 150, 1], input_node_name = None, output_node_name = None):
        self.pb_path = pb_path
        self.input_node_name = input_node_name
        self.output_node_name = output_node_name
        self.input_shape = input_shape

        self.input_node = None
        self.output_node = None

        self.sess =  tf.Session()
        self.graph = tf.Graph()
        self.input = None

        self.load_pb(self.pb_path)


    def load_pb(self, pb_path):
        '''
        load the trained model in protobuf format "*.pb"
        '''

        print("loading model from protobuf file: {}".format(pb_path))
        with gfile.FastGFile(pb_path) as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            self.sess.graph.as_default()
           

        ##to handle the batch normailzation issue in TensorFlow: https://github.com/tensorflow/tensorflow/issues/3628
        for node in graph_def.node:
            if node.op == 'RefSwitch':
                node.op = 'Switch'
                for index in range(len(node.input)):
                    if 'moving_' in node.input[index]:
                        node.input[index] = node.input[index] + '/read'
            elif node.op == 'AssignSub':
                    node.op = 'Sub'
                    if 'use_locking' in node.attr: del node.attr['use_locking']       


        
        with self.graph.as_default():
            print("input node name: ", self.input_node_name)
            tf.import_graph_def(graph_def, name='')

            node_name = [n.name for n in tf.get_default_graph().as_graph_def().node]
            print(node_name[:10])    

            self.input_node = self.graph.get_tensor_by_name(self.input_node_name)
            self.output_node = self.graph.get_tensor_by_name(self.output_node_name)
            print("input node tensor: {}".format(self.input_node))
            print("output node tensor: {}".format(self.output_node))
            

        #self.sess.run(tf.global_variables_initializer())
        self.graph.finalize()
        print("model loading finished.")

        # Get layer names
        #layers = [op.name for op in self.graph.get_operations()]
        #for layer in layers:
        #    print("layer", layer)
            
            
        print("input node:") 
        #nodes = [n.name + ' => ' +  n.op for n in tf.get_default_graph().as_graph_def().node if n.op in ('Placeholder')]
        nodes = [n.name + ' => ' +  n.op for n in self.graph.as_graph_def().node if n.op in ('Placeholder')]
        for node in nodes:
            print(node)

        self.sess = tf.Session(graph = self.graph)
    

    def evaluate(self, input_data):
        output = self.sess.run(self.output_node, feed_dict = {self.input_node: [input_data]})
        print("output shape: {}".format(output.shape))
        return output

def main():
    
    input_X = np.load(args.input_x)
    input_Y = np.load(args.input_y)
    print(input_X.shape) 

    model = TFModel(pb_path = args.pb_path, input_shape = [150, 150, 1], input_node_name = "model_input:0", output_node_name = "model_output:0")
    output = model.evaluate(input_X[0])
    print(output)



if __name__ == "__main__":
    main()