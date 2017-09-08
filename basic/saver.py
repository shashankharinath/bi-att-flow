import tensorflow as tf
from tensorflow.python.framework import graph_util
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils


def save_model(checkpoint_path, checkpoint_name, saved_model_path):
    # Restore the model checkpoint
    sess_train = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True))
    saver = tf.train.import_meta_graph(checkpoint_path + '/' + checkpoint_name + '.meta', clear_devices=True)
    saver.restore(sess_train, checkpoint_path + '/' + checkpoint_name)

    # Node names
    output_node_names = "model_0/main/Reshape_15," \
                        "model_0/main/Reshape_17"
    input_node_names = "model_0/emb/word/embedding_lookup:0," \
                       "model_0/emb/char/embedding_lookup:0," \
                       "model_0/x_mask:0," \
                       "model_0/emb/word/embedding_lookup_1:0," \
                       "model_0/emb/char/embedding_lookup_1:0," \
                       "model_0/q_mask:0," \
                       "model_0/train:0 "

    # Get the model graph
    graph_train = tf.get_default_graph()

    # Freeze Graph
    with graph_train.as_default():
        output_graph_def = graph_util.convert_variables_to_constants(
            sess_train,
            graph_train.as_graph_def(add_shapes=True),
            output_node_names.split(","))

    # Create new session with the new graph (all variables converted to constants)
    graph_export = tf.Graph()
    with graph_export.as_default():
        tf.import_graph_def(output_graph_def, name="")

    sess_export = tf.Session(graph=graph_export)
    _build_saved_model(sess_export, saved_model_path)


def _build_saved_model(sess, export_path):
    with sess.graph.as_default():
        # Input nodes
        x = sess.graph.get_tensor_by_name('model_0/emb/word/embedding_lookup:0')  # shape=(?, 8?, 400?, 100)
        cx = sess.graph.get_tensor_by_name('model_0/emb/char/embedding_lookup:0')  # (?, 8?, 400?, 16, 8)
        x_mask = sess.graph.get_tensor_by_name('model_0/x_mask:0')
        q = sess.graph.get_tensor_by_name('model_0/emb/word/embedding_lookup_1:0')  # shape=(?, 30?, 100)
        cq = sess.graph.get_tensor_by_name('model_0/emb/char/embedding_lookup_1:0')  # (?, 30?, 16, 8)
        q_mask = sess.graph.get_tensor_by_name('model_0/q_mask:0')
        is_train = sess.graph.get_tensor_by_name('model_0/train:0')

        # Output  nodes
        yp = sess.graph.get_tensor_by_name('model_0/main/Reshape_15:0')
        yp2 = sess.graph.get_tensor_by_name('model_0/main/Reshape_17:0')

        # Input nodes tensor info
        x_info = utils.build_tensor_info(x)
        cx_info = utils.build_tensor_info(cx)
        x_mask_info = utils.build_tensor_info(x_mask)
        q_info = utils.build_tensor_info(q)
        cq_info = utils.build_tensor_info(cq)
        q_mask_info = utils.build_tensor_info(q_mask)
        is_train_info = utils.build_tensor_info(is_train)

        # Output nodes tensor info
        yp_info = utils.build_tensor_info(yp)
        yp2_info = utils.build_tensor_info(yp2)

        # Saved model input and output map
        inps = {'question_words': q_info, 'context_words': x_info, 'question_chars': cq_info, 'context_chars': cx_info,
                'question_mask': q_mask_info, 'context_mask': x_mask_info, 'train': is_train_info}
        outs = {'yp': yp_info, 'yp2': yp2_info}

        # Saved model prediction signature
        prediction_signature = signature_def_utils.build_signature_def(
            inputs=inps,
            outputs=outs,
            method_name=signature_constants.PREDICT_METHOD_NAME)

        builder = saved_model_builder.SavedModelBuilder(export_path)
        legacy_init_op = tf.group(tf.tables_initializer(), name='legacy_init_op')

        # Add graph and variables to the saved model
        builder.add_meta_graph_and_variables(
            sess, [tag_constants.SERVING],
            clear_devices=True,
            signature_def_map={
                'answer':
                    prediction_signature
            },
            legacy_init_op=legacy_init_op)

        # Save the new model
        builder.save()
