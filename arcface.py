import tensorflow as tf

def calculate_arcface_logits(embedding_layer, weights, y, class_num, s, m, epsilon=1e-5):
    embedding_layer = tf.nn.l2_normalize(embedding_layer, axis=1, name='normed_embd')
    weights = tf.nn.l2_normalize(weights, axis=0)
    
    cos_matrix = tf.matmul(embedding_layer, weights)
    cos_matrix = tf.clip_by_value(cos_matrix, -1 + epsilon, 1 - epsilon)
    theta_matrix = tf.acos(cos_matrix)
    margin_cos_matrix = tf.cos(theta_matrix + m)
    margin_cos_matrix = tf.clip_by_value(margin_cos_matrix, -1 + epsilon, 1 - epsilon)
    
    mask = y
    inverse_mask = tf.subtract(1., mask)
    output = tf.add(tf.multiply(cos_matrix, inverse_mask), tf.multiply(margin_cos_matrix, mask))
    output *= s
    return output