from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math
import time
from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import
import tensorflow as tf
from tensorflow.compat.v1.gfile import Exists as exists
import model
import data_utils
# import tpu_estimator

import numpy as np
from time import sleep



# init flag
flags.DEFINE_enum("init", default="normal",
      enum_values=["normal", "uniform"],
      help="Initialization method.")
flags.DEFINE_float("init_range", default=0.1,
      help="Initialization std when init is uniform.")
flags.DEFINE_float("init_std", default=0.02,
      help="Initialization std when init is normal.")
flags.DEFINE_float("proj_init_std", default=0.01,
      help="Initialization std for embedding projection.")
flags.DEFINE_bool("proj_share_all_but_first", default=False,
      help="True to share all but first projs, False not to share.")

# Model paramenters
flags.DEFINE_integer("tgt_len", default=128,
      help="Number of steps to predict")
flags.DEFINE_integer("mem_len", default=3800,
      help="Number of steps to cache")
flags.DEFINE_bool("same_length", default=True,
      help="Same length attention")
flags.DEFINE_integer("clamp_len", default=1000,
      help="Clamp length")
flags.DEFINE_integer("n_layer", default=24,
      help="Number of layers.")
flags.DEFINE_integer("d_model", default=1024,
      help="Dimension of the model.")
flags.DEFINE_integer("d_embed", default=1024,
      help="Dimension of the embeddings.")
flags.DEFINE_integer("n_head", default=8,
      help="Number of attention heads.")
flags.DEFINE_integer("d_head", default=128,
      help="Dimension of each attention head.")
flags.DEFINE_integer("d_inner", default=3072,
      help="Dimension of inner hidden size in positionwise feed-forward.")
flags.DEFINE_float("dropout", default=0.15,
      help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.15,
      help="Attention dropout rate.")
flags.DEFINE_bool("untie_r", default=False,
      help="untie r_w_bias and r_r_bias")

# Adaptive Softmax / Embedding
flags.DEFINE_bool("tie_weight", default=True,
      help="Tie embedding and softmax weight.")
flags.DEFINE_integer("div_val", default=1,
      help="Divide the embedding size by this val for each bin")
flags.DEFINE_bool("proj_share_all_but_first", default=False,
      help="True to share all but first projs, False not to share.")
flags.DEFINE_bool("proj_same_dim", default=True,
      help="Project the bin with the same dimension.")

# for cosine decay
flags.DEFINE_float("min_lr_ratio", default=0.01,
      help="Minimum ratio learning rate.")
flags.DEFINE_integer("warmup_steps", default=0,
      help="Number of steps for linear lr warmup.")

# Training parameters
flags.DEFINE_integer("train_batch_size", default=32,
      help="Size of train batch.")
flags.DEFINE_integer("eval_batch_size", default=16,
      help="Size of valid batch.")
flags.DEFINE_integer("train_steps", default=100,
      help="Total number of training steps.")
flags.DEFINE_integer("iterations", default=10,
      help="Number of iterations per repeat loop.")

# Optimization paramenters
flags.DEFINE_float("learning_rate", default=2.5e-4,
      help="Maximum learning rate.")
flags.DEFINE_float("clip", default=0.25,
      help="Gradient clipping value.")

# Experiment (data/checkpoint/directory) parameters
flags.DEFINE_string("data_dir", default="/root/cyliu/tftuner/selftf/tf_job/nlp/zmwu/transformer_xl/data/processed_enwik8/enwik8-tfrecords",
      help="Path to tf-records directory.")
flags.DEFINE_string("record_info_dir", default="/root/cyliu/tftuner/selftf/tf_job/nlp/zmwu/transformer_xl/data/enwik8/tfrecords/",
      help="Path to local directory containing filenames.txt.")
flags.DEFINE_string("corpus_info_path", default="/root/cyliu/tftuner/selftf/tf_job/nlp/zmwu/transformer_xl/data/enwik8/corpus-info.json",
      help="Path to corpus-info.json file.")
flags.DEFINE_string("model_dir", default="/root/cyliu/tftuner/selftf/tf_job/nlp/zmwu/transformer_xl/",
      help="Estimator model_dir.")


flags.DEFINE_bool("use_tpu", default=False,
      help="Use TPUs rather than plain CPUs.")
flags.DEFINE_integer("num_hosts", default=1,
      help="number of TPU hosts")
flags.DEFINE_integer("num_core_per_host", default=8,
      help="number of cores per host")

def get_model_fn(n_token, cutoffs, train_bin_sizes, eval_bin_sizes):
  def model_fn(features, labels, mode, params):
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)


    batch_size = params["batch_size"]

    mems = params["cache"]
    inp = tf.transpose(features["inputs"], [1, 0])
    tgt = tf.transpose(features["labels"], [1, 0])

    bin_sizes = train_bin_sizes if is_training else eval_bin_sizes
    if bin_sizes:
      inp_perms = [tf.transpose(features["inp_mask"], [1, 0])]
      tgt_perms = [tf.transpose(features["tgt_mask"], [1, 0])]

      head_tgt = tf.transpose(features["head_labels"], [1, 0])

      for b in range(len(bin_sizes)):
        inp_perm = tf.transpose(features["inp_perm_{}".format(b)], [1, 0, 2])
        tgt_perm = tf.transpose(features["tgt_perm_{}".format(b)], [1, 0, 2])

        inp_perms.append(inp_perm)
        tgt_perms.append(tgt_perm)
    else:
      inp_perms, tgt_perms, head_tgt = None, None, None

    if FLAGS.init == "uniform":
      initializer = tf.initializers.random_uniform(
          minval=-FLAGS.init_range,
          maxval=FLAGS.init_range,
          seed=None)
    elif FLAGS.init == "normal":
      initializer = tf.initializers.random_normal(
          stddev=FLAGS.init_std,
          seed=None)
      proj_initializer = tf.initializers.random_normal(
          stddev=FLAGS.proj_init_std,
          seed=None)

    tie_projs = [False for _ in range(len(cutoffs) + 1)]
    if FLAGS.proj_share_all_but_first:
      for i in range(1, len(tie_projs)):
        tie_projs[i] = True

    tf.logging.info("Vocab size : {}".format(n_token))
    tf.logging.info("Batch size : {}".format(batch_size))

    loss, new_mems = model.transformer(
        dec_inp=inp,
        target=tgt,
        mems=mems,
        n_token=n_token,
        n_layer=FLAGS.n_layer,
        d_model=FLAGS.d_model,
        d_embed=FLAGS.d_embed,
        n_head=FLAGS.n_head,
        d_head=FLAGS.d_head,
        d_inner=FLAGS.d_inner,
        dropout=FLAGS.dropout,
        dropatt=FLAGS.dropatt,
        initializer=initializer,
        is_training=is_training,
        mem_len=FLAGS.mem_len,
        cutoffs=cutoffs,
        div_val=FLAGS.div_val,
        tie_projs=tie_projs,
        input_perms=inp_perms,
        target_perms=tgt_perms,
        head_target=head_tgt,
        same_length=FLAGS.same_length,
        clamp_len=FLAGS.clamp_len,
        use_tpu=FLAGS.use_tpu,
        untie_r=FLAGS.untie_r,
        proj_same_dim=FLAGS.proj_same_dim)

    total_loss = tf.reduce_mean(loss)

    if mode == tf.estimator.ModeKeys.EVAL:
      if FLAGS.use_tpu:
        with tf.colocate_with(total_loss):
          total_loss = tf.contrib.tpu.cross_replica_sum(total_loss) \
                     / FLAGS.num_hosts / FLAGS.num_core_per_host
      metric_loss = tf.tile(tf.reshape(total_loss, [1, 1]), [batch_size, 1])
      eval_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=(metric_fn, [metric_loss]))

      eval_spec.cache = new_mems

      return eval_spec

    # Configuring the optimization step.
    global_step = tf.train.get_global_step()

    # increase the learning rate linearly
    if FLAGS.warmup_steps > 0:
      warmup_lr = tf.to_float(global_step) / tf.to_float(FLAGS.warmup_steps) \
                  * FLAGS.learning_rate
    else:
      warmup_lr = 0.0

    # number of parameters
    num_params = np.sum([np.prod(v.shape) for v in tf.trainable_variables()])
    tf.logging.info("#params: {}".format(num_params))

    # format_str = '{{:<{0}s}}\t{{}}'.format(
    #     max([len(v.name) for v in tf.trainable_variables()]))
    # for v in tf.trainable_variables():
    #   tf.logging.info(format_str.format(v.name, v.get_shape()))


    # decay the learning rate using the cosine schedule
    decay_lr = tf.train.cosine_decay(
        FLAGS.learning_rate,
        global_step=global_step-FLAGS.warmup_steps,
        decay_steps=FLAGS.train_steps-FLAGS.warmup_steps,
        alpha=FLAGS.min_lr_ratio)

    learning_rate = tf.where(global_step < FLAGS.warmup_steps,
                             warmup_lr, decay_lr)

    if FLAGS.use_tpu:
      optimizer = tf.compat.v1.tpu.CrossShardOptimizer(
          tf.train.AdamOptimizer(learning_rate=learning_rate))
      #GradientDescentOptimizer
    else:
      optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

    grads_and_vars = optimizer.compute_gradients(total_loss)
    gradients, variables = zip(*grads_and_vars)
    clipped, _ = tf.clip_by_global_norm(gradients, FLAGS.clip)
    train_op = optimizer.apply_gradients(
        zip(clipped, variables), global_step=tf.train.get_global_step())

    # Constucting TPUEstimatorSpec with cache.
    train_spec = tf.compat.v1.tpu.TPUEstimatorSpec(
        mode=mode, loss=total_loss, train_op=train_op)

    if FLAGS.mem_len < FLAGS.tgt_len:
      new_mems = [new_mems[: FLAGS.mem_len] for mem_t in new_mems]
    train_spec.cache = new_mems

    return train_spec

  return model_fn


def main(unused_argv):
  tf.logging.set_verbosity(tf.logging.INFO)

  # Get corpus info
  corpus_info = data_utils.get_corpus_info(FLAGS.corpus_info_path)
  n_token = corpus_info["vocab_size"]
  cutoffs = corpus_info["cutoffs"][1:-1]

  train_input_fn, train_record_info = data_utils.get_input_fn(
        record_info_dir=FLAGS.record_info_dir,
        split="train",
        per_host_bsz=FLAGS.train_batch_size // FLAGS.num_hosts,
        tgt_len=FLAGS.tgt_len,
        num_core_per_host=FLAGS.num_core_per_host,
        num_hosts=FLAGS.num_hosts,
        use_tpu=FLAGS.use_tpu)
  train_bin_sizes = train_record_info["bin_sizes"]
  num_train_batch = train_record_info["num_batch"]

  eval_input_fn, eval_record_info = data_utils.get_input_fn(
        record_info_dir=FLAGS.record_info_dir,
        split="valid",
        per_host_bsz=FLAGS.eval_batch_size // FLAGS.num_hosts,
        tgt_len=FLAGS.tgt_len,
        num_core_per_host=FLAGS.num_core_per_host,
        num_hosts=FLAGS.num_hosts,
        use_tpu=FLAGS.use_tpu)
  eval_bin_sizes = eval_record_info["bin_sizes"]
  num_eval_batch = eval_record_info["num_batch"]


  model_fn = get_model_fn(n_token, cutoffs, train_bin_sizes, eval_bin_sizes)

  config = tf.compat.v1.estimator.tpu.RunConfig(
      model_dir=FLAGS.model_dir,
      session_config=tf.ConfigProto(
          allow_soft_placement=True, log_device_placement=True),
      save_checkpoints_secs=None,
      save_checkpoints_steps=None
  )

  estimator = tf.compat.v1.estimator.tpu.TPUEstimator(
      use_tpu=False, #If False training will fall on CPU or GPU, depending on what is available 
      model_fn=model_fn,
      config=config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size)

  train_spec = tf.estimator.TrainSpec(input_fn=train_input_fn, max_steps=FLAGS.train_steps)
  eval_spec = tf.estimator.EvalSpec(input_fn=eval_input_fn)

  if not (mltunerUtil.is_chief() or mltunerUtil.is_ps()):
    time.sleep(1)
    if not tf.io.gfile.exists(model_dir):
      logging.debug("wait for chief init")
      time.sleep(1)
  tf.estimator.train_and_evaluate(estimator,train_spec,eval_spec)


if __name__ == "__main__":
  tf.app.run()