import json
from flask import request, abort
from flask.ext import restful
from flask.ext.restful import reqparse
from flask_rest_service import app, api, mongo
from bson.objectid import ObjectId
import os
import tensorflow as tf
import numpy as np
from nlp import NLP
import cnn
import util
import RAKE

def predict(x, config, raw_text=True):
    """ Build evaluation graph and run. """
    if raw_text:
        vocab = util.VocabLoader(config['data_dir'])
        x = vocab.text2id(x)
        class_names = vocab.class_names

        x_input = np.array([x])
    else:
        x_input = x

    with tf.Graph().as_default():
        with tf.variable_scope('cnn'):
            m = cnn.Model(config, is_train=False)
        saver = tf.train.Saver(tf.global_variables())

        with tf.Session() as sess:
            train_dir = config['train_dir']
            ckpt = tf.train.get_checkpoint_state(train_dir)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)
            else:
                raise IOError("Loading checkpoint file failed!")

            scores = sess.run(m.scores, feed_dict={m.inputs:x_input})

    if raw_text:
        scores = [float(str(i)) for i in scores[0]]
        y_pred = class_names[int(np.argmax(scores))]
        scores = dict(zip(class_names, scores))
    else:
        y_pred = np.argmax(scores, axis=1)

    ret = dict()
    ret['prediction'] = y_pred
    ret['scores'] = scores
    return ret

class Detect(restful.Resource):
    def get(self):
        text = request.args.get('text')
        text = u"".join(text)
        config = util.load_from_dump(os.path.join(FLAGS.train_dir, 'flags.cPickle'))
        config['data_dir']  = os.path.join(this_dir, 'data', 'ted500')
        config['train_dir'] = os.path.join(this_dir, 'model', 'ted500')

        # predict
        result = predict(text, config, raw_text=True)
        language_codes = util.load_language_codes()

        return {
            'status':   'OK',
            'langauge': language_codes[result['prediction']]
        }

class Expand(restful.Resource):
    def get(self):
        expansions = nlp.expand(request.args.get('word'))

        return {
            'status': 'OK',
            'expansions': expansions
        }

class Rake(restful.Resource):
    def get(self):
        Rake = RAKE.Rake(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'FoxStoplist.txt'))

        return {
            'status':  'OK',
            'keywords': Rake.run(request.args.get('text'))
        }

class Linguine(restful.Resource):
    def get(self):
        Rake = RAKE.Rake(os.path.join(os.path.abspath(os.path.dirname(__file__)), 'FoxStoplist.txt'))

        text = request.args.get('text')
        text = u"".join(text)
        # config = util.load_from_dump(os.path.join(FLAGS.train_dir, 'flags.cPickle'))
        # config['data_dir']  = os.path.join(this_dir, 'data', 'ted500')
        # config['train_dir'] = os.path.join(this_dir, 'model', 'ted500')
        #
        # # predict
        # result = predict(text, config, raw_text=True)
        # language_codes = util.load_language_codes()

        keywords   = Rake.run(text)
        expansions = {}

        for word in keywords:
            expansions[word[0]] = nlp.expand(word[0])

        return {
            'status':   'OK',
            'linguine': {
                'original':   text,
                'language':   'English',
                'keywords':   keywords,
                'expansions': expansions
            }
        }

class Root(restful.Resource):
    def get(self):
        return {
            'status': 'OK',
            'mongo': str(mongo.db),
        }

# Load the NLP class.
nlp  = NLP()

FLAGS = tf.app.flags.FLAGS
this_dir = os.path.abspath(os.path.dirname(__file__))
tf.app.flags.DEFINE_string('data_dir', os.path.join(this_dir, 'data', 'ted500'), 'Directory of the data')
tf.app.flags.DEFINE_string('train_dir', os.path.join(this_dir, 'model', 'ted500'), 'Where to read model')
FLAGS._parse_flags()


api.add_resource(Root, '/')
api.add_resource(Expand, '/expand')
api.add_resource(Detect, '/detect')
api.add_resource(Rake, '/rake')
api.add_resource(Linguine, '/linguine')
