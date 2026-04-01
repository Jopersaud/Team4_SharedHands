import sys
import tf_keras
sys.modules['keras'] = tf_keras

exec(open('train_the_model_asl.py').read())
