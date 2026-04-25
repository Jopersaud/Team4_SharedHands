(sharedhands-env) Joshuas-MacBook-Pro:artifacts josh$ pip install opencv-python mediapipe numpy keras scikit-learn pandas tensorflow-macos tensorflow-metal
Requirement already satisfied: opencv-python in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (4.13.0.92)
Requirement already satisfied: mediapipe in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (0.10.33)
Requirement already satisfied: numpy in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (2.4.3)
Requirement already satisfied: keras in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (3.13.2)
Collecting scikit-learn
  Downloading scikit_learn-1.8.0-cp314-cp314-macosx_12_0_arm64.whl.metadata (11 kB)
Collecting pandas
  Downloading pandas-3.0.1-cp314-cp314-macosx_11_0_arm64.whl.metadata (79 kB)
ERROR: Could not find a version that satisfies the requirement tensorflow-macos (from versions: none)
ERROR: No matching distribution found for tensorflow-macos
(sharedhands-env) Joshuas-MacBook-Pro:artifacts josh$ python -m pip install opencv-python mediapipe numpy keras scikit-learn pandas tensorflow-macos tensorflow-metal
Collecting opencv-python
  Using cached opencv_python-4.13.0.92-cp37-abi3-macosx_13_0_arm64.whl.metadata (19 kB)
Collecting mediapipe
  Using cached mediapipe-0.10.33-py3-none-macosx_11_0_arm64.whl.metadata (9.8 kB)
Collecting numpy
  Downloading numpy-2.4.3-cp311-cp311-macosx_14_0_arm64.whl.metadata (6.6 kB)
Collecting keras
  Using cached keras-3.13.2-py3-none-any.whl.metadata (6.3 kB)
Collecting scikit-learn
  Downloading scikit_learn-1.8.0-cp311-cp311-macosx_12_0_arm64.whl.metadata (11 kB)
Collecting pandas
  Downloading pandas-3.0.1-cp311-cp311-macosx_11_0_arm64.whl.metadata (79 kB)
Collecting tensorflow-macos
  Downloading tensorflow_macos-2.16.2-cp311-cp311-macosx_12_0_arm64.whl.metadata (3.3 kB)
Collecting tensorflow-metal
  Downloading tensorflow_metal-1.2.0-cp311-cp311-macosx_12_0_arm64.whl.metadata (1.3 kB)
Collecting absl-py~=2.3 (from mediapipe)
  Using cached absl_py-2.4.0-py3-none-any.whl.metadata (3.3 kB)
Collecting sounddevice~=0.5 (from mediapipe)
  Using cached sounddevice-0.5.5-py3-none-macosx_10_6_x86_64.macosx_10_6_universal2.whl.metadata (1.4 kB)
Collecting flatbuffers~=25.9 (from mediapipe)
  Using cached flatbuffers-25.12.19-py2.py3-none-any.whl.metadata (1.0 kB)
Collecting opencv-contrib-python (from mediapipe)
  Using cached opencv_contrib_python-4.13.0.92-cp37-abi3-macosx_13_0_arm64.whl.metadata (19 kB)
Collecting matplotlib (from mediapipe)
  Downloading matplotlib-3.10.8-cp311-cp311-macosx_11_0_arm64.whl.metadata (52 kB)
Collecting cffi (from sounddevice~=0.5->mediapipe)
  Downloading cffi-2.0.0-cp311-cp311-macosx_11_0_arm64.whl.metadata (2.6 kB)
Collecting rich (from keras)
  Using cached rich-14.3.3-py3-none-any.whl.metadata (18 kB)
Collecting namex (from keras)
  Using cached namex-0.1.0-py3-none-any.whl.metadata (322 bytes)
Collecting h5py (from keras)
  Downloading h5py-3.16.0-cp311-cp311-macosx_11_0_arm64.whl.metadata (3.0 kB)
Collecting optree (from keras)
  Downloading optree-0.19.0-cp311-cp311-macosx_11_0_arm64.whl.metadata (34 kB)
Collecting ml-dtypes (from keras)
  Downloading ml_dtypes-0.5.4-cp311-cp311-macosx_10_9_universal2.whl.metadata (8.9 kB)
Requirement already satisfied: packaging in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from keras) (25.0)
Collecting scipy>=1.10.0 (from scikit-learn)
  Downloading scipy-1.17.1-cp311-cp311-macosx_14_0_arm64.whl.metadata (62 kB)
Collecting joblib>=1.3.0 (from scikit-learn)
  Using cached joblib-1.5.3-py3-none-any.whl.metadata (5.5 kB)
Collecting threadpoolctl>=3.2.0 (from scikit-learn)
  Using cached threadpoolctl-3.6.0-py3-none-any.whl.metadata (13 kB)
Collecting python-dateutil>=2.8.2 (from pandas)
  Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
Collecting tensorflow==2.16.2 (from tensorflow-macos)
  Downloading tensorflow-2.16.2-cp311-cp311-macosx_12_0_arm64.whl.metadata (4.1 kB)
Collecting astunparse>=1.6.0 (from tensorflow==2.16.2->tensorflow-macos)
  Using cached astunparse-1.6.3-py2.py3-none-any.whl.metadata (4.4 kB)
Collecting gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 (from tensorflow==2.16.2->tensorflow-macos)
  Using cached gast-0.7.0-py3-none-any.whl.metadata (1.5 kB)
Collecting google-pasta>=0.1.1 (from tensorflow==2.16.2->tensorflow-macos)
  Using cached google_pasta-0.2.0-py3-none-any.whl.metadata (814 bytes)
Collecting libclang>=13.0.0 (from tensorflow==2.16.2->tensorflow-macos)
  Using cached libclang-18.1.1-1-py2.py3-none-macosx_11_0_arm64.whl.metadata (5.2 kB)
Collecting ml-dtypes (from keras)
  Downloading ml_dtypes-0.3.2-cp311-cp311-macosx_10_9_universal2.whl.metadata (20 kB)
Collecting opt-einsum>=2.3.2 (from tensorflow==2.16.2->tensorflow-macos)
  Using cached opt_einsum-3.4.0-py3-none-any.whl.metadata (6.3 kB)
Collecting protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 (from tensorflow==2.16.2->tensorflow-macos)
  Downloading protobuf-4.25.8-cp37-abi3-macosx_10_9_universal2.whl.metadata (541 bytes)
Collecting requests<3,>=2.21.0 (from tensorflow==2.16.2->tensorflow-macos)
  Using cached requests-2.32.5-py3-none-any.whl.metadata (4.9 kB)
Requirement already satisfied: setuptools in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from tensorflow==2.16.2->tensorflow-macos) (80.10.2)
Collecting six>=1.12.0 (from tensorflow==2.16.2->tensorflow-macos)
  Using cached six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
Collecting termcolor>=1.1.0 (from tensorflow==2.16.2->tensorflow-macos)
  Using cached termcolor-3.3.0-py3-none-any.whl.metadata (6.5 kB)
Collecting typing-extensions>=3.6.6 (from tensorflow==2.16.2->tensorflow-macos)
  Using cached typing_extensions-4.15.0-py3-none-any.whl.metadata (3.3 kB)
Collecting wrapt>=1.11.0 (from tensorflow==2.16.2->tensorflow-macos)
  Downloading wrapt-2.1.2-cp311-cp311-macosx_11_0_arm64.whl.metadata (7.4 kB)
Collecting grpcio<2.0,>=1.24.3 (from tensorflow==2.16.2->tensorflow-macos)
  Downloading grpcio-1.78.0-cp311-cp311-macosx_11_0_universal2.whl.metadata (3.8 kB)
Collecting tensorboard<2.17,>=2.16 (from tensorflow==2.16.2->tensorflow-macos)
  Downloading tensorboard-2.16.2-py3-none-any.whl.metadata (1.6 kB)
Collecting tensorflow-io-gcs-filesystem>=0.23.1 (from tensorflow==2.16.2->tensorflow-macos)
  Downloading tensorflow_io_gcs_filesystem-0.37.1-cp311-cp311-macosx_12_0_arm64.whl.metadata (14 kB)
INFO: pip is looking at multiple versions of tensorflow to determine which version is compatible with other requirements. This could take a while.
Collecting tensorflow-macos
  Downloading tensorflow_macos-2.16.1-cp311-cp311-macosx_12_0_arm64.whl.metadata (3.5 kB)
Collecting tensorflow==2.16.1 (from tensorflow-macos)
  Downloading tensorflow-2.16.1-cp311-cp311-macosx_12_0_arm64.whl.metadata (4.1 kB)
Collecting tensorflow-macos
  Downloading tensorflow_macos-2.15.1-cp311-cp311-macosx_12_0_arm64.whl.metadata (3.4 kB)
Requirement already satisfied: wheel~=0.35 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from tensorflow-metal) (0.46.3)
Collecting pycparser (from cffi->sounddevice~=0.5->mediapipe)
  Using cached pycparser-3.0-py3-none-any.whl.metadata (8.2 kB)
Collecting contourpy>=1.0.1 (from matplotlib->mediapipe)
  Downloading contourpy-1.3.3-cp311-cp311-macosx_11_0_arm64.whl.metadata (5.5 kB)
Collecting cycler>=0.10 (from matplotlib->mediapipe)
  Using cached cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)
Collecting fonttools>=4.22.0 (from matplotlib->mediapipe)
  Downloading fonttools-4.62.1-cp311-cp311-macosx_10_9_universal2.whl.metadata (117 kB)
Collecting kiwisolver>=1.3.1 (from matplotlib->mediapipe)
  Downloading kiwisolver-1.5.0-cp311-cp311-macosx_11_0_arm64.whl.metadata (5.1 kB)
Collecting pillow>=8 (from matplotlib->mediapipe)
  Downloading pillow-12.1.1-cp311-cp311-macosx_11_0_arm64.whl.metadata (8.8 kB)
Collecting pyparsing>=3 (from matplotlib->mediapipe)
  Using cached pyparsing-3.3.2-py3-none-any.whl.metadata (5.8 kB)
Collecting markdown-it-py>=2.2.0 (from rich->keras)
  Using cached markdown_it_py-4.0.0-py3-none-any.whl.metadata (7.3 kB)
Collecting pygments<3.0.0,>=2.13.0 (from rich->keras)
  Using cached pygments-2.19.2-py3-none-any.whl.metadata (2.5 kB)
Collecting mdurl~=0.1 (from markdown-it-py>=2.2.0->rich->keras)
  Using cached mdurl-0.1.2-py3-none-any.whl.metadata (1.6 kB)
Using cached opencv_python-4.13.0.92-cp37-abi3-macosx_13_0_arm64.whl (46.2 MB)
Using cached mediapipe-0.10.33-py3-none-macosx_11_0_arm64.whl (29.4 MB)
Using cached absl_py-2.4.0-py3-none-any.whl (135 kB)
Using cached flatbuffers-25.12.19-py2.py3-none-any.whl (26 kB)
Using cached sounddevice-0.5.5-py3-none-macosx_10_6_x86_64.macosx_10_6_universal2.whl (108 kB)
Downloading numpy-2.4.3-cp311-cp311-macosx_14_0_arm64.whl (5.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 5.5/5.5 MB 21.5 MB/s  0:00:00
Using cached keras-3.13.2-py3-none-any.whl (1.5 MB)
Downloading scikit_learn-1.8.0-cp311-cp311-macosx_12_0_arm64.whl (8.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.1/8.1 MB 22.4 MB/s  0:00:00
Downloading pandas-3.0.1-cp311-cp311-macosx_11_0_arm64.whl (9.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 9.9/9.9 MB 21.2 MB/s  0:00:00
Downloading tensorflow_macos-2.15.1-cp311-cp311-macosx_12_0_arm64.whl (2.2 kB)
Downloading tensorflow_metal-1.2.0-cp311-cp311-macosx_12_0_arm64.whl (1.4 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.4/1.4 MB 15.7 MB/s  0:00:00
Using cached joblib-1.5.3-py3-none-any.whl (309 kB)
Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
Downloading scipy-1.17.1-cp311-cp311-macosx_14_0_arm64.whl (20.3 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 20.3/20.3 MB 26.4 MB/s  0:00:00
Using cached six-1.17.0-py2.py3-none-any.whl (11 kB)
Using cached threadpoolctl-3.6.0-py3-none-any.whl (18 kB)
Downloading cffi-2.0.0-cp311-cp311-macosx_11_0_arm64.whl (180 kB)
Downloading h5py-3.16.0-cp311-cp311-macosx_11_0_arm64.whl (3.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 3.1/3.1 MB 23.4 MB/s  0:00:00
Downloading matplotlib-3.10.8-cp311-cp311-macosx_11_0_arm64.whl (8.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.1/8.1 MB 20.3 MB/s  0:00:00
Downloading contourpy-1.3.3-cp311-cp311-macosx_11_0_arm64.whl (270 kB)
Using cached cycler-0.12.1-py3-none-any.whl (8.3 kB)
Downloading fonttools-4.62.1-cp311-cp311-macosx_10_9_universal2.whl (2.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.9/2.9 MB 23.4 MB/s  0:00:00
Downloading kiwisolver-1.5.0-cp311-cp311-macosx_11_0_arm64.whl (63 kB)
Downloading pillow-12.1.1-cp311-cp311-macosx_11_0_arm64.whl (4.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 4.7/4.7 MB 23.6 MB/s  0:00:00
Using cached pyparsing-3.3.2-py3-none-any.whl (122 kB)
Downloading ml_dtypes-0.5.4-cp311-cp311-macosx_10_9_universal2.whl (679 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 679.7/679.7 kB 25.1 MB/s  0:00:00
Using cached namex-0.1.0-py3-none-any.whl (5.9 kB)
Using cached opencv_contrib_python-4.13.0.92-cp37-abi3-macosx_13_0_arm64.whl (52.0 MB)
Downloading optree-0.19.0-cp311-cp311-macosx_11_0_arm64.whl (378 kB)
Using cached typing_extensions-4.15.0-py3-none-any.whl (44 kB)
Using cached pycparser-3.0-py3-none-any.whl (48 kB)
Using cached rich-14.3.3-py3-none-any.whl (310 kB)
Using cached pygments-2.19.2-py3-none-any.whl (1.2 MB)
Using cached markdown_it_py-4.0.0-py3-none-any.whl (87 kB)
Using cached mdurl-0.1.2-py3-none-any.whl (10.0 kB)
Installing collected packages: namex, flatbuffers, typing-extensions, threadpoolctl, six, pyparsing, pygments, pycparser, pillow, numpy, mdurl, kiwisolver, joblib, fonttools, cycler, absl-py, tensorflow-metal, scipy, python-dateutil, optree, opencv-python, opencv-contrib-python, ml-dtypes, markdown-it-py, h5py, contourpy, cffi, sounddevice, scikit-learn, rich, pandas, matplotlib, mediapipe, keras, tensorflow-macos
Successfully installed absl-py-2.4.0 cffi-2.0.0 contourpy-1.3.3 cycler-0.12.1 flatbuffers-25.12.19 fonttools-4.62.1 h5py-3.16.0 joblib-1.5.3 keras-3.13.2 kiwisolver-1.5.0 markdown-it-py-4.0.0 matplotlib-3.10.8 mdurl-0.1.2 mediapipe-0.10.33 ml-dtypes-0.5.4 namex-0.1.0 numpy-2.4.3 opencv-contrib-python-4.13.0.92 opencv-python-4.13.0.92 optree-0.19.0 pandas-3.0.1 pillow-12.1.1 pycparser-3.0 pygments-2.19.2 pyparsing-3.3.2 python-dateutil-2.9.0.post0 rich-14.3.3 scikit-learn-1.8.0 scipy-1.17.1 six-1.17.0 sounddevice-0.5.5 tensorflow-macos-2.15.1 tensorflow-metal-1.2.0 threadpoolctl-3.6.0 typing-extensions-4.15.0
(sharedhands-env) Joshuas-MacBook-Pro:artifacts josh$ python ASL_Version\ 1.0.py 
Traceback (most recent call last):
  File "/Users/josh/Team4_SharedHands/Team4_SharedHands/artifacts/ASL_Version 1.0.py", line 5, in <module>
    import keras
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/keras/__init__.py", line 7, in <module>
    from keras import _tf_keras as _tf_keras
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/keras/_tf_keras/__init__.py", line 1, in <module>
    from keras._tf_keras import keras
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/keras/_tf_keras/keras/__init__.py", line 7, in <module>
    from keras import activations as activations
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/keras/activations/__init__.py", line 7, in <module>
    from keras.src.activations import deserialize as deserialize
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/keras/src/__init__.py", line 1, in <module>
    from keras.src import activations
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/keras/src/activations/__init__.py", line 3, in <module>
    from keras.src.activations.activations import celu
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/keras/src/activations/activations.py", line 1, in <module>
    from keras.src import backend
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/keras/src/backend/__init__.py", line 1, in <module>
    from keras.src.backend.config import backend
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/keras/src/backend/config.py", line 448, in <module>
    set_nnx_enabled(_NNX_ENABLED)
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/keras/src/backend/config.py", line 249, in set_nnx_enabled
    from keras.src.backend.common import global_state
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/keras/src/backend/common/__init__.py", line 2, in <module>
    from keras.src.backend.common.dtypes import result_type
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/keras/src/backend/common/dtypes.py", line 5, in <module>
    from keras.src.backend.common.variables import standardize_dtype
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/keras/src/backend/common/variables.py", line 11, in <module>
    from keras.src.utils.module_utils import tensorflow as tf
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/keras/src/utils/__init__.py", line 1, in <module>
    from keras.src.utils.audio_dataset_utils import audio_dataset_from_directory
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/keras/src/utils/audio_dataset_utils.py", line 4, in <module>
    from keras.src.utils import dataset_utils
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/keras/src/utils/dataset_utils.py", line 10, in <module>
    from keras.src import tree
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/keras/src/tree/__init__.py", line 1, in <module>
    from keras.src.tree.tree_api import assert_same_paths
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/keras/src/tree/tree_api.py", line 13, in <module>
    from keras.src.tree import optree_impl as tree_impl
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/keras/src/tree/optree_impl.py", line 13, in <module>
    from tensorflow.python.trackable.data_structures import ListWrapper
ModuleNotFoundError: No module named 'tensorflow'
(sharedhands-env) Joshuas-MacBook-Pro:artifacts josh$  python -c "import tensorflow as tf; print(tf.__version__)" 
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'tensorflow'
(sharedhands-env) Joshuas-MacBook-Pro:artifacts josh$ python -c "import tensorflow as tf; print(tf.__version__)" 
Traceback (most recent call last):
  File "<string>", line 1, in <module>
ModuleNotFoundError: No module named 'tensorflow'
(sharedhands-env) Joshuas-MacBook-Pro:artifacts josh$ python -m pip install tensorflow 
Collecting tensorflow
  Downloading tensorflow-2.21.0-cp311-cp311-macosx_12_0_arm64.whl.metadata (4.4 kB)
Requirement already satisfied: absl-py>=1.0.0 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from tensorflow) (2.4.0)
Collecting astunparse>=1.6.0 (from tensorflow)
  Using cached astunparse-1.6.3-py2.py3-none-any.whl.metadata (4.4 kB)
Requirement already satisfied: flatbuffers>=25.9.23 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from tensorflow) (25.12.19)
Collecting gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 (from tensorflow)
  Using cached gast-0.7.0-py3-none-any.whl.metadata (1.5 kB)
Collecting google_pasta>=0.1.1 (from tensorflow)
  Using cached google_pasta-0.2.0-py3-none-any.whl.metadata (814 bytes)
Collecting libclang>=13.0.0 (from tensorflow)
  Using cached libclang-18.1.1-1-py2.py3-none-macosx_11_0_arm64.whl.metadata (5.2 kB)
Collecting opt_einsum>=2.3.2 (from tensorflow)
  Using cached opt_einsum-3.4.0-py3-none-any.whl.metadata (6.3 kB)
Requirement already satisfied: packaging in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from tensorflow) (25.0)
Collecting protobuf<8.0.0,>=6.31.1 (from tensorflow)
  Downloading protobuf-7.34.1-cp310-abi3-macosx_10_9_universal2.whl.metadata (595 bytes)
Collecting requests<3,>=2.21.0 (from tensorflow)
  Using cached requests-2.32.5-py3-none-any.whl.metadata (4.9 kB)
Requirement already satisfied: setuptools in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from tensorflow) (80.10.2)
Requirement already satisfied: six>=1.12.0 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from tensorflow) (1.17.0)
Collecting termcolor>=1.1.0 (from tensorflow)
  Using cached termcolor-3.3.0-py3-none-any.whl.metadata (6.5 kB)
Requirement already satisfied: typing_extensions>=3.6.6 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from tensorflow) (4.15.0)
Collecting wrapt>=1.11.0 (from tensorflow)
  Using cached wrapt-2.1.2-cp311-cp311-macosx_11_0_arm64.whl.metadata (7.4 kB)
Collecting grpcio<2.0,>=1.24.3 (from tensorflow)
  Using cached grpcio-1.78.0-cp311-cp311-macosx_11_0_universal2.whl.metadata (3.8 kB)
Requirement already satisfied: keras>=3.12.0 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from tensorflow) (3.13.2)
Requirement already satisfied: numpy>=1.26.0 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from tensorflow) (2.4.3)
Collecting h5py<3.15.0,>=3.11.0 (from tensorflow)
  Downloading h5py-3.14.0-cp311-cp311-macosx_11_0_arm64.whl.metadata (2.7 kB)
Requirement already satisfied: ml_dtypes<1.0.0,>=0.5.1 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from tensorflow) (0.5.4)
Collecting charset_normalizer<4,>=2 (from requests<3,>=2.21.0->tensorflow)
  Downloading charset_normalizer-3.4.6-cp311-cp311-macosx_10_9_universal2.whl.metadata (40 kB)
Collecting idna<4,>=2.5 (from requests<3,>=2.21.0->tensorflow)
  Using cached idna-3.11-py3-none-any.whl.metadata (8.4 kB)
Collecting urllib3<3,>=1.21.1 (from requests<3,>=2.21.0->tensorflow)
  Using cached urllib3-2.6.3-py3-none-any.whl.metadata (6.9 kB)
Collecting certifi>=2017.4.17 (from requests<3,>=2.21.0->tensorflow)
  Using cached certifi-2026.2.25-py3-none-any.whl.metadata (2.5 kB)
Requirement already satisfied: wheel<1.0,>=0.23.0 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from astunparse>=1.6.0->tensorflow) (0.46.3)
Requirement already satisfied: rich in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from keras>=3.12.0->tensorflow) (14.3.3)
Requirement already satisfied: namex in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from keras>=3.12.0->tensorflow) (0.1.0)
Requirement already satisfied: optree in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from keras>=3.12.0->tensorflow) (0.19.0)
Requirement already satisfied: markdown-it-py>=2.2.0 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from rich->keras>=3.12.0->tensorflow) (4.0.0)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from rich->keras>=3.12.0->tensorflow) (2.19.2)
Requirement already satisfied: mdurl~=0.1 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from markdown-it-py>=2.2.0->rich->keras>=3.12.0->tensorflow) (0.1.2)
Downloading tensorflow-2.21.0-cp311-cp311-macosx_12_0_arm64.whl (223.4 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 223.4/223.4 MB 24.1 MB/s  0:00:09
Downloading grpcio-1.78.0-cp311-cp311-macosx_11_0_universal2.whl (11.8 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 11.8/11.8 MB 24.5 MB/s  0:00:00
Downloading h5py-3.14.0-cp311-cp311-macosx_11_0_arm64.whl (2.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.9/2.9 MB 19.9 MB/s  0:00:00
Downloading protobuf-7.34.1-cp310-abi3-macosx_10_9_universal2.whl (429 kB)
Using cached requests-2.32.5-py3-none-any.whl (64 kB)
Downloading charset_normalizer-3.4.6-cp311-cp311-macosx_10_9_universal2.whl (293 kB)
Using cached idna-3.11-py3-none-any.whl (71 kB)
Using cached urllib3-2.6.3-py3-none-any.whl (131 kB)
Using cached astunparse-1.6.3-py2.py3-none-any.whl (12 kB)
Using cached certifi-2026.2.25-py3-none-any.whl (153 kB)
Using cached gast-0.7.0-py3-none-any.whl (22 kB)
Using cached google_pasta-0.2.0-py3-none-any.whl (57 kB)
Using cached libclang-18.1.1-1-py2.py3-none-macosx_11_0_arm64.whl (25.8 MB)
Using cached opt_einsum-3.4.0-py3-none-any.whl (71 kB)
Using cached termcolor-3.3.0-py3-none-any.whl (7.7 kB)
Downloading wrapt-2.1.2-cp311-cp311-macosx_11_0_arm64.whl (61 kB)
Installing collected packages: libclang, wrapt, urllib3, termcolor, protobuf, opt_einsum, idna, h5py, grpcio, google_pasta, gast, charset_normalizer, certifi, requests, astunparse, tensorflow
  Attempting uninstall: h5py
    Found existing installation: h5py 3.16.0
    Uninstalling h5py-3.16.0:
      Successfully uninstalled h5py-3.16.0
Successfully installed astunparse-1.6.3 certifi-2026.2.25 charset_normalizer-3.4.6 gast-0.7.0 google_pasta-0.2.0 grpcio-1.78.0 h5py-3.14.0 idna-3.11 libclang-18.1.1 opt_einsum-3.4.0 protobuf-7.34.1 requests-2.32.5 tensorflow-2.21.0 termcolor-3.3.0 urllib3-2.6.3 wrapt-2.1.2
(sharedhands-env) Joshuas-MacBook-Pro:artifacts josh$ python ASL_Version\ 1.0.py 
Traceback (most recent call last):
  File "/Users/josh/Team4_SharedHands/Team4_SharedHands/artifacts/ASL_Version 1.0.py", line 2, in <module>
    import mediapipe as mp
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/mediapipe/__init__.py", line 15, in <module>
    import mediapipe.tasks.python as tasks
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/mediapipe/tasks/python/__init__.py", line 17, in <module>
    from . import audio
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/mediapipe/tasks/python/audio/__init__.py", line 18, in <module>
    import mediapipe.tasks.python.audio.audio_classifier
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/mediapipe/tasks/python/audio/audio_classifier.py", line 21, in <module>
    from mediapipe.tasks.python.audio.core import audio_task_running_mode
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/mediapipe/tasks/python/audio/core/audio_task_running_mode.py", line 19, in <module>
    from mediapipe.tasks.python.core.optional_dependencies import doc_controls
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/mediapipe/tasks/python/core/optional_dependencies.py", line 20, in <module>
    from tensorflow.tools.docs import doc_controls
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/tensorflow/__init__.py", line 439, in <module>
    _ll.load_library(_plugin_dir)
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/tensorflow/python/framework/load_library.py", line 151, in load_library
    py_tf.TF_LoadLibrary(lib)
tensorflow.python.framework.errors_impl.NotFoundError: dlopen(/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/tensorflow-plugins/libmetal_plugin.dylib, 0x0006): Library not loaded: @rpath/_pywrap_tensorflow_internal.so
  Referenced from: <8B62586B-B082-3113-93AB-FD766A9960AE> /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/tensorflow-plugins/libmetal_plugin.dylib
  Reason: tried: '/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/tensorflow-plugins/../_solib_darwin_arm64/_U@local_Uconfig_Utf_S_S_C_Upywrap_Utensorflow_Uinternal___Uexternal_Slocal_Uconfig_Utf/_pywrap_tensorflow_internal.so' (no such file), '/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/tensorflow-plugins/../_solib_darwin_arm64/_U@local_Uconfig_Utf_S_S_C_Upywrap_Utensorflow_Uinternal___Uexternal_Slocal_Uconfig_Utf/_pywrap_tensorflow_internal.so' (no such file), '/Users/josh/miniconda3/envs/sharedhands-env/bin/../lib/_pywrap_tensorflow_internal.so' (no such file)
(sharedhands-env) Joshuas-MacBook-Pro:artifacts josh$ python -m pip uninstall tensorflow tensorflow-metal
Found existing installation: tensorflow 2.21.0
Uninstalling tensorflow-2.21.0:
  Would remove:
    /Users/josh/miniconda3/envs/sharedhands-env/bin/saved_model_cli
    /Users/josh/miniconda3/envs/sharedhands-env/bin/tf_upgrade_v2
    /Users/josh/miniconda3/envs/sharedhands-env/bin/tflite_convert
    /Users/josh/miniconda3/envs/sharedhands-env/bin/toco
    /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/tensorflow-2.21.0.dist-info/*
    /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/tensorflow/*
Proceed (Y/n)? y
  Successfully uninstalled tensorflow-2.21.0
Found existing installation: tensorflow-metal 1.2.0
Uninstalling tensorflow-metal-1.2.0:
  Would remove:
    /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/tensorflow-plugins/*
    /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/tensorflow_metal-1.2.0.dist-info/*
Proceed (Y/n)? y
  Successfully uninstalled tensorflow-metal-1.2.0
(sharedhands-env) Joshuas-MacBook-Pro:artifacts josh$ python -m pip install tensorflow tensorflow-metal
Collecting tensorflow
  Using cached tensorflow-2.21.0-cp311-cp311-macosx_12_0_arm64.whl.metadata (4.4 kB)
Collecting tensorflow-metal
  Using cached tensorflow_metal-1.2.0-cp311-cp311-macosx_12_0_arm64.whl.metadata (1.3 kB)
Requirement already satisfied: absl-py>=1.0.0 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from tensorflow) (2.4.0)
Requirement already satisfied: astunparse>=1.6.0 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from tensorflow) (1.6.3)
Requirement already satisfied: flatbuffers>=25.9.23 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from tensorflow) (25.12.19)
Requirement already satisfied: gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from tensorflow) (0.7.0)
Requirement already satisfied: google_pasta>=0.1.1 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from tensorflow) (0.2.0)
Requirement already satisfied: libclang>=13.0.0 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from tensorflow) (18.1.1)
Requirement already satisfied: opt_einsum>=2.3.2 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from tensorflow) (3.4.0)
Requirement already satisfied: packaging in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from tensorflow) (25.0)
Requirement already satisfied: protobuf<8.0.0,>=6.31.1 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from tensorflow) (7.34.1)
Requirement already satisfied: requests<3,>=2.21.0 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from tensorflow) (2.32.5)
Requirement already satisfied: setuptools in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from tensorflow) (80.10.2)
Requirement already satisfied: six>=1.12.0 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from tensorflow) (1.17.0)
Requirement already satisfied: termcolor>=1.1.0 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from tensorflow) (3.3.0)
Requirement already satisfied: typing_extensions>=3.6.6 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from tensorflow) (4.15.0)
Requirement already satisfied: wrapt>=1.11.0 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from tensorflow) (2.1.2)
Requirement already satisfied: grpcio<2.0,>=1.24.3 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from tensorflow) (1.78.0)
Requirement already satisfied: keras>=3.12.0 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from tensorflow) (3.13.2)
Requirement already satisfied: numpy>=1.26.0 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from tensorflow) (2.4.3)
Requirement already satisfied: h5py<3.15.0,>=3.11.0 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from tensorflow) (3.14.0)
Requirement already satisfied: ml_dtypes<1.0.0,>=0.5.1 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from tensorflow) (0.5.4)
Requirement already satisfied: charset_normalizer<4,>=2 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from requests<3,>=2.21.0->tensorflow) (3.4.6)
Requirement already satisfied: idna<4,>=2.5 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from requests<3,>=2.21.0->tensorflow) (3.11)
Requirement already satisfied: urllib3<3,>=1.21.1 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from requests<3,>=2.21.0->tensorflow) (2.6.3)
Requirement already satisfied: certifi>=2017.4.17 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from requests<3,>=2.21.0->tensorflow) (2026.2.25)
Requirement already satisfied: wheel~=0.35 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from tensorflow-metal) (0.46.3)
Requirement already satisfied: rich in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from keras>=3.12.0->tensorflow) (14.3.3)
Requirement already satisfied: namex in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from keras>=3.12.0->tensorflow) (0.1.0)
Requirement already satisfied: optree in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from keras>=3.12.0->tensorflow) (0.19.0)
Requirement already satisfied: markdown-it-py>=2.2.0 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from rich->keras>=3.12.0->tensorflow) (4.0.0)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from rich->keras>=3.12.0->tensorflow) (2.19.2)
Requirement already satisfied: mdurl~=0.1 in /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages (from markdown-it-py>=2.2.0->rich->keras>=3.12.0->tensorflow) (0.1.2)
Using cached tensorflow-2.21.0-cp311-cp311-macosx_12_0_arm64.whl (223.4 MB)
Using cached tensorflow_metal-1.2.0-cp311-cp311-macosx_12_0_arm64.whl (1.4 MB)
Installing collected packages: tensorflow-metal, tensorflow
Successfully installed tensorflow-2.21.0 tensorflow-metal-1.2.0
(sharedhands-env) Joshuas-MacBook-Pro:artifacts josh$ python ASL_Version\ 1.0.py 
Traceback (most recent call last):
  File "/Users/josh/Team4_SharedHands/Team4_SharedHands/artifacts/ASL_Version 1.0.py", line 2, in <module>
    import mediapipe as mp
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/mediapipe/__init__.py", line 15, in <module>
    import mediapipe.tasks.python as tasks
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/mediapipe/tasks/python/__init__.py", line 17, in <module>
    from . import audio
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/mediapipe/tasks/python/audio/__init__.py", line 18, in <module>
    import mediapipe.tasks.python.audio.audio_classifier
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/mediapipe/tasks/python/audio/audio_classifier.py", line 21, in <module>
    from mediapipe.tasks.python.audio.core import audio_task_running_mode
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/mediapipe/tasks/python/audio/core/audio_task_running_mode.py", line 19, in <module>
    from mediapipe.tasks.python.core.optional_dependencies import doc_controls
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/mediapipe/tasks/python/core/optional_dependencies.py", line 20, in <module>
    from tensorflow.tools.docs import doc_controls
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/tensorflow/__init__.py", line 439, in <module>
    _ll.load_library(_plugin_dir)
  File "/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/tensorflow/python/framework/load_library.py", line 151, in load_library
    py_tf.TF_LoadLibrary(lib)
tensorflow.python.framework.errors_impl.NotFoundError: dlopen(/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/tensorflow-plugins/libmetal_plugin.dylib, 0x0006): Library not loaded: @rpath/_pywrap_tensorflow_internal.so
  Referenced from: <8B62586B-B082-3113-93AB-FD766A9960AE> /Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/tensorflow-plugins/libmetal_plugin.dylib
  Reason: tried: '/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/tensorflow-plugins/../_solib_darwin_arm64/_U@local_Uconfig_Utf_S_S_C_Upywrap_Utensorflow_Uinternal___Uexternal_Slocal_Uconfig_Utf/_pywrap_tensorflow_internal.so' (no such file), '/Users/josh/miniconda3/envs/sharedhands-env/lib/python3.11/site-packages/tensorflow-plugins/../_solib_darwin_arm64/_U@local_Uconfig_Utf_S_S_C_Upywrap_Utensorflow_Uinternal___Uexternal_Slocal_Uconfig_Utf/_pywrap_tensorflow_internal.so' (no such file), '/Users/josh/miniconda3/envs/sharedhands-env/bin/../lib/_pywrap_tensorflow_internal.so' (no such file)
(sharedhands-env) Joshuas-MacBook-Pro:artifacts josh$ python -m pip uninstall -y tensorflow tensorflow-metal
Found existing installation: tensorflow 2.21.0
Uninstalling tensorflow-2.21.0:
  Successfully uninstalled tensorflow-2.21.0
Found existing installation: tensorflow-metal 1.2.0
Uninstalling tensorflow-metal-1.2.0:
  Successfully uninstalled tensorflow-metal-1.2.0
(sharedhands-env) Joshuas-MacBook-Pro:artifacts josh$ conda install -c apple tensorflow-deps -y
2 channel Terms of Service accepted
Channels:
 - apple
 - defaults
Platform: osx-arm64
Collecting package metadata (repodata.json): done
Solving environment: failed

LibMambaUnsatisfiableError: Encountered problems while solving:
  - nothing provides absl-py >=0.10,<0.11 needed by tensorflow-deps-2.5.0-0

Could not solve for environment specs
The following packages are incompatible
├─ pin on python =3.11 * is installable and it requires
│  └─ python =3.11 *, which can be installed;
└─ tensorflow-deps =* * is not installable because there are no viable options
   ├─ tensorflow-deps [2.5.0|2.6.0|2.7.0] would require
   │  └─ absl-py >=0.10,<0.11 *, which does not exist (perhaps a missing channel);
   ├─ tensorflow-deps [2.8.0|2.9.0] would require
   │  ├─ h5py >=3.6.0,<3.7 * but there are no viable options
   │  │  ├─ h5py 3.6.0 would require
   │  │  │  └─ python >=3.10,<3.11.0a0 *, which conflicts with any installable versions previously reported;
   │  │  ├─ h5py 3.6.0 would require
   │  │  │  └─ python >=3.8,<3.9.0a0 *, which conflicts with any installable versions previously reported;
   │  │  └─ h5py 3.6.0 would require
   │  │     └─ python >=3.9,<3.10.0a0 *, which conflicts with any installable versions previously reported;
   │  └─ numpy >=1.21.2,<1.22 * but there are no viable options
   │     ├─ numpy [1.21.2|1.21.5|1.21.6] would require
   │     │  └─ python >=3.10,<3.11.0a0 *, which conflicts with any installable versions previously reported;
   │     ├─ numpy [1.21.2|1.21.5|1.21.6] would require
   │     │  └─ python >=3.8,<3.9.0a0 *, which conflicts with any installable versions previously reported;
   │     └─ numpy [1.21.2|1.21.5|1.21.6] would require
   │        └─ python >=3.9,<3.10.0a0 *, which conflicts with any installable versions previously reported;
   └─ tensorflow-deps 2.10.0 would require
      └─ numpy >=1.23.2,<1.23.3 *, which does not exist (perhaps a missing channel).

Pins seem to be involved in the conflict. Currently pinned specs:
 - python=3.11


(sharedhands-env) Joshuas-MacBook-Pro:artifacts josh$ conda create -n sharedhands-py310 python=3.10 -y 
2 channel Terms of Service accepted
Channels:
 - defaults
Platform: osx-arm64
Collecting package metadata (repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /Users/josh/miniconda3/envs/sharedhands-py310

  added / updated specs:
    - python=3.10


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    python-3.10.20             |       h4f1bc5c_0        12.3 MB
    ------------------------------------------------------------
                                           Total:        12.3 MB

The following NEW packages will be INSTALLED:

  bzip2              pkgs/main/osx-arm64::bzip2-1.0.8-h80987f9_6 
  ca-certificates    pkgs/main/osx-arm64::ca-certificates-2025.12.2-hca03da5_0 
  libcxx             pkgs/main/osx-arm64::libcxx-21.1.8-hb4ce287_0 
  libexpat           pkgs/main/osx-arm64::libexpat-2.7.4-h50f4ffc_0 
  libffi             pkgs/main/osx-arm64::libffi-3.4.4-hca03da5_1 
  libzlib            pkgs/main/osx-arm64::libzlib-1.3.1-h5f15de7_0 
  ncurses            pkgs/main/osx-arm64::ncurses-6.5-hee39554_0 
  openssl            pkgs/main/osx-arm64::openssl-3.5.5-ha0b305a_0 
  packaging          pkgs/main/osx-arm64::packaging-25.0-py310hca03da5_1 
  pip                pkgs/main/noarch::pip-26.0.1-pyhc872135_0 
  python             pkgs/main/osx-arm64::python-3.10.20-h4f1bc5c_0 
  readline           pkgs/main/osx-arm64::readline-8.3-h0b18652_0 
  setuptools         pkgs/main/osx-arm64::setuptools-80.10.2-py310hca03da5_0 
  sqlite             pkgs/main/osx-arm64::sqlite-3.51.2-h67002bf_0 
  tk                 pkgs/main/osx-arm64::tk-8.6.15-hcd8a7d5_0 
  tzdata             pkgs/main/noarch::tzdata-2026a-he532380_0 
  wheel              pkgs/main/osx-arm64::wheel-0.46.3-py310hca03da5_0 
  xz                 pkgs/main/osx-arm64::xz-5.8.2-h8bbcb1d_0 
  zlib               pkgs/main/osx-arm64::zlib-1.3.1-h5f15de7_0 



Downloading and Extracting Packages:
                                                                                                                                                                                                     
Preparing transaction: done
Verifying transaction: done
Executing transaction: done
#
# To activate this environment, use
#
#     $ conda activate sharedhands-py310
#
# To deactivate an active environment, use
#
#     $ conda deactivate

(sharedhands-env) Joshuas-MacBook-Pro:artifacts josh$ conda activate sharedhands-py310
(sharedhands-py310) Joshuas-MacBook-Pro:artifacts josh$ conda install -c apple tensorflow-deps -y 
2 channel Terms of Service accepted
Channels:
 - apple
 - defaults
Platform: osx-arm64
Collecting package metadata (repodata.json): done
Solving environment: done

## Package Plan ##

  environment location: /Users/josh/miniconda3/envs/sharedhands-py310

  added / updated specs:
    - tensorflow-deps


The following packages will be downloaded:

    package                    |            build
    ---------------------------|-----------------
    blas-1.0                   |         openblas          10 KB
    grpcio-1.78.0              |  py310h1d739a6_0         699 KB
    h5py-3.6.0                 |  py310h181c318_0         1.0 MB
    hdf5-1.12.1                |       h05c076b_3         5.3 MB
    libabseil-20260107.0       | cxx17_hdc7b58b_0         1.2 MB
    libcurl-8.19.0             |       hc5dca5f_0         396 KB
    libgfortran-15.2.0         |       h09d7db9_1         130 KB
    libgfortran5-15.2.0        |       hb654fa1_1         742 KB
    libgrpc-1.78.0             |       h7694377_0         4.6 MB
    libopenblas-0.3.31         |       h7813bb4_0         5.6 MB
    libprotobuf-6.33.5         |       h55f9a99_0         2.6 MB
    libre2-11-2025.11.05       |       hce96306_1         160 KB
    llvm-openmp-21.1.8         |       hfab7639_0         295 KB
    numpy-1.22.3               |  py310h901140f_3          14 KB
    numpy-base-1.22.3          |  py310hae06d03_3         5.6 MB
    re2-2025.11.05             |       hff8a8d8_1          25 KB
    tensorflow-deps-2.9.0      |                0           3 KB  apple
    typing-extensions-4.15.0   |  py310hca03da5_0          12 KB
    typing_extensions-4.15.0   |  py310hca03da5_0          81 KB
    ------------------------------------------------------------
                                           Total:        28.3 MB

The following NEW packages will be INSTALLED:

  blas               pkgs/main/osx-arm64::blas-1.0-openblas 
  c-ares             pkgs/main/osx-arm64::c-ares-1.34.6-hfe05a68_0 
  gettext            pkgs/main/osx-arm64::gettext-0.25.1-h50c8ec2_0 
  gettext-tools      pkgs/main/osx-arm64::gettext-tools-0.25.1-h61de102_0 
  grpcio             pkgs/main/osx-arm64::grpcio-1.78.0-py310h1d739a6_0 
  h5py               pkgs/main/osx-arm64::h5py-3.6.0-py310h181c318_0 
  hdf5               pkgs/main/osx-arm64::hdf5-1.12.1-h05c076b_3 
  icu                pkgs/main/osx-arm64::icu-73.1-h313beb8_0 
  jansson            pkgs/main/osx-arm64::jansson-2.14-h80987f9_1 
  libabseil          pkgs/main/osx-arm64::libabseil-20260107.0-cxx17_hdc7b58b_0 
  libasprintf        pkgs/main/osx-arm64::libasprintf-0.25.1-h7b764f5_0 
  libasprintf-devel  pkgs/main/osx-arm64::libasprintf-devel-0.25.1-h7b764f5_0 
  libbrotlicommon    pkgs/main/osx-arm64::libbrotlicommon-1.2.0-hbd7815e_0 
  libbrotlidec       pkgs/main/osx-arm64::libbrotlidec-1.2.0-h1e834b2_0 
  libbrotlienc       pkgs/main/osx-arm64::libbrotlienc-1.2.0-h5439a07_0 
  libcurl            pkgs/main/osx-arm64::libcurl-8.19.0-hc5dca5f_0 
  libev              pkgs/main/osx-arm64::libev-4.33-h1a28f6b_1 
  libgettextpo       pkgs/main/osx-arm64::libgettextpo-0.25.1-h7b764f5_0 
  libgettextpo-devel pkgs/main/osx-arm64::libgettextpo-devel-0.25.1-h7b764f5_0 
  libgfortran        pkgs/main/osx-arm64::libgfortran-15.2.0-h09d7db9_1 
  libgfortran5       pkgs/main/osx-arm64::libgfortran5-15.2.0-hb654fa1_1 
  libgrpc            pkgs/main/osx-arm64::libgrpc-1.78.0-h7694377_0 
  libiconv           pkgs/main/osx-arm64::libiconv-1.18-h92f5915_0 
  libidn2            pkgs/main/osx-arm64::libidn2-2.3.8-h9681e36_0 
  libintl            pkgs/main/osx-arm64::libintl-0.25.1-h7b764f5_0 
  libintl-devel      pkgs/main/osx-arm64::libintl-devel-0.25.1-h7b764f5_0 
  libkrb5            pkgs/main/osx-arm64::libkrb5-1.22.1-ha46c28b_0 
  libnghttp2         pkgs/main/osx-arm64::libnghttp2-1.67.1-h8189af8_0 
  libopenblas        pkgs/main/osx-arm64::libopenblas-0.3.31-h7813bb4_0 
  libprotobuf        pkgs/main/osx-arm64::libprotobuf-6.33.5-h55f9a99_0 
  libre2-11          pkgs/main/osx-arm64::libre2-11-2025.11.05-hce96306_1 
  libssh2            pkgs/main/osx-arm64::libssh2-1.11.1-h3e2b118_0 
  libunistring       pkgs/main/osx-arm64::libunistring-1.3-h1799b2a_0 
  libxml2            pkgs/main/osx-arm64::libxml2-2.13.9-h528a072_0 
  llvm-openmp        pkgs/main/osx-arm64::llvm-openmp-21.1.8-hfab7639_0 
  lmdb               pkgs/main/osx-arm64::lmdb-0.9.31-h79febb2_0 
  lz4-c              pkgs/main/osx-arm64::lz4-c-1.9.4-h313beb8_1 
  numpy              pkgs/main/osx-arm64::numpy-1.22.3-py310h901140f_3 
  numpy-base         pkgs/main/osx-arm64::numpy-base-1.22.3-py310hae06d03_3 
  re2                pkgs/main/osx-arm64::re2-2025.11.05-hff8a8d8_1 
  tensorflow-deps    apple/osx-arm64::tensorflow-deps-2.9.0-0 
  typing-extensions  pkgs/main/osx-arm64::typing-extensions-4.15.0-py310hca03da5_0 
  typing_extensions  pkgs/main/osx-arm64::typing_extensions-4.15.0-py310hca03da5_0 
  zstd               pkgs/main/osx-arm64::zstd-1.5.7-h817c040_0 



Downloading and Extracting Packages:
                                                                                                                                                                                                     
Preparing transaction: done                                                                                                                                                                          
Verifying transaction: done                                                                                                                                                                          
Executing transaction: done                                                                                                                                                                          
(sharedhands-py310) Joshuas-MacBook-Pro:artifacts josh$ python -m pip install opencv-python mediapipe numpy keras scikit-learn pandas tensorflow-macos tensorflow-metal                              
Collecting opencv-python                                                                                                                                                                             
  Using cached opencv_python-4.13.0.92-cp37-abi3-macosx_13_0_arm64.whl.metadata (19 kB)                                                                                                              
Collecting mediapipe                                                                                                                                                                                 
  Using cached mediapipe-0.10.33-py3-none-macosx_11_0_arm64.whl.metadata (9.8 kB)                                                                                                                    
Requirement already satisfied: numpy in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (1.22.3)                                                                          
Collecting keras                                                                                                                                                                                     
  Downloading keras-3.12.1-py3-none-any.whl.metadata (5.9 kB)                                                                                                                                        
Collecting scikit-learn                                                                                                                                                                              
  Downloading scikit_learn-1.7.2-cp310-cp310-macosx_12_0_arm64.whl.metadata (11 kB)                                                                                                                  
Collecting pandas                                                                                                                                                                                    
  Using cached pandas-2.3.3-cp310-cp310-macosx_11_0_arm64.whl.metadata (91 kB)                                                                                                                       
Collecting tensorflow-macos                                                                                                                                                                          
  Downloading tensorflow_macos-2.16.2-cp310-cp310-macosx_12_0_arm64.whl.metadata (3.3 kB)                                                                                                            
Collecting tensorflow-metal                                                                                                                                                                          
  Downloading tensorflow_metal-1.2.0-cp310-cp310-macosx_12_0_arm64.whl.metadata (1.3 kB)
Collecting numpy
  Using cached numpy-2.2.6-cp310-cp310-macosx_14_0_arm64.whl.metadata (62 kB)
Collecting absl-py~=2.3 (from mediapipe)
  Using cached absl_py-2.4.0-py3-none-any.whl.metadata (3.3 kB)
Collecting sounddevice~=0.5 (from mediapipe)
  Using cached sounddevice-0.5.5-py3-none-macosx_10_6_x86_64.macosx_10_6_universal2.whl.metadata (1.4 kB)
Collecting flatbuffers~=25.9 (from mediapipe)
  Using cached flatbuffers-25.12.19-py2.py3-none-any.whl.metadata (1.0 kB)
Collecting opencv-contrib-python (from mediapipe)
  Using cached opencv_contrib_python-4.13.0.92-cp37-abi3-macosx_13_0_arm64.whl.metadata (19 kB)
Collecting matplotlib (from mediapipe)
  Downloading matplotlib-3.10.8-cp310-cp310-macosx_11_0_arm64.whl.metadata (52 kB)
Collecting cffi (from sounddevice~=0.5->mediapipe)
  Downloading cffi-2.0.0-cp310-cp310-macosx_11_0_arm64.whl.metadata (2.6 kB)
Collecting rich (from keras)
  Using cached rich-14.3.3-py3-none-any.whl.metadata (18 kB)
Collecting namex (from keras)
  Using cached namex-0.1.0-py3-none-any.whl.metadata (322 bytes)
Requirement already satisfied: h5py in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from keras) (3.6.0)
Collecting optree (from keras)
  Downloading optree-0.19.0-cp310-cp310-macosx_11_0_arm64.whl.metadata (34 kB)
Collecting ml-dtypes (from keras)
  Downloading ml_dtypes-0.5.4-cp310-cp310-macosx_10_9_universal2.whl.metadata (8.9 kB)
Requirement already satisfied: packaging in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from keras) (25.0)
Collecting scipy>=1.8.0 (from scikit-learn)
  Downloading scipy-1.15.3-cp310-cp310-macosx_14_0_arm64.whl.metadata (61 kB)
Collecting joblib>=1.2.0 (from scikit-learn)
  Using cached joblib-1.5.3-py3-none-any.whl.metadata (5.5 kB)
Collecting threadpoolctl>=3.1.0 (from scikit-learn)
  Using cached threadpoolctl-3.6.0-py3-none-any.whl.metadata (13 kB)
Collecting python-dateutil>=2.8.2 (from pandas)
  Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl.metadata (8.4 kB)
Collecting pytz>=2020.1 (from pandas)
  Using cached pytz-2026.1.post1-py2.py3-none-any.whl.metadata (22 kB)
Collecting tzdata>=2022.7 (from pandas)
  Using cached tzdata-2025.3-py2.py3-none-any.whl.metadata (1.4 kB)
Collecting tensorflow==2.16.2 (from tensorflow-macos)
  Downloading tensorflow-2.16.2-cp310-cp310-macosx_12_0_arm64.whl.metadata (4.1 kB)
Collecting astunparse>=1.6.0 (from tensorflow==2.16.2->tensorflow-macos)
  Using cached astunparse-1.6.3-py2.py3-none-any.whl.metadata (4.4 kB)
Collecting gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 (from tensorflow==2.16.2->tensorflow-macos)
  Using cached gast-0.7.0-py3-none-any.whl.metadata (1.5 kB)
Collecting google-pasta>=0.1.1 (from tensorflow==2.16.2->tensorflow-macos)
  Using cached google_pasta-0.2.0-py3-none-any.whl.metadata (814 bytes)
Collecting h5py (from keras)
  Downloading h5py-3.16.0-cp310-cp310-macosx_11_0_arm64.whl.metadata (3.0 kB)
Collecting libclang>=13.0.0 (from tensorflow==2.16.2->tensorflow-macos)
  Using cached libclang-18.1.1-1-py2.py3-none-macosx_11_0_arm64.whl.metadata (5.2 kB)
Collecting ml-dtypes (from keras)
  Downloading ml_dtypes-0.3.2-cp310-cp310-macosx_10_9_universal2.whl.metadata (20 kB)
Collecting opt-einsum>=2.3.2 (from tensorflow==2.16.2->tensorflow-macos)
  Using cached opt_einsum-3.4.0-py3-none-any.whl.metadata (6.3 kB)
Collecting protobuf!=4.21.0,!=4.21.1,!=4.21.2,!=4.21.3,!=4.21.4,!=4.21.5,<5.0.0dev,>=3.20.3 (from tensorflow==2.16.2->tensorflow-macos)
  Using cached protobuf-4.25.8-cp37-abi3-macosx_10_9_universal2.whl.metadata (541 bytes)
Collecting requests<3,>=2.21.0 (from tensorflow==2.16.2->tensorflow-macos)
  Using cached requests-2.32.5-py3-none-any.whl.metadata (4.9 kB)
Requirement already satisfied: setuptools in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from tensorflow==2.16.2->tensorflow-macos) (80.10.2)
Collecting six>=1.12.0 (from tensorflow==2.16.2->tensorflow-macos)
  Using cached six-1.17.0-py2.py3-none-any.whl.metadata (1.7 kB)
Collecting termcolor>=1.1.0 (from tensorflow==2.16.2->tensorflow-macos)
  Using cached termcolor-3.3.0-py3-none-any.whl.metadata (6.5 kB)
Requirement already satisfied: typing-extensions>=3.6.6 in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from tensorflow==2.16.2->tensorflow-macos) (4.15.0)
Collecting wrapt>=1.11.0 (from tensorflow==2.16.2->tensorflow-macos)
  Downloading wrapt-2.1.2-cp310-cp310-macosx_11_0_arm64.whl.metadata (7.4 kB)
Requirement already satisfied: grpcio<2.0,>=1.24.3 in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from tensorflow==2.16.2->tensorflow-macos) (1.78.0)
Collecting tensorboard<2.17,>=2.16 (from tensorflow==2.16.2->tensorflow-macos)
  Using cached tensorboard-2.16.2-py3-none-any.whl.metadata (1.6 kB)
Collecting tensorflow-io-gcs-filesystem>=0.23.1 (from tensorflow==2.16.2->tensorflow-macos)
  Downloading tensorflow_io_gcs_filesystem-0.37.1-cp310-cp310-macosx_12_0_arm64.whl.metadata (14 kB)
INFO: pip is looking at multiple versions of tensorflow to determine which version is compatible with other requirements. This could take a while.
Collecting tensorflow-macos
  Downloading tensorflow_macos-2.16.1-cp310-cp310-macosx_12_0_arm64.whl.metadata (3.5 kB)
Collecting tensorflow==2.16.1 (from tensorflow-macos)
  Downloading tensorflow-2.16.1-cp310-cp310-macosx_12_0_arm64.whl.metadata (4.1 kB)
Collecting tensorflow-macos
  Downloading tensorflow_macos-2.15.1-cp310-cp310-macosx_12_0_arm64.whl.metadata (3.4 kB)
Requirement already satisfied: wheel~=0.35 in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from tensorflow-metal) (0.46.3)
Collecting pycparser (from cffi->sounddevice~=0.5->mediapipe)
  Using cached pycparser-3.0-py3-none-any.whl.metadata (8.2 kB)
Collecting contourpy>=1.0.1 (from matplotlib->mediapipe)
  Downloading contourpy-1.3.2-cp310-cp310-macosx_11_0_arm64.whl.metadata (5.5 kB)
Collecting cycler>=0.10 (from matplotlib->mediapipe)
  Using cached cycler-0.12.1-py3-none-any.whl.metadata (3.8 kB)
Collecting fonttools>=4.22.0 (from matplotlib->mediapipe)
  Downloading fonttools-4.62.1-cp310-cp310-macosx_10_9_universal2.whl.metadata (117 kB)
Collecting kiwisolver>=1.3.1 (from matplotlib->mediapipe)
  Downloading kiwisolver-1.5.0-cp310-cp310-macosx_11_0_arm64.whl.metadata (5.1 kB)
Collecting pillow>=8 (from matplotlib->mediapipe)
  Using cached pillow-12.1.1-cp310-cp310-macosx_11_0_arm64.whl.metadata (8.8 kB)
Collecting pyparsing>=3 (from matplotlib->mediapipe)
  Using cached pyparsing-3.3.2-py3-none-any.whl.metadata (5.8 kB)
Collecting markdown-it-py>=2.2.0 (from rich->keras)
  Using cached markdown_it_py-4.0.0-py3-none-any.whl.metadata (7.3 kB)
Collecting pygments<3.0.0,>=2.13.0 (from rich->keras)
  Using cached pygments-2.19.2-py3-none-any.whl.metadata (2.5 kB)
Collecting mdurl~=0.1 (from markdown-it-py>=2.2.0->rich->keras)
  Using cached mdurl-0.1.2-py3-none-any.whl.metadata (1.6 kB)
Using cached opencv_python-4.13.0.92-cp37-abi3-macosx_13_0_arm64.whl (46.2 MB)
Using cached mediapipe-0.10.33-py3-none-macosx_11_0_arm64.whl (29.4 MB)
Using cached absl_py-2.4.0-py3-none-any.whl (135 kB)
Using cached flatbuffers-25.12.19-py2.py3-none-any.whl (26 kB)
Using cached sounddevice-0.5.5-py3-none-macosx_10_6_x86_64.macosx_10_6_universal2.whl (108 kB)
Using cached numpy-2.2.6-cp310-cp310-macosx_14_0_arm64.whl (5.3 MB)
Downloading keras-3.12.1-py3-none-any.whl (1.5 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.5/1.5 MB 13.3 MB/s  0:00:00
Downloading scikit_learn-1.7.2-cp310-cp310-macosx_12_0_arm64.whl (8.7 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.7/8.7 MB 22.9 MB/s  0:00:00
Using cached pandas-2.3.3-cp310-cp310-macosx_11_0_arm64.whl (10.8 MB)
Downloading tensorflow_macos-2.15.1-cp310-cp310-macosx_12_0_arm64.whl (2.2 kB)
Downloading tensorflow_metal-1.2.0-cp310-cp310-macosx_12_0_arm64.whl (1.4 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 1.4/1.4 MB 17.7 MB/s  0:00:00
Using cached joblib-1.5.3-py3-none-any.whl (309 kB)
Using cached python_dateutil-2.9.0.post0-py2.py3-none-any.whl (229 kB)
Using cached pytz-2026.1.post1-py2.py3-none-any.whl (510 kB)
Downloading scipy-1.15.3-cp310-cp310-macosx_14_0_arm64.whl (22.4 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 22.4/22.4 MB 16.9 MB/s  0:00:01
Using cached six-1.17.0-py2.py3-none-any.whl (11 kB)
Using cached threadpoolctl-3.6.0-py3-none-any.whl (18 kB)
Using cached tzdata-2025.3-py2.py3-none-any.whl (348 kB)
Downloading cffi-2.0.0-cp310-cp310-macosx_11_0_arm64.whl (180 kB)
Downloading matplotlib-3.10.8-cp310-cp310-macosx_11_0_arm64.whl (8.1 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 8.1/8.1 MB 5.7 MB/s  0:00:01
Downloading contourpy-1.3.2-cp310-cp310-macosx_11_0_arm64.whl (253 kB)
Using cached cycler-0.12.1-py3-none-any.whl (8.3 kB)
Downloading fonttools-4.62.1-cp310-cp310-macosx_10_9_universal2.whl (2.9 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.9/2.9 MB 4.3 MB/s  0:00:00
Downloading kiwisolver-1.5.0-cp310-cp310-macosx_11_0_arm64.whl (63 kB)
Using cached pillow-12.1.1-cp310-cp310-macosx_11_0_arm64.whl (4.7 MB)
Using cached pyparsing-3.3.2-py3-none-any.whl (122 kB)
Downloading ml_dtypes-0.5.4-cp310-cp310-macosx_10_9_universal2.whl (679 kB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 679.7/679.7 kB 4.5 MB/s  0:00:00
Using cached namex-0.1.0-py3-none-any.whl (5.9 kB)
Using cached opencv_contrib_python-4.13.0.92-cp37-abi3-macosx_13_0_arm64.whl (52.0 MB)
Downloading optree-0.19.0-cp310-cp310-macosx_11_0_arm64.whl (363 kB)
Using cached pycparser-3.0-py3-none-any.whl (48 kB)
Using cached rich-14.3.3-py3-none-any.whl (310 kB)
Using cached pygments-2.19.2-py3-none-any.whl (1.2 MB)
Using cached markdown_it_py-4.0.0-py3-none-any.whl (87 kB)
Using cached mdurl-0.1.2-py3-none-any.whl (10.0 kB)
Installing collected packages: pytz, namex, flatbuffers, tzdata, threadpoolctl, six, pyparsing, pygments, pycparser, pillow, optree, numpy, mdurl, kiwisolver, joblib, fonttools, cycler, absl-py, tensorflow-metal, scipy, python-dateutil, opencv-python, opencv-contrib-python, ml-dtypes, markdown-it-py, contourpy, cffi, sounddevice, scikit-learn, rich, pandas, matplotlib, mediapipe, keras, tensorflow-macos
  Attempting uninstall: numpy
    Found existing installation: numpy 1.22.3
    Uninstalling numpy-1.22.3:
      Successfully uninstalled numpy-1.22.3
Successfully installed absl-py-2.4.0 cffi-2.0.0 contourpy-1.3.2 cycler-0.12.1 flatbuffers-25.12.19 fonttools-4.62.1 joblib-1.5.3 keras-3.12.1 kiwisolver-1.5.0 markdown-it-py-4.0.0 matplotlib-3.10.8 mdurl-0.1.2 mediapipe-0.10.33 ml-dtypes-0.5.4 namex-0.1.0 numpy-2.2.6 opencv-contrib-python-4.13.0.92 opencv-python-4.13.0.92 optree-0.19.0 pandas-2.3.3 pillow-12.1.1 pycparser-3.0 pygments-2.19.2 pyparsing-3.3.2 python-dateutil-2.9.0.post0 pytz-2026.1.post1 rich-14.3.3 scikit-learn-1.7.2 scipy-1.15.3 six-1.17.0 sounddevice-0.5.5 tensorflow-macos-2.15.1 tensorflow-metal-1.2.0 threadpoolctl-3.6.0 tzdata-2025.3
(sharedhands-py310) Joshuas-MacBook-Pro:artifacts josh$ python ASL_Version\ 1.0.py 
Traceback (most recent call last):
  File "/Users/josh/Team4_SharedHands/Team4_SharedHands/artifacts/ASL_Version 1.0.py", line 5, in <module>
    import keras
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/keras/__init__.py", line 7, in <module>
    from keras import _tf_keras as _tf_keras
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/keras/_tf_keras/__init__.py", line 1, in <module>
    from keras._tf_keras import keras
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/keras/_tf_keras/keras/__init__.py", line 7, in <module>
    from keras import activations as activations
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/keras/activations/__init__.py", line 7, in <module>
    from keras.src.activations import deserialize as deserialize
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/keras/src/__init__.py", line 1, in <module>
    from keras.src import activations
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/keras/src/activations/__init__.py", line 3, in <module>
    from keras.src.activations.activations import celu
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/keras/src/activations/activations.py", line 1, in <module>
    from keras.src import backend
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/keras/src/backend/__init__.py", line 1, in <module>
    from keras.src.backend.config import backend
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/keras/src/backend/config.py", line 448, in <module>
    set_nnx_enabled(_NNX_ENABLED)
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/keras/src/backend/config.py", line 249, in set_nnx_enabled
    from keras.src.backend.common import global_state
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/keras/src/backend/common/__init__.py", line 2, in <module>
    from keras.src.backend.common.dtypes import result_type
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/keras/src/backend/common/dtypes.py", line 5, in <module>
    from keras.src.backend.common.variables import standardize_dtype
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/keras/src/backend/common/variables.py", line 11, in <module>
    from keras.src.utils.module_utils import tensorflow as tf
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/keras/src/utils/__init__.py", line 1, in <module>
    from keras.src.utils.audio_dataset_utils import audio_dataset_from_directory
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/keras/src/utils/audio_dataset_utils.py", line 4, in <module>
    from keras.src.utils import dataset_utils
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/keras/src/utils/dataset_utils.py", line 10, in <module>
    from keras.src import tree
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/keras/src/tree/__init__.py", line 1, in <module>
    from keras.src.tree.tree_api import assert_same_paths
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/keras/src/tree/tree_api.py", line 13, in <module>
    from keras.src.tree import optree_impl as tree_impl
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/keras/src/tree/optree_impl.py", line 13, in <module>
    from tensorflow.python.trackable.data_structures import ListWrapper
ModuleNotFoundError: No module named 'tensorflow'
(sharedhands-py310) Joshuas-MacBook-Pro:artifacts josh$ python -m pip uninstall -y tensorflow-macos
Found existing installation: tensorflow-macos 2.15.1
Uninstalling tensorflow-macos-2.15.1:
  Successfully uninstalled tensorflow-macos-2.15.1
(sharedhands-py310) Joshuas-MacBook-Pro:artifacts josh$ python -m pip install tensorflow tensorflow-metal
Collecting tensorflow
  Downloading tensorflow-2.21.0-cp310-cp310-macosx_12_0_arm64.whl.metadata (4.4 kB)
Requirement already satisfied: tensorflow-metal in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (1.2.0)
Requirement already satisfied: absl-py>=1.0.0 in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from tensorflow) (2.4.0)
Collecting astunparse>=1.6.0 (from tensorflow)
  Using cached astunparse-1.6.3-py2.py3-none-any.whl.metadata (4.4 kB)
Requirement already satisfied: flatbuffers>=25.9.23 in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from tensorflow) (25.12.19)
Collecting gast!=0.5.0,!=0.5.1,!=0.5.2,>=0.2.1 (from tensorflow)
  Using cached gast-0.7.0-py3-none-any.whl.metadata (1.5 kB)
Collecting google_pasta>=0.1.1 (from tensorflow)
  Using cached google_pasta-0.2.0-py3-none-any.whl.metadata (814 bytes)
Collecting libclang>=13.0.0 (from tensorflow)
  Using cached libclang-18.1.1-1-py2.py3-none-macosx_11_0_arm64.whl.metadata (5.2 kB)
Collecting opt_einsum>=2.3.2 (from tensorflow)
  Using cached opt_einsum-3.4.0-py3-none-any.whl.metadata (6.3 kB)
Requirement already satisfied: packaging in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from tensorflow) (25.0)
Collecting protobuf<8.0.0,>=6.31.1 (from tensorflow)
  Using cached protobuf-7.34.1-cp310-abi3-macosx_10_9_universal2.whl.metadata (595 bytes)
Collecting requests<3,>=2.21.0 (from tensorflow)
  Using cached requests-2.32.5-py3-none-any.whl.metadata (4.9 kB)
Requirement already satisfied: setuptools in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from tensorflow) (80.10.2)
Requirement already satisfied: six>=1.12.0 in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from tensorflow) (1.17.0)
Collecting termcolor>=1.1.0 (from tensorflow)
  Using cached termcolor-3.3.0-py3-none-any.whl.metadata (6.5 kB)
Requirement already satisfied: typing_extensions>=3.6.6 in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from tensorflow) (4.15.0)
Collecting wrapt>=1.11.0 (from tensorflow)
  Using cached wrapt-2.1.2-cp310-cp310-macosx_11_0_arm64.whl.metadata (7.4 kB)
Requirement already satisfied: grpcio<2.0,>=1.24.3 in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from tensorflow) (1.78.0)
Requirement already satisfied: keras>=3.12.0 in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from tensorflow) (3.12.1)
Requirement already satisfied: numpy>=1.26.0 in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from tensorflow) (2.2.6)
Collecting h5py<3.15.0,>=3.11.0 (from tensorflow)
  Downloading h5py-3.14.0-cp310-cp310-macosx_11_0_arm64.whl.metadata (2.7 kB)
Requirement already satisfied: ml_dtypes<1.0.0,>=0.5.1 in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from tensorflow) (0.5.4)
Collecting charset_normalizer<4,>=2 (from requests<3,>=2.21.0->tensorflow)
  Downloading charset_normalizer-3.4.6-cp310-cp310-macosx_10_9_universal2.whl.metadata (40 kB)
Collecting idna<4,>=2.5 (from requests<3,>=2.21.0->tensorflow)
  Using cached idna-3.11-py3-none-any.whl.metadata (8.4 kB)
Collecting urllib3<3,>=1.21.1 (from requests<3,>=2.21.0->tensorflow)
  Using cached urllib3-2.6.3-py3-none-any.whl.metadata (6.9 kB)
Collecting certifi>=2017.4.17 (from requests<3,>=2.21.0->tensorflow)
  Using cached certifi-2026.2.25-py3-none-any.whl.metadata (2.5 kB)
Requirement already satisfied: wheel~=0.35 in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from tensorflow-metal) (0.46.3)
Requirement already satisfied: rich in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from keras>=3.12.0->tensorflow) (14.3.3)
Requirement already satisfied: namex in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from keras>=3.12.0->tensorflow) (0.1.0)
Requirement already satisfied: optree in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from keras>=3.12.0->tensorflow) (0.19.0)
Requirement already satisfied: markdown-it-py>=2.2.0 in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from rich->keras>=3.12.0->tensorflow) (4.0.0)
Requirement already satisfied: pygments<3.0.0,>=2.13.0 in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from rich->keras>=3.12.0->tensorflow) (2.19.2)
Requirement already satisfied: mdurl~=0.1 in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from markdown-it-py>=2.2.0->rich->keras>=3.12.0->tensorflow) (0.1.2)
Downloading tensorflow-2.21.0-cp310-cp310-macosx_12_0_arm64.whl (223.2 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 223.2/223.2 MB 27.3 MB/s  0:00:08
Downloading h5py-3.14.0-cp310-cp310-macosx_11_0_arm64.whl (2.8 MB)
   ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━ 2.8/2.8 MB 23.2 MB/s  0:00:00
Using cached protobuf-7.34.1-cp310-abi3-macosx_10_9_universal2.whl (429 kB)
Using cached requests-2.32.5-py3-none-any.whl (64 kB)
Downloading charset_normalizer-3.4.6-cp310-cp310-macosx_10_9_universal2.whl (298 kB)
Using cached idna-3.11-py3-none-any.whl (71 kB)
Using cached urllib3-2.6.3-py3-none-any.whl (131 kB)
Using cached astunparse-1.6.3-py2.py3-none-any.whl (12 kB)
Using cached certifi-2026.2.25-py3-none-any.whl (153 kB)
Using cached gast-0.7.0-py3-none-any.whl (22 kB)
Using cached google_pasta-0.2.0-py3-none-any.whl (57 kB)
Using cached libclang-18.1.1-1-py2.py3-none-macosx_11_0_arm64.whl (25.8 MB)
Using cached opt_einsum-3.4.0-py3-none-any.whl (71 kB)
Using cached termcolor-3.3.0-py3-none-any.whl (7.7 kB)
Downloading wrapt-2.1.2-cp310-cp310-macosx_11_0_arm64.whl (61 kB)
Installing collected packages: libclang, wrapt, urllib3, termcolor, protobuf, opt_einsum, idna, h5py, google_pasta, gast, charset_normalizer, certifi, requests, astunparse, tensorflow
  Attempting uninstall: h5py
    Found existing installation: h5py 3.6.0
    Uninstalling h5py-3.6.0:
      Successfully uninstalled h5py-3.6.0
Successfully installed astunparse-1.6.3 certifi-2026.2.25 charset_normalizer-3.4.6 gast-0.7.0 google_pasta-0.2.0 h5py-3.14.0 idna-3.11 libclang-18.1.1 opt_einsum-3.4.0 protobuf-7.34.1 requests-2.32.5 tensorflow-2.21.0 termcolor-3.3.0 urllib3-2.6.3 wrapt-2.1.2
(sharedhands-py310) Joshuas-MacBook-Pro:artifacts josh$ python ASL_Version\ 1.0.py 
Traceback (most recent call last):
  File "/Users/josh/Team4_SharedHands/Team4_SharedHands/artifacts/ASL_Version 1.0.py", line 2, in <module>
    import mediapipe as mp
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/mediapipe/__init__.py", line 15, in <module>
    import mediapipe.tasks.python as tasks
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/mediapipe/tasks/python/__init__.py", line 17, in <module>
    from . import audio
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/mediapipe/tasks/python/audio/__init__.py", line 18, in <module>
    import mediapipe.tasks.python.audio.audio_classifier
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/mediapipe/tasks/python/audio/audio_classifier.py", line 21, in <module>
    from mediapipe.tasks.python.audio.core import audio_task_running_mode
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/mediapipe/tasks/python/audio/core/audio_task_running_mode.py", line 19, in <module>
    from mediapipe.tasks.python.core.optional_dependencies import doc_controls
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/mediapipe/tasks/python/core/optional_dependencies.py", line 20, in <module>
    from tensorflow.tools.docs import doc_controls
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/tensorflow/__init__.py", line 439, in <module>
    _ll.load_library(_plugin_dir)
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/tensorflow/python/framework/load_library.py", line 151, in load_library
    py_tf.TF_LoadLibrary(lib)
tensorflow.python.framework.errors_impl.NotFoundError: dlopen(/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/tensorflow-plugins/libmetal_plugin.dylib, 0x0006): Library not loaded: @rpath/_pywrap_tensorflow_internal.so
  Referenced from: <8B62586B-B082-3113-93AB-FD766A9960AE> /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/tensorflow-plugins/libmetal_plugin.dylib
  Reason: tried: '/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/tensorflow-plugins/../_solib_darwin_arm64/_U@local_Uconfig_Utf_S_S_C_Upywrap_Utensorflow_Uinternal___Uexternal_Slocal_Uconfig_Utf/_pywrap_tensorflow_internal.so' (no such file), '/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/tensorflow-plugins/../_solib_darwin_arm64/_U@local_Uconfig_Utf_S_S_C_Upywrap_Utensorflow_Uinternal___Uexternal_Slocal_Uconfig_Utf/_pywrap_tensorflow_internal.so' (no such file), '/Users/josh/miniconda3/envs/sharedhands-py310/bin/../lib/_pywrap_tensorflow_internal.so' (no such file)
(sharedhands-py310) Joshuas-MacBook-Pro:artifacts josh$ python -m pip uninstall -y mediapipe
Found existing installation: mediapipe 0.10.33
Uninstalling mediapipe-0.10.33:
  Successfully uninstalled mediapipe-0.10.33
(sharedhands-py310) Joshuas-MacBook-Pro:artifacts josh$ python -m pip install mediapipe==0.8.9.1
ERROR: Could not find a version that satisfies the requirement mediapipe==0.8.9.1 (from versions: 0.10.5, 0.10.7, 0.10.8, 0.10.9, 0.10.10, 0.10.11, 0.10.13, 0.10.14, 0.10.15, 0.10.18, 0.10.20, 0.10.21, 0.10.30, 0.10.31, 0.10.32, 0.10.33)
ERROR: No matching distribution found for mediapipe==0.8.9.1
(sharedhands-py310) Joshuas-MacBook-Pro:artifacts josh$ python ASL_Version\ 1.0.py
Traceback (most recent call last):
  File "/Users/josh/Team4_SharedHands/Team4_SharedHands/artifacts/ASL_Version 1.0.py", line 2, in <module>
    from mediapipe import solutions
ModuleNotFoundError: No module named 'mediapipe'
(sharedhands-py310) Joshuas-MacBook-Pro:artifacts josh$ pip install mediapipe
Requirement already satisfied: mediapipe in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (0.10.33)
Requirement already satisfied: absl-py~=2.3 in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from mediapipe) (2.4.0)
Requirement already satisfied: numpy in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from mediapipe) (2.4.3)
Requirement already satisfied: sounddevice~=0.5 in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from mediapipe) (0.5.5)
Requirement already satisfied: flatbuffers~=25.9 in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from mediapipe) (25.12.19)
Requirement already satisfied: opencv-contrib-python in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from mediapipe) (4.13.0.92)
Requirement already satisfied: matplotlib in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from mediapipe) (3.10.8)
Requirement already satisfied: cffi in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from sounddevice~=0.5->mediapipe) (2.0.0)
Requirement already satisfied: pycparser in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from cffi->sounddevice~=0.5->mediapipe) (3.0)
Requirement already satisfied: contourpy>=1.0.1 in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from matplotlib->mediapipe) (1.3.3)
Requirement already satisfied: cycler>=0.10 in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from matplotlib->mediapipe) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from matplotlib->mediapipe) (4.62.1)
Requirement already satisfied: kiwisolver>=1.3.1 in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from matplotlib->mediapipe) (1.5.0)
Requirement already satisfied: packaging>=20.0 in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from matplotlib->mediapipe) (26.0)
Requirement already satisfied: pillow>=8 in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from matplotlib->mediapipe) (12.1.1)
Requirement already satisfied: pyparsing>=3 in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from matplotlib->mediapipe) (3.3.2)
Requirement already satisfied: python-dateutil>=2.7 in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from matplotlib->mediapipe) (2.9.0.post0)
Requirement already satisfied: six>=1.5 in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from python-dateutil>=2.7->matplotlib->mediapipe) (1.17.0)
(sharedhands-py310) Joshuas-MacBook-Pro:artifacts josh$ python ASL_Version\ 1.0.py
Traceback (most recent call last):
  File "/Users/josh/Team4_SharedHands/Team4_SharedHands/artifacts/ASL_Version 1.0.py", line 2, in <module>
    from mediapipe import solutions
ModuleNotFoundError: No module named 'mediapipe'
(sharedhands-py310) Joshuas-MacBook-Pro:artifacts josh$ pip3 install mediapipe
Requirement already satisfied: mediapipe in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (0.10.33)
Requirement already satisfied: absl-py~=2.3 in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from mediapipe) (2.4.0)
Requirement already satisfied: numpy in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from mediapipe) (2.4.3)
Requirement already satisfied: sounddevice~=0.5 in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from mediapipe) (0.5.5)
Requirement already satisfied: flatbuffers~=25.9 in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from mediapipe) (25.12.19)
Requirement already satisfied: opencv-contrib-python in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from mediapipe) (4.13.0.92)
Requirement already satisfied: matplotlib in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from mediapipe) (3.10.8)
Requirement already satisfied: cffi in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from sounddevice~=0.5->mediapipe) (2.0.0)
Requirement already satisfied: pycparser in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from cffi->sounddevice~=0.5->mediapipe) (3.0)
Requirement already satisfied: contourpy>=1.0.1 in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from matplotlib->mediapipe) (1.3.3)
Requirement already satisfied: cycler>=0.10 in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from matplotlib->mediapipe) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from matplotlib->mediapipe) (4.62.1)
Requirement already satisfied: kiwisolver>=1.3.1 in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from matplotlib->mediapipe) (1.5.0)
Requirement already satisfied: packaging>=20.0 in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from matplotlib->mediapipe) (26.0)
Requirement already satisfied: pillow>=8 in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from matplotlib->mediapipe) (12.1.1)
Requirement already satisfied: pyparsing>=3 in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from matplotlib->mediapipe) (3.3.2)
Requirement already satisfied: python-dateutil>=2.7 in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from matplotlib->mediapipe) (2.9.0.post0)
Requirement already satisfied: six>=1.5 in /Library/Frameworks/Python.framework/Versions/3.14/lib/python3.14/site-packages (from python-dateutil>=2.7->matplotlib->mediapipe) (1.17.0)
(sharedhands-py310) Joshuas-MacBook-Pro:artifacts josh$ python -m pip install mediapipe
Collecting mediapipe
  Using cached mediapipe-0.10.33-py3-none-macosx_11_0_arm64.whl.metadata (9.8 kB)
Requirement already satisfied: absl-py~=2.3 in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from mediapipe) (2.4.0)
Requirement already satisfied: numpy in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from mediapipe) (2.2.6)
Requirement already satisfied: sounddevice~=0.5 in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from mediapipe) (0.5.5)
Requirement already satisfied: flatbuffers~=25.9 in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from mediapipe) (25.12.19)
Requirement already satisfied: opencv-contrib-python in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from mediapipe) (4.13.0.92)
Requirement already satisfied: matplotlib in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from mediapipe) (3.10.8)
Requirement already satisfied: cffi in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from sounddevice~=0.5->mediapipe) (2.0.0)
Requirement already satisfied: pycparser in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from cffi->sounddevice~=0.5->mediapipe) (3.0)
Requirement already satisfied: contourpy>=1.0.1 in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from matplotlib->mediapipe) (1.3.2)
Requirement already satisfied: cycler>=0.10 in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from matplotlib->mediapipe) (0.12.1)
Requirement already satisfied: fonttools>=4.22.0 in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from matplotlib->mediapipe) (4.62.1)
Requirement already satisfied: kiwisolver>=1.3.1 in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from matplotlib->mediapipe) (1.5.0)
Requirement already satisfied: packaging>=20.0 in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from matplotlib->mediapipe) (25.0)
Requirement already satisfied: pillow>=8 in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from matplotlib->mediapipe) (12.1.1)
Requirement already satisfied: pyparsing>=3 in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from matplotlib->mediapipe) (3.3.2)
Requirement already satisfied: python-dateutil>=2.7 in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from matplotlib->mediapipe) (2.9.0.post0)
Requirement already satisfied: six>=1.5 in /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages (from python-dateutil>=2.7->matplotlib->mediapipe) (1.17.0)
Using cached mediapipe-0.10.33-py3-none-macosx_11_0_arm64.whl (29.4 MB)
Installing collected packages: mediapipe
Successfully installed mediapipe-0.10.33
(sharedhands-py310) Joshuas-MacBook-Pro:artifacts josh$ python ASL_Version\ 1.0.py
Traceback (most recent call last):
  File "/Users/josh/Team4_SharedHands/Team4_SharedHands/artifacts/ASL_Version 1.0.py", line 2, in <module>
    from mediapipe import solutions
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/mediapipe/__init__.py", line 15, in <module>
    import mediapipe.tasks.python as tasks
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/mediapipe/tasks/python/__init__.py", line 17, in <module>
    from . import audio
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/mediapipe/tasks/python/audio/__init__.py", line 18, in <module>
    import mediapipe.tasks.python.audio.audio_classifier
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/mediapipe/tasks/python/audio/audio_classifier.py", line 21, in <module>
    from mediapipe.tasks.python.audio.core import audio_task_running_mode
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/mediapipe/tasks/python/audio/core/audio_task_running_mode.py", line 19, in <module>
    from mediapipe.tasks.python.core.optional_dependencies import doc_controls
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/mediapipe/tasks/python/core/optional_dependencies.py", line 20, in <module>
    from tensorflow.tools.docs import doc_controls
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/tensorflow/__init__.py", line 439, in <module>
    _ll.load_library(_plugin_dir)
  File "/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/tensorflow/python/framework/load_library.py", line 151, in load_library
    py_tf.TF_LoadLibrary(lib)
tensorflow.python.framework.errors_impl.NotFoundError: dlopen(/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/tensorflow-plugins/libmetal_plugin.dylib, 0x0006): Library not loaded: @rpath/_pywrap_tensorflow_internal.so
  Referenced from: <8B62586B-B082-3113-93AB-FD766A9960AE> /Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/tensorflow-plugins/libmetal_plugin.dylib
  Reason: tried: '/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/tensorflow-plugins/../_solib_darwin_arm64/_U@local_Uconfig_Utf_S_S_C_Upywrap_Utensorflow_Uinternal___Uexternal_Slocal_Uconfig_Utf/_pywrap_tensorflow_internal.so' (no such file), '/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/tensorflow-plugins/../_solib_darwin_arm64/_U@local_Uconfig_Utf_S_S_C_Upywrap_Utensorflow_Uinternal___Uexternal_Slocal_Uconfig_Utf/_pywrap_tensorflow_internal.so' (no such file), '/Users/josh/miniconda3/envs/sharedhands-py310/bin/../lib/_pywrap_tensorflow_internal.so' (no such file)
(sharedhands-py310) Joshuas-MacBook-Pro:artifacts josh$ python -m pip uninstall -y tensorflow-metal
Found existing installation: tensorflow-metal 1.2.0
Uninstalling tensorflow-metal-1.2.0:
  Successfully uninstalled tensorflow-metal-1.2.0
(sharedhands-py310) Joshuas-MacBook-Pro:artifacts josh$ python ASL_Version\ 1.0.py
Traceback (most recent call last):
  File "/Users/josh/Team4_SharedHands/Team4_SharedHands/artifacts/ASL_Version 1.0.py", line 2, in <module>
    from mediapipe import solutions
ImportError: cannot import name 'solutions' from 'mediapipe' (/Users/josh/miniconda3/envs/sharedhands-py310/lib/python3.10/site-packages/mediapipe/__init__.py)
(sharedhands-py310) Joshuas-MacBook-Pro:artifacts josh$ python ASL_Version\ 1.0.py
Traceback (most recent call last):
  File "/Users/josh/Team4_SharedHands/Team4_SharedHands/artifacts/ASL_Version 1.0.py", line 13, in <module>
    mp_hands = mp.solutions.hands
AttributeError: module 'mediapipe' has no attribute 'solutions'
(sharedhands-py310) Joshuas-MacBook-Pro:artifacts josh$ 
