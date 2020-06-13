import autogluon as ag
from autogluon import ObjectDetection as task
from console_logging.console import Console
console = Console()


console.log("Baixando Dataset...")
root = './'
filename_zip = ag.download('https://autogluon.s3.amazonaws.com/datasets/tiny_motorbike.zip',
                        path=root)
filename = ag.unzip(filename_zip, root=root)

console.log("Criando TASK TRAIN ")
import os
data_root = os.path.join(root, filename)
dataset_train = task.Dataset(data_root, classes=('motorbike',))

console.info("TRAINING DATA MODEL...")
time_limits = 5*60*60  # 5 hours
epochs = 30
detector = task.fit(dataset_train,
                    num_trials=2,
                    epochs=epochs,
                    lr=ag.Categorical(5e-4, 1e-4),
                    ngpus_per_trial=1,
                    time_limits=time_limits)
console.success("TRAINING DONE !")
console.log("START TEST MODEL ")
dataset_test = task.Dataset(data_root, index_file_name='test', classes=('motorbike',))

test_map = detector.evaluate(dataset_test)
console.log("mAP on test dataset: {}".format(test_map[1][1]))

console.success("SAVE MODEL")
savefile = 'model.pkl'
detector.save(savefile)

from autogluon import Detector
new_detector = Detector.load(savefile)
