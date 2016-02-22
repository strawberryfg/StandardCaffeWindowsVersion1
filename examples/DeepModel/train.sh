LOG="logs/log.txt.`date +'%Y-%m-%d_%H-%M-%S'`"
exec &> >(tee -a "$LOG")
echo Logging output to "$LOG"
python ../../python/draw_net.py DeepModel.prototxt net.svg --rankdir BT
./../../build/tools/caffe train --solver solver.prototxt
python Evaluation.py
