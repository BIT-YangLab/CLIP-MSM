PATH1="NOD_save_train_info.py"
PATH2="NOD_describe_neurons.py"

for k in $(seq 0 1); do
    for i in $(seq 1 10); do
        for j in $(seq 0 4); do
            echo "Running model$k, Subject$i of 9, ROI_$j of 5";
            python $PATH1 --md $k --cs $i --cr $j;
            python $PATH2;
        done
    done
done
