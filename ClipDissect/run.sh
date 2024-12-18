
PATH1="save_train_info.py"
PATH2="describe_neurons.py"

for k in $(seq 0 4); do
    for i in $(seq 1 9); do
        for j in $(seq 0 5); do
            echo "Running model$k, Subject$i of 8, ROI_$j of 6";
            python $PATH1 --md $k --cs $i --cr $j;
            python $PATH2;
        done
    done
done
