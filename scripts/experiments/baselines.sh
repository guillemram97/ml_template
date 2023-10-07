source scripts/cluster.sh
for N_INIT in 100 500
do
    for BUDGET in 300 400 500 600 700 800 900 1000 1500 2000 2500 3000 3500
    do
        for TASK_NAME in isear openbook rt-polarity 
        do
            for STRATEGY in MV
            do
                for SEED in 1 2 0
                do
                    export SEED
                    export N_VOTES=2
                    export STRATEGY
                    export PERCENTILE=10
                    export BUDGET
                    export RETRAIN_FREQ=100
                    export TASK_NAME
                    export N_INIT
                    export TRAIN_SAMPLES=10000
                    export TAGS=$TASK_NAME,$STRATEGY,$BUDGET,$SEED,$N_INIT
                    sbatch scripts/sub_$PART.sh
                done
            done
        done
    done
done