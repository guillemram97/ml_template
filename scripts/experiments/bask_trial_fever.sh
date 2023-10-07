source scripts/cluster.sh
export RETRAIN_FREQ=1000
export SOFT_LABELS=0
export TARGET=gold
export BUDGET=5000

for N_INIT in 0
do
    for SEED in 0
    do
        for TASK_NAME in fever_explore
        do 
            export N_INIT
            if [ $TASK_NAME == "isear" ]
            then
                export N_INIT=100
            fi
            if [ $TASK_NAME == "rt-polarity" ]
            then
                export N_INIT=100
            fi
            for STRATEGY in b1
            do
                export TASK_NAME
                export N_INIT
                export STRATEGY
                export SEED
                export CHECKPOINT=-1
                export TAGS=FEVER_exp

                if [ $STRATEGY == "b1" ]
                then
                    export P_STRAT=0
                    sbatch --export=ALL scripts/sub_$PART.sh
                fi
                if [ $STRATEGY == "b2" ]
                then
                    export P_STRAT=0
                    sbatch --export=ALL scripts/sub_$PART.sh
                fi
                if [ $STRATEGY == "BT" ]
                then 
                    for P_STRAT in 5
                    do
                        export P_STRAT
                        sbatch --export=ALL scripts/sub_$PART.sh
                    done
                fi
                if [ $STRATEGY == "MV" ]
                then
                    export P_STRAT=3
                    sbatch --export=ALL scripts/sub_$PART.sh
                fi
                if [ $STRATEGY == "EN" ]
                then
                    for P_STRAT in 0.5
                    do
                        export P_STRAT
                        sbatch --export=ALL scripts/sub_$PART.sh
                    done
                fi 
                if [ $STRATEGY == "CS" ]
                then
                    for P_STRAT in 0.95
                    do
                        export P_STRAT
                        sbatch --export=ALL scripts/sub_$PART.sh
                    done
                fi
            done
        done
    done
done
