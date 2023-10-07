#source scripts/cluster.sh
export TRAIN_SAMPLES=10000
export RETRAIN_FREQ=100
export TARGET=llm
export DATA_PATH=/work/dc007/dc007/cs-rami1/data
export PART=cirrus
export BASE_MODEL=t5-base

for SOFT_LABELS in 1 0
do
    for TASK_NAME in isear rt-polarity openbook mmlu-ss ag_news fever trec
    do 
        for STRATEGY in b1 BT EN CS
        do
            export TASK_NAME
            export STRATEGY
            export SOFT_LABELS
            export TAGS=$BASE_MODEL,$TASK_NAME,$STRATEGY,$TARGET,cirrus,LAST

            if [ $STRATEGY == "b1" ]
            then
                sbatch --export=ALL scripts/sub_$PART.sh
            fi
            if [ $STRATEGY == "BT" ]
            then 
                for P_STRAT in 10 5
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
                for P_STRAT in 0.5 1
                do
                    export P_STRAT
                    sbatch --export=ALL scripts/sub_$PART.sh
                done
            fi 
            if [ $STRATEGY == "CS" ]
            then
                for P_STRAT in 0.9 0.8 0.7
                do
                    export P_STRAT
                    sbatch --export=ALL scripts/sub_$PART.sh
                done
            fi
        done
    done  
done



