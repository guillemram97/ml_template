source scripts/cluster.sh
export TRAIN_SAMPLES=10000
export RETRAIN_FREQ=1000
export TARGET=gold

for ONLY_IMPROVE in 0
do
    for BASE_MODEL in t5-large t5-base
    do
        for SEED in 0
        do
            for BUDGET in 5000
            do
                for TASK_NAME in qa_wikidata
                do 
                    for STRATEGY in b1
                    do
                        export ONLY_IMPROVE
                        export SEED
                        export BASE_MODEL
                        export TARGET
                        export BUDGET
                        export TASK_NAME
                        export STRATEGY
                        export RETRAIN_FREQ
                        export TAGS=$BASE_MODEL,$TASK_NAME,$STRATEGY,$BUDGET,$TARGET,baskerville,INITIAL_RUN
                        sbatch scripts/sub_$PART.sh
                    done
                done
            
            done
        done
    done
done

