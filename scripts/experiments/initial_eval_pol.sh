source scripts/cluster.sh
for TASK_NAME in rt-polarity
do
    for SEED in 0
    do
        for BUDGET in 1500
        do
            for BASE_MODEL in t5-large
            do
                export TAGS=$TASK_NAME,$BASE_MODEL,cirrus,$BUDGET,BT10
                export TASK_NAME
                export STRATEGY=BT
                export P_STRAT=5
                export BASE_MODEL
                export BUDGET
                export TARGET=gold
                sbatch scripts/sub_$PART.sh
            done
        done
    done
done