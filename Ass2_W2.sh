spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --num-executors 3 \
    Assignment2_Workload_2.py \
    --output $1 
