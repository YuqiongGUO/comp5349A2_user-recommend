spark-submit \
    --master yarn \
    --deploy-mode cluster \
    --num-executors 3 \
    Assignment2_Workload_1_TFIDF.py \
    --output $1 
