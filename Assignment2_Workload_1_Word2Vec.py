#!/usr/bin/env python
# coding: utf-8

# ### Assignment 2-Workload 1

# In[ ]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import pyspark.sql.functions as F


# In[ ]:


spark = SparkSession     .builder     .appName("Assignment2_workload1_Word2Vec")     .getOrCreate()


# ### Data preparation

# In[ ]:


rawdata = spark.read.option("multiline","true").json("tweets.json")
usertweet = rawdata.select("user_id","replyto_id","retweet_id")

#combine reply and retweet
#groupby user and collect reply&retweet id named as DP
user_rp_rt = usertweet.withColumn("rp_rt", concat_ws(',', usertweet['replyto_id'],usertweet['retweet_id']))              .groupBy("user_id")              .agg((collect_list("rp_rt")))              .withColumnRenamed("collect_list(rp_rt)","document_presentation")
user_rp_rt = user_rp_rt.withColumn("document_presentation", concat_ws(',', user_rp_rt['document_presentation']))
user_rp_rt = user_rp_rt.withColumn("document_presentation",F.split(user_rp_rt.document_presentation, ","))
# user_rp_rt.show(20)  #truncate=False
# user_rp_rt.printSchema()


# ### Feature extractors:Word2Vec

# In[ ]:


#implement Word2Vec
from pyspark.ml.feature import Word2Vec

word2Vec = Word2Vec(vectorSize=5, minCount=0, inputCol="document_presentation",outputCol="word2vec")
model = word2Vec.fit(user_rp_rt)

user_rp_rt_word2Vec = model.transform(user_rp_rt)
# user_rp_rt_word2Vec.select("user_id","word2vec").show(truncate=False)


# In[ ]:


#find the vector for selected test_id
test_id = 157101980 
test_row = user_rp_rt_word2Vec.filter(user_rp_rt_word2Vec["user_id"] == test_id).collect()
test_vector = test_row[0].asDict()["word2vec"]

#find the vectors for all the other users except selected test_id
compare_vector_rdd = user_rp_rt_word2Vec.filter(user_rp_rt_word2Vec["user_id"] != test_id)                                         .rdd.map(lambda x: (x[0], x[2]))

#cosine similarity function
def cosine_similarity(a,b):
    similarity = a.dot(b)/(a.norm(2)*b.norm(2))
    return similarity

#calculate cosine similarity for all other users compared with selected user
sim_user = compare_vector_rdd.map(lambda x : (x[0], cosine_similarity(test_vector, x[1])))

#find the top 5 user
sim_user_top5 = sim_user.sortBy(lambda x: x[1],ascending=False).take(5)

print("Top 5 similar interest user with", test_id, "is")

for items in sim_user_top5:
    print(items[0])


# In[ ]:




