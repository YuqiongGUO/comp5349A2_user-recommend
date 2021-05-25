#!/usr/bin/env python
# coding: utf-8

# ### Assignment 2-Workload 2

# In[ ]:


from pyspark.sql import SparkSession
from pyspark.sql.functions import *
import pyspark.sql.functions as F
from pyspark.ml.recommendation import *


# In[ ]:


spark = SparkSession     .builder     .appName("Assignment2_workload2")     .getOrCreate()
sc =SparkContext.getOrCreate()


# ### Data preparation

# In[ ]:


#Loading rawdata
rawdata = spark.read.option("multiline","true").json("tweets.json")
# rawdata.show(1)
usermention = rawdata.select("user_id","user_mentions")
# usermention.show(10, truncate=False)

#convert user id to fit ALS(longint cannot be accepted by ALS)
update_user_id = rawdata.select("user_id")                  .rdd.map(lambda x: x[0])                  .distinct().collect()

new_user_id = dict()

for item in update_user_id:
    new_user_id[item] = len(new_user_id)

#convert user_mentions to fit ALS(longint cannot be accepted by ALS)
update_user_mentions =  rawdata.select("user_mentions")                                .where("user_mentions is not null")                                .rdd.flatMap(lambda x: [y[0] for y in x[0]])                                .distinct().collect()

new_user_mentions = dict()

for item in update_user_mentions:
    new_user_mentions[item] = len(new_user_mentions)

#broadcast new_user_id and new_user_mentions to all nodes.
sc.broadcast(new_user_id)
sc.broadcast(new_user_mentions)


# ### User Recommendation

# In[ ]:


#convert user_id and user_mention to pair and count
def user_mention_cov(x):
    user_id = x[0]
    user_mention_list = list()
    for user_mentions in x[1]:
        user_mention_list.append(tuple([user_id,new_user_mentions[user_mentions[0]],1]))
    return user_mention_list

#archieve the RDD with user_id, user_mention and mention_count, which can be fed to ALS.
usermention_rdd = usermention.where("user_mentions is not null")                              .rdd.map(lambda x: (new_user_id[x[0]],x[1]))                              .flatMap(user_mention_cov)
usermention_rdd_count = usermention_rdd.map(lambda x: ((x[0],x[1]),x[2]))                                        .reduceByKey(lambda x1,x2: x1+x2)                                        .map(lambda x: (x[0][0],x[0][1],x[1]))                                        .toDF(["user_id","user_mentions","mention_count"])

#Collaborative Filtering
als =ALS(userCol="user_id", itemCol="user_mentions", ratingCol = "mention_count", coldStartStrategy="drop")
model = als.fit(usermention_rdd_count)

#get the top 5 recoomondation
userRecs = model.recommendForAllUsers(5).collect()

print("Top 5 mention user for each user are")

for row in userRecs:
    print(update_user_id[row[0]],":", end = " ")
    for b in row[1]:
        print(update_user_mentions[b[0]], end = " ")
    print()


# In[ ]:




