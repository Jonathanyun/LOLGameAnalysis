# Databricks notebook source
from pyspark.ml import Pipeline
from pyspark.sql.types import StructType, StructField, LongType
from pyspark.ml.classification import LogisticRegression, DecisionTreeClassifier
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import StringIndexer, Bucketizer
from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

schema = StructType( \
                   [StructField("ID", LongType(), True),\
                    StructField("gameID", LongType(), True), \
                    StructField("firstBlood", LongType(), True),\
                    StructField("firstTower", LongType(), True),\
                    StructField("firstInhibitor", LongType(), True),\
                    StructField("firstBaron", LongType(), True),\
                    StructField("firstDragon", LongType(), True),\
                    StructField("firstRiftHerald", LongType(), True),\
                    StructField("winner", LongType(), True)
                   ])

trainingDF = spark.read.format('csv').option('header', True).schema(schema).load("dbfs:/FileStore/tables/trainingGames.csv")
# trainingDF.show()

testingDF = spark.read.format('csv').option('header', True).schema(schema).load("dbfs:/FileStore/tables/testingGames.csv")
# testingDF.show()

lr = LogisticRegression(maxIter = 10, regParam = 0.01)

# trainingDF.show()

predIndexer = StringIndexer(inputCol = 'winner', outputCol = 'label')

baronSplits = [-float("inf"), 0, 1, 2, float("inf")]
baronBucketizer = Bucketizer(splits = baronSplits, inputCol = 'firstBaron', outputCol = "baronBucket")

heraldSplits = [-float("inf"), 0, 1, 2, float("inf")]
heraldBucketizer = Bucketizer(splits = heraldSplits, inputCol = 'firstRiftHerald', outputCol = "heraldBucket")

winIndexer = StringIndexer(inputCol = 'winner', outputCol = 'label')
                
bloodIndexer = StringIndexer(inputCol = 'firstBlood', outputCol = 'bloodIndex')

towerIndexer = StringIndexer(inputCol = 'firstTower', outputCol = 'towerIndex')

inhibIndexer = StringIndexer(inputCol = 'firstInhibitor', outputCol = 'inhibIndex')

dragIndexer = StringIndexer(inputCol = 'firstDragon', outputCol = 'dragIndex')

trainingData = baronBucketizer.transform(trainingDF)
trainingData = heraldBucketizer.transform(trainingData)
winModel = winIndexer.fit(trainingData)
trainingData = winModel.transform(trainingData)
bloodModel = bloodIndexer.fit(trainingData)
trainingData = bloodModel.transform(trainingData)
towerModel = towerIndexer.fit(trainingData)
trainingData = towerModel.transform(trainingData)
inhibModel = inhibIndexer.fit(trainingData)
trainingData = inhibModel.transform(trainingData)
dragModel = dragIndexer.fit(trainingData)
trainingData = dragModel.transform(trainingData)
# trainingData.show()

testData = dragModel.transform(inhibModel.transform(towerModel.transform(bloodModel.transform(heraldBucketizer.transform(baronBucketizer.transform(testingDF))))))

vecAssem = VectorAssembler(inputCols = ['baronBucket', 'heraldBucket', 'bloodIndex', 'towerIndex', 'inhibIndex', 'dragIndex'], outputCol = 'features')

myStages = [baronBucketizer, heraldBucketizer, winIndexer, bloodIndexer, towerIndexer, inhibIndexer, dragIndexer, vecAssem, lr]

p = Pipeline(stages = myStages)

pModel = p.fit(trainingDF)

pred = pModel.transform(testingDF)
pred.select('id', 'probability', 'prediction').show()

evaluator = MulticlassClassificationEvaluator(labelCol = 'winner', predictionCol = 'prediction', metricName = 'accuracy')
accuracy = evaluator.evaluate(pred)
print("Train Accuracy = %g " % (accuracy))

# COMMAND ----------

# dt = DecisionTreeClassifier(labelCol = 'label', featuresCol = 'features')
# myStages = [baronBucketizer, heraldBucketizer, winIndexer, bloodIndexer, towerIndexer, inhibIndexer, dragIndexer, vecAssem, dt]
# p = Pipeline(stages = myStages)

# pModel = p.fit(trainingDF)

# pred = pModel.transform(testingDF)
# pred.select('id', 'probability', 'prediction').show()

# evaluator = MulticlassClassificationEvaluator(labelCol = 'winner', predictionCol = 'prediction', metricName = 'accuracy')
# accuracy = evaluator.evaluate(pred)
# print("Train Accuracy = %g " % (accuracy))


# COMMAND ----------

testData = testingDF.repartition(10)

# dbutils.fs.rm("FileStore/tables/testGames/", True)

# testData.write.format("csv").option("header", True).save("FileStore/tables/testGame/")

sourceStream = spark.readStream.format("csv").option("header", True).schema(schema).option("ignoreLeadingWhiteSpace", True).option("mode", "dropMalformed").option("maxFilesPerTrigger", 1).load('dbfs:/FileStore/tables/testGame/').withColumnRenamed('winner', 'label')

streamingGames = pModel.transform(sourceStream).select('label', 'probability', 'prediction')
display(streamingGames)

# COMMAND ----------


