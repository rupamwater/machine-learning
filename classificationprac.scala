import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession

import org.apache.log4j._

Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()

val data = spark.read.option("header", "true").option("inferSchema","true").format("csv").load("titanic.csv")

data.printSchema()

val colnames = data.columns
//val firstrow = data.take(5)(4)
val firstrow = data.head



println("\n")
println("Example data row")

for(ind <- Range(1, colnames.length)){
  println(colnames(ind))
  println(firstrow(ind))
  println("\n")
}

val logregdataall = (data.select(data("Survived").as("label"), $"Pclass", $"Name", $"Sex", $"Age",
                                                 $"SibSp", $"Parch", $"Fare", $"Embarked"))

val logregdata = logregdataall.na.drop()

import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors

val GenderIndexer = new StringIndexer().setInputCol("Sex").setOutputCol("SexIndex")
val EmbarkIndexer = new StringIndexer().setInputCol("Embarked").setOutputCol("EmbarkIndex")

val GenderEncoder = new OneHotEncoder().setInputCol("SexIndex").setOutputCol("SexVec")
val EmbarkEncoder = new OneHotEncoder().setInputCol("EmbarkIndex").setOutputCol("EmbarkVec")

val assembler = (new VectorAssembler()
                    .setInputCols(Array("Pclass", "SexVec", "Age",
                    "SibSp", "Parch", "Fare", "EmbarkVec")).setOutputCol("features"))

val Array(training, test) = logregdata.randomSplit(Array(0.9, 0.1), seed = 12345)

import org.apache.spark.ml.Pipeline

val lr = new LogisticRegression()

val pipeline = new Pipeline().setStages(Array(GenderIndexer, EmbarkIndexer, GenderEncoder, EmbarkEncoder, assembler, lr))
//val pipeline = new Pipeline().setStages(Array(GenderIndexer,GenderEncoder))



val model = pipeline.fit(training)


val results = model.transform(test)

results.printSchema
results.show(10,false)
/*
import org.apache.spark.mllib.evaluation.MulticlassMetrics

val predictionAndLabels = results.select($"prediction", $"label").as[(Double, Double)].rdd
val metrics = new MulticlassMetrics(predictionAndLabels)

println("Confusion matrix:")
println(metrics.confusionMatrix)
*/
