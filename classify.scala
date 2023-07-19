import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.sql.SparkSession
import org.apache.log4j._
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.{VectorAssembler, StringIndexer, VectorIndexer, OneHotEncoder}
import org.apache.spark.ml.linalg.Vectors
import org.apache.spark.mllib.evaluation.MulticlassMetrics


Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()

val data = spark.read.option("header", "true").format("csv").load("table.csv")
val data_test = spark.read.option("header", "true").format("csv").load("table_test.csv")


data.printSchema()

val dataenhanced = data
.withColumn("Client_Name_Class",
             when(col("CORP_PROFILE_NAME").contains("ABSA"), lit("BANK_NAME"))
            .when(col("CORP_PROFILE_NAME").contains("ABK") or col("CORP_PROFILE_NAME").contains("ABCAP"), lit("RELATED_NAME"))
           .otherwise(lit("CLIENT_NAME")))
.withColumn("Client_Email_Class",
                 when(col("EMAIL").contains("absa.africa"), lit("COLLEAGUE_MAIL"))
                .when(col("EMAIL").contains("absa"), lit("RELATED_MAIL"))
                .otherwise(lit("CLIENT_MAIL")))
.withColumn("Userdigit", when(col("RealUserType").contains("ClientUser"), 1)
                                          .otherwise(0))

val data_test2 = data_test.withColumn("Client_Name_Class", when(col("CORP_PROFILE_NAME").contains("ABSA"), lit("BANK_NAME")).otherwise(
                                             when(col("CORP_PROFILE_NAME").contains("ABK") or col("CORP_PROFILE_NAME").contains("ABCAP"),
                                             lit("RELATED_NAME")).otherwise(lit("CLIENT_NAME"))))

val data_test3 = data_test2.withColumn("Client_Email_Class", when(col("EMAIL").contains("absa.africa"), lit("COLLEAGUE_MAIL"))
                                             .when(col("EMAIL").contains("absa"), lit("RELATED_MAIL"))
                                             .otherwise(lit("CLIENT_MAIL")))

val data_test4 = data3.withColumn("Userdigit", when(col("RealUserType").contains("ClientUser"), 1)
                                            .otherwise(0))

data_test4.show(20,false)

val data5 = data4.select($"Userdigit".as("label"), $"CORP_PROFILE_NAME", $"CorpProfileID",
                              $"EMAIL", $"USER PROFILE_ID", $"Client_Name_Class",
                              $"JURISDICTION", $"UserStatus", $"User_UserType",
                              $"Client_Email_Class")

val data_test5 = data_test4.select($"Userdigit".as("label"), $"CORP_PROFILE_NAME",$"CorpProfileID",
                                $"EMAIL", $"USER PROFILE_ID", $"Client_Name_Class",
                               $"JURISDICTION", $"UserStatus", $"User_UserType",
                               $"Client_Email_Class")



val ClientNameIndexer = new StringIndexer().setInputCol("Client_Name_Class").setOutputCol("Client_Name_Index")
val ClientEmailIndexer = new StringIndexer().setInputCol("Client_Email_Class").setOutputCol("Client_Email_Index")
val CountryIndexer = new StringIndexer().setInputCol("JURISDICTION").setOutputCol("JURISDICTION_Index")
val ClientNameEncoder = new OneHotEncoder().setInputCol("Client_Name_Index").setOutputCol("Client_Name_Vec")
val ClientEmailEncoder = new OneHotEncoder().setInputCol("Client_Email_Index").setOutputCol("Client_Email_Vec")
val CountryEncoder = new OneHotEncoder().setInputCol("JURISDICTION_Index").setOutputCol("JURISDICTION_Vec")

val assembler = (new VectorAssembler()
                    .setInputCols(Array("JURISDICTION_Vec","Client_Name_Vec",
                    "Client_Email_Vec"))).setOutputCol("features")

//val Array(training, test) = data3.randomSplit(Array(1.00, 0.25), seed = 12345)

val training = data_test5
val TrainClientNameIndexCount = training.select($"Client_Name_Class").distinct.count


val test = data5
val TestClientNameIndexCount = test.select($"Client_Name_Class").distinct.count

if(TrainClientNameIndexCount != TestClientNameIndexCount ){
  println("Error")

}

val lr = new LogisticRegression()


val pipeline = new Pipeline().setStages(Array(ClientNameIndexer,
                                                ClientEmailIndexer,
                                                CountryIndexer,
                                                ClientNameEncoder,
                                                ClientEmailEncoder,
                                                CountryEncoder,
                                                assembler,
                                                lr))


val model = pipeline.fit(training)
val results = model.transform(test)

results.show(6, false)
results.printSchema()


val predictionAndLabels = results.select($"prediction", $"label").as[(Double, Double)].rdd


val metrics = new MulticlassMetrics(predictionAndLabels)

println("Confusion matrix:")
println(metrics.confusionMatrix)
