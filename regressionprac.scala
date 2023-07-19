import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.Vectors
import org.apache.log4j._

Logger.getLogger("org").setLevel(Level.ERROR)

def main():Unit = {

  val spark = SparkSession.builder().appName("LinearRegressionExample")getOrCreate()
  val path = "Clean_USA_Housing.csv"
  val data = spark.read.option("header","true").option("inferSchema", "true").format("csv").load(path)
  data.printSchema()



  val df = data.select(data("Price").as("label"),  $"Avg Area Income", $"Avg Area House Age", $"Avg Area Number of Rooms",
  $"Avg Area Number of Bedrooms", $"Area Population")

  val assembler = (new VectorAssembler().setInputCols(Array("Avg Area Income", "Avg Area House Age", "Avg Area Number of Rooms",
  "Avg Area Number of Bedrooms", "Area Population"))).setOutputCol("features")

  val output = assembler.transform(df).select($"label", $"features")
  output.show()

  val lr = new LinearRegression().setMaxIter(100).setRegParam(0.3).setElasticNetParam(0.8)
  val lrModel = lr.fit(output)

  val trainingSummary = lrModel.summary
  val r2 = trainingSummary.r2

  println(s"R2 : $r2")
  trainingSummary.predictions.show
  //spark.stop()

}
main()
