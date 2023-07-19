import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.regression.LinearRegression

import org.apache.log4j._

Logger.getLogger("org").setLevel(Level.ERROR)

val spark = SparkSession.builder().getOrCreate()

val df = spark.read.option("header", "true").option("inferschema", "true").csv("Affordable_housing_2006_2008.csv")
val df2 = spark.read.csv("Affordable_housing_2006_2008.csv")
df.describe().show()


val df3 = df.withColumn("Total", df("Tenant Rental Assistance") + df("Government Assisted"))

df3.printSchema()

for (row <- df3.head(5)){
  println(row)
}
df3.groupBy("Town").mean().orderBy("Town").show()
