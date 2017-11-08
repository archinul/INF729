package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions._


object Preprocessor_old {

  def udf_country = udf{(country: String, currency: String) =>
    if (country != null) country else currency
    }

  def udf_currency = udf { (currency: String) =>
    if (currency != null && currency.length == 3 && currency.matches("\\D{3}")) currency else null

  }

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()

    import spark.implicits._

    /*******************************************************************************
      *
      *       TP 2-3
      *
      *       - Charger un fichier csv dans un dataFrame
      *       - Pre-processing: cleaning, filters, feature engineering => filter, select, drop, na.fill, join, udf, distinct, count, describe, collect
      *       - Sauver le dataframe au format parquet
      *
      *       if problems with unimported modules => sbt plugins update
      *
      ********************************************************************************/

    /** 1 - CHARGEMENT DES DONNEES **/
    //a
    //val filtered_df = spark.read.option("header",true).option("nullValue", "false").csv("/home/jiaqi/Documents/Work/INF729/Spark/TP2_3/train_cleaned.csv")
    val filtered_df = spark.read
      .option("header",true)
      .option("nullValue", "")
      .option("nullValue", "false")
      .csv("/home/jiaqi/Documents/Work/INF729/Spark/TP2_3/train.csv")
    //val filtered_df=df.withColumn("disable_communication",when($"disable_communication"==="false",null).otherwise($"disable_communication"))

    //b
    println("Nombre de lignes: "+filtered_df.count())
    println("Nombre de colonnes: "+filtered_df.columns.length)

    //c
//    filtered_df.show()

    //d
//    filtered_df.printSchema()

    //e
    val df_int=filtered_df
      .withColumn("goal",$"goal".cast("int"))
      .withColumn("deadline",$"deadline".cast("int"))
      .withColumn("state_changed_at",$"state_changed_at".cast("int"))
      .withColumn("created_at",$"created_at".cast("int"))
      .withColumn("launched_at",$"launched_at".cast("int"))
      .withColumn("backers_count",$"backers_count".cast("int"))
      .withColumn("final_status",$"final_status".cast("int"))
//    df_int.printSchema()


    /** 2 - CLEANING **/

    //a
    //val goal_filtered=df_int.filter($"goal" >= 0)
    val goal_filtered=df_int
    //goal_filtered.select($"goal").distinct.show()

    //b
//    goal_filtered.select($"goal",$"backers_count",$"final_status").describe().show()

    // b
//    goal_filtered.describe().show()
//    goal_filtered.groupBy("goal").count().sort(desc("count")).show()
//    goal_filtered.groupBy("country").count().sort(desc("count")).show()
//    goal_filtered.groupBy("currency").count().orderBy(desc("count")).show()
//    goal_filtered.groupBy("final_status").count().sort(desc("count")).show()

    // c
    val df_com_drop = goal_filtered.drop($"disable_communication")
    // Let's now remove all the records with no project name
    val df_name_notNull = df_com_drop.filter($"name".isNotNull)

    // d
    // We remove the columns filled after the fact
    val df_future_drop = df_name_notNull.drop("backers_count").drop("state_changed_at")

    // e
    val df_currency_cleaned = df_future_drop
      .withColumn("country2", udf_country($"country", $"currency"))
      .withColumn("currency2", udf_currency($"currency"))

//    df_currency_cleaned.groupBy("country2").count().sort(desc("count")).show()
//    df_currency_cleaned.groupBy("currency2").count().orderBy(desc("count")).show()

    // f
    val df_status_cleaned = df_currency_cleaned.filter($"final_status".isin(List(0,1):_*))
//    df_status_cleaned.groupBy("final_status").count().sort(desc("count")).show()


    /** 3 - FEATURE ENGINEERING: Ajouter et manipuler des colonnes **/
    // a b
    val df_durations = df_currency_cleaned
      .withColumn("days_campaign",datediff(from_unixtime($"deadline"),from_unixtime($"launched_at")))
      .withColumn("hours_prepa", round(($"launched_at" - $"created_at")/3600,3))

    // c
    val df_featured = df_durations
      .drop($"created_at")
      .drop($"launched_at")
      .drop($"deadline")

    // d
    val df_add_text = df_featured.withColumn("text", concat_ws(" ", $"name", $"desc", $"keywords"))

    /** 4 - Valeurs nulles **/
    val final_cleaned_df = df_add_text
      .withColumn("goal", when($"goal".isNull, lit(-1)).otherwise($"goal"))
      .withColumn("days_campaign", when($"days_campaign".isNull, lit(-1)).otherwise($"days_campaign"))
      .withColumn("hours_prepa", when($"hours_prepa".isNull,lit(-1)).otherwise($"hours_prepa"))

    final_cleaned_df.printSchema()
//    final_cleaned_df.show(false)
    println("Nombre de lignes: "+final_cleaned_df.count())

    final_cleaned_df.coalesce(1).write.mode("overwrite").parquet("/home/jiaqi/Documents/Work/INF729/Spark/TP2_3/train_prepared.csv")
  }

}
