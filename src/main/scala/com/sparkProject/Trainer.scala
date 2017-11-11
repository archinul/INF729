//Jiaqi ZHANG
package com.sparkProject

import org.apache.spark.SparkConf
import org.apache.spark.sql.{SaveMode, SparkSession, functions}
import org.apache.spark.ml.feature._


object Trainer {

  def main(args: Array[String]): Unit = {

    val conf = new SparkConf().setAll(Map(
      "spark.scheduler.mode" -> "FIFO",
      "spark.speculation" -> "false",
      "spark.reducer.maxSizeInFlight" -> "48m",
      "spark.serializer" -> "org.apache.spark.serializer.KryoSerializer",
      "spark.kryoserializer.buffer.max" -> "1g",
      "spark.shuffle.file.buffer" -> "32k",
      "spark.default.parallelism" -> "12",
      "spark.sql.shuffle.partitions" -> "12",
      "spark.driver.maxResultSize" -> "2g"
    ))

    val spark = SparkSession
      .builder
      .config(conf)
      .appName("TP_spark")
      .getOrCreate()



    /** *****************************************************************************
      *
      * TP 4-5
      *
      *       - lire le fichier sauvegarder précédemment
      *       - construire les Stages du pipeline, puis les assembler
      *       - trouver les meilleurs hyperparamètres pour l'entraînement du pipeline avec une grid-search
      *       - Sauvegarder le pipeline entraîné
      *
      * if problems with unimported modules => sbt plugins update
      *
      * *******************************************************************************/

    /** 1) CHARGER LE DATASET **/
    val df = spark.read
      .option("header", true)
      .option("nullValue", "")
      .option("nullValue", "false")
//      .parquet("/home/jiaqi/Documents/Work/INF729/Spark/TP2_3/train_prepared.csv")
      .parquet("/tmp/train_prepared.csv")

    /** 2) UTILISER LES DONNEES TEXTUELLES **/
    // a) Tokenizer
    val tokenizer = new RegexTokenizer()
      .setPattern("\\W+")
      .setGaps(true)
      .setInputCol("text")
      .setOutputCol("tokens")

    val tokenized = tokenizer.transform(df)


    // b) Stop words remover
    val remover = new StopWordsRemover()
      .setInputCol("tokens")
      .setOutputCol("words")

    val removed = remover.transform(tokenized)


    // c) TF
    val cvModel: CountVectorizerModel = new CountVectorizer()
      .setInputCol("words")
      .setOutputCol("rawfeatures")
      .fit(removed)

    val modelized = cvModel.transform(removed)


    // d) IDF
    val idfModel = new IDF()
      .setInputCol("rawfeatures")
      .setOutputCol("tfidf")
      .fit(modelized)

    val rescaledData = idfModel.transform(modelized)


    /** 3) CONVERTIR LES CATEGORIES EN DONNEES NUMERIQUES **/
    // e) Country2
    val indexerCountry = new StringIndexer()
      .setInputCol("country2")
      .setOutputCol("country_indexed")
      .fit(rescaledData)

    val dfCountry = indexerCountry.transform(rescaledData)

    // f) Currency2
    val indexerCurrency = new StringIndexer()
      .setInputCol("currency2")
      .setOutputCol("currency_indexed")
      .fit(rescaledData)

    val dfCurrency = indexerCurrency.transform(rescaledData)


    /** 4) METTRE LES DONNEES SOUS UNE FORME UNTILISABLE PAR SPARK.ML **/
    // g) Vector assembler
    val vectAssembler = new VectorAssembler()
      .setInputCols(Array("tfidf", "days_campaign", "hours_prepa", "goal", "country_indexed", "currency_indexed"))
      .setOutputCol("features")


    // h) Classification model
    import org.apache.spark.ml.classification.LogisticRegression
    import org.apache.spark.ml.classification.LogisticRegressionModel

    val lr = new LogisticRegression()
      .setElasticNetParam(0.0)
      .setFitIntercept(true)
      .setFeaturesCol("features")
      .setLabelCol("final_status")
      .setStandardization(true)
      .setPredictionCol("predictions")
      .setRawPredictionCol("raw_predictions")
      .setThreshold(0.3)
      .setTol(1.0e-6)
      .setMaxIter(300)


    // i) Pipeline
    import org.apache.spark.ml.{Pipeline, PipelineModel}
    val pipeline = new Pipeline()
      .setStages(Array(tokenizer, remover, cvModel, idfModel, indexerCountry, indexerCurrency, vectAssembler, lr))


    /** 5) ENTRAINEMENT ET TUNING DU MODELE **/
    // j) Split data
//    Split dataset randomly into Training and Test sets. Set seed for reproducibility
    val Array(trainingData, testData) = df.randomSplit(Array(0.9, 0.1),seed = 100)

    // k) Grid Search
    import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
    import org.apache.spark.ml.tuning.{ParamGridBuilder, TrainValidationSplit}

    val evaluatorF1 = new MulticlassClassificationEvaluator()
      .setLabelCol("final_status")
      .setPredictionCol("predictions")
      .setMetricName("f1")

    val paramGrid = new ParamGridBuilder()
      .addGrid(lr.regParam, Array(1e-8, 1e-6, 1e-4, 1e-2))
      .addGrid(cvModel.minDF, Array(55.0,75.0,95.0))
      .build()

    val trainValidationSplit = new TrainValidationSplit()
      .setEstimator(pipeline)
      .setEvaluator(evaluatorF1)
      .setEstimatorParamMaps(paramGrid)
      // 70% of the data will be used for training and the remaining 30% for validation.
      .setTrainRatio(0.7)
      .setSeed(100)

    // l) Run grid search and show F1-score.
    val model = trainValidationSplit.fit(trainingData)

    val df_WithPredictions = model.transform(testData)

    // F1-score
    val f1 = evaluatorF1.evaluate(df_WithPredictions)
    println("Model F1: " + f1)

    // Displaying the parameters found via grid search
    val bestPipelineModel = model.bestModel.asInstanceOf[PipelineModel]
    val stages = bestPipelineModel.stages
    println("Best parameters found on grid search :")

    val hashingStage = stages(2).asInstanceOf[CountVectorizerModel]
    println("\tminDF = " + hashingStage.getMinDF)

    val lrStage = stages(stages.length - 1).asInstanceOf[LogisticRegressionModel]
    println("\tregParam = " + lrStage.getRegParam)


    // m) Show predictions
    df_WithPredictions.groupBy("final_status", "predictions").count.show()


    // Save the model for future use
//    model.write.overwrite().save("/home/jiaqi/Documents/Work/INF729/Spark/TP4_5/myLogisticR")
    model.write.overwrite().save("/tmp/myLogisticR")
  }
}
