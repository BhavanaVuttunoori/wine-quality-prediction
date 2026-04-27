package com.cs643;

import org.apache.spark.ml.Pipeline;
import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.PipelineStage;
import org.apache.spark.ml.classification.RandomForestClassifier;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.ml.feature.VectorAssembler;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;
import org.apache.spark.sql.functions;

public class WineTrainer {

    public static void main(String[] args) {
        String trainPath = args.length > 0 ? args[0] : "data/TrainingDataset.csv";
        String valPath   = args.length > 1 ? args[1] : "data/ValidationDataset.csv";
        String modelOut  = args.length > 2 ? args[2] : "model";

        SparkSession spark = SparkSession.builder()
                .appName("WineTrainer")
                .getOrCreate();
        spark.sparkContext().setLogLevel("WARN");

        Dataset<Row> trainRaw = loadWineCsv(spark, trainPath);
        Dataset<Row> valRaw   = loadWineCsv(spark, valPath);

        String[] featureCols = {
            "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
            "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide",
            "density", "pH", "sulphates", "alcohol"
        };

        VectorAssembler assembler = new VectorAssembler()
                .setInputCols(featureCols)
                .setOutputCol("features");

        RandomForestClassifier rf = new RandomForestClassifier()
                .setLabelCol("label")
                .setFeaturesCol("features")
                .setNumTrees(50)
                .setMaxDepth(10)
                .setSeed(42);

        Pipeline pipeline = new Pipeline()
                .setStages(new PipelineStage[]{ assembler, rf });

        PipelineModel model = pipeline.fit(trainRaw);

        Dataset<Row> predictions = model.transform(valRaw);

        MulticlassClassificationEvaluator f1Eval = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("f1");

        double f1 = f1Eval.evaluate(predictions);
        System.out.println("=========================================");
        System.out.println("Validation F1 score: " + f1);
        System.out.println("=========================================");

        try {
            model.write().overwrite().save(modelOut);
            System.out.println("Model saved to: " + modelOut);
        } catch (Exception e) {
            e.printStackTrace();
        }

        spark.stop();
    }

    public static Dataset<Row> loadWineCsv(SparkSession spark, String path) {
        Dataset<Row> raw = spark.read()
                .option("header", "true")
                .option("sep", ";")
                .option("quote", "\"")
                .option("inferSchema", "true")
                .csv(path);

        String[] cleanNames = {
            "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
            "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide",
            "density", "pH", "sulphates", "alcohol", "label"
        };
        String[] originalNames = raw.columns();
        for (int i = 0; i < originalNames.length && i < cleanNames.length; i++) {
            raw = raw.withColumnRenamed(originalNames[i], cleanNames[i]);
        }

        raw = raw.withColumn("label", functions.col("label").cast("double"));
        for (int i = 0; i < cleanNames.length - 1; i++) {
            raw = raw.withColumn(cleanNames[i], functions.col(cleanNames[i]).cast("double"));
        }
        return raw;
    }
}