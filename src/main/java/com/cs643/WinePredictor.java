package com.cs643;

import org.apache.spark.ml.PipelineModel;
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator;
import org.apache.spark.sql.Dataset;
import org.apache.spark.sql.Row;
import org.apache.spark.sql.SparkSession;

public class WinePredictor {

    public static void main(String[] args) {
        String testPath = args.length > 0 ? args[0] : "data/ValidationDataset.csv";
        String modelIn  = args.length > 1 ? args[1] : "model";

        SparkSession spark = SparkSession.builder()
                .appName("WinePredictor")
                .getOrCreate();
        spark.sparkContext().setLogLevel("WARN");

        Dataset<Row> testData = WineTrainer.loadWineCsv(spark, testPath);

        PipelineModel model = PipelineModel.load(modelIn);
        Dataset<Row> predictions = model.transform(testData);

        MulticlassClassificationEvaluator f1Eval = new MulticlassClassificationEvaluator()
                .setLabelCol("label")
                .setPredictionCol("prediction")
                .setMetricName("f1");

        double f1 = f1Eval.evaluate(predictions);

        System.out.println("=========================================");
        System.out.println("Test F1 score: " + f1);
        System.out.println("=========================================");

        spark.stop();
    }
}