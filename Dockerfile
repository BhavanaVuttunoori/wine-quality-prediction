FROM bitnamilegacy/spark:3.5.0
USER root
WORKDIR /app
COPY target/wine-quality-1.0.jar /app/wine-quality.jar
COPY model /app/model
ENTRYPOINT ["spark-submit", "--class", "com.cs643.WinePredictor", "--master", "local[*]", "/app/wine-quality.jar"]
CMD ["/app/data/test.csv", "/app/model"]
