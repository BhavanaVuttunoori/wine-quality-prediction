# CS 643 Programming Assignment 2 - Wine Quality Prediction

This project trains a wine quality prediction model using Apache Spark MLlib in Java. Training runs in parallel across 4 EC2 instances. Prediction runs on a single EC2 instance and is also packaged as a Docker container for portable deployment.

Author: Bhavana Vuttunoori
Course: CS 643, Cloud Computing


## Links

GitHub repository: https://github.com/BhavanaVuttunoori/wine-quality-prediction

Docker Hub image: https://hub.docker.com/r/bhavanavuttunoori/wine-predictor


## What this project does

The training application reads TrainingDataset.csv, fits a Random Forest classifier on 11 wine features (acidity, pH, alcohol, etc.), evaluates the model on ValidationDataset.csv, and saves the trained pipeline to disk.

The prediction application loads the saved model and computes the F1 score on a test dataset. It runs either directly with spark-submit on an EC2 instance, or inside a Docker container.


## Model details

The model is a Random Forest classifier from Spark MLlib, configured with 50 trees and a maximum depth of 10. Features are combined with a VectorAssembler before training. Evaluation uses MulticlassClassificationEvaluator with the F1 metric.

Validation F1 score on the provided validation set: 0.5549


## Repository contents

```
wine-quality-prediction/
  pom.xml
  Dockerfile
  README.md
  .gitignore
  data/
    TrainingDataset.csv
    ValidationDataset.csv
  src/main/java/com/cs643/
    WineTrainer.java
    WinePredictor.java
```


## Building the project

Requires Java 11 and Maven 3.x.

From the project root:

```
mvn clean package
```

This produces a fat JAR at target/wine-quality-1.0.jar that includes all Spark dependencies.


## Part 1 - Parallel training on 4 EC2 instances

### Create AWS resources

Launch 4 EC2 instances with the following configuration:

- AMI: Ubuntu Server 22.04 LTS
- Instance type: t2.medium
- Storage: 20 GB
- Key pair: wine-key (download wine-key.pem)
- Security group: spark-sg

The security group needs the following inbound rules:

- SSH (port 22) from your IP
- All TCP within the same security group (so Spark workers and master can communicate)
- TCP 8080 from your IP, for the Spark Web UI

### Install Java and Spark on each instance

SSH into every instance and run:

```
sudo apt update
sudo apt install -y openjdk-11-jdk wget
wget https://archive.apache.org/dist/spark/spark-3.5.0/spark-3.5.0-bin-hadoop3.tgz
tar -xzf spark-3.5.0-bin-hadoop3.tgz
mv spark-3.5.0-bin-hadoop3 spark
echo 'export SPARK_HOME=$HOME/spark' >> ~/.bashrc
echo 'export PATH=$PATH:$SPARK_HOME/bin:$SPARK_HOME/sbin' >> ~/.bashrc
echo 'export JAVA_HOME=/usr/lib/jvm/java-11-openjdk-amd64' >> ~/.bashrc
source ~/.bashrc
```

### Start the Spark cluster

Pick one instance as the master. The other three are workers.

On the master:

```
$SPARK_HOME/sbin/start-master.sh
```

On each of the three workers:

```
$SPARK_HOME/sbin/start-worker.sh spark://<MASTER_PRIVATE_IP>:7077
```

Optionally start a worker on the master too, so the cluster has 4 workers.

Verify the cluster by visiting http://<MASTER_PUBLIC_IP>:8080 in a browser. All workers should be listed.

### Copy the JAR and datasets

From your local machine:

```
scp -i wine-key.pem target/wine-quality-1.0.jar ubuntu@<MASTER_IP>:~/
scp -i wine-key.pem data/TrainingDataset.csv ubuntu@<MASTER_IP>:~/
scp -i wine-key.pem data/ValidationDataset.csv ubuntu@<MASTER_IP>:~/
```

The CSVs also need to exist on every worker, so repeat the scp commands for each worker IP.

### Run the training job

SSH into the master and submit the job:

```
$SPARK_HOME/bin/spark-submit \
  --class com.cs643.WineTrainer \
  --master spark://<MASTER_PRIVATE_IP>:7077 \
  --deploy-mode client \
  ~/wine-quality-1.0.jar \
  /home/ubuntu/TrainingDataset.csv \
  /home/ubuntu/ValidationDataset.csv \
  /home/ubuntu/model
```

The job runs in parallel across all workers. You can watch task progress at http://<MASTER_PUBLIC_IP>:8080. When it finishes, the trained model is saved to /home/ubuntu/model.


## Part 2 - Single-machine prediction without Docker

The prediction application loads the trained model and prints the F1 score on a test CSV.

On any EC2 instance with Spark installed:

```
$SPARK_HOME/bin/spark-submit \
  --class com.cs643.WinePredictor \
  --master local[*] \
  ~/wine-quality-1.0.jar \
  /home/ubuntu/TestDataset.csv \
  /home/ubuntu/model
```

The argument --master local[*] tells Spark to run on a single machine using all available cores, which matches the assignment's requirement of running prediction on one EC2 instance.

Replace TestDataset.csv with the path to whatever test file you want to evaluate. ValidationDataset.csv works too.


## Part 3 - Prediction with Docker

The Docker image already contains the trained model and the prediction application, so it can run anywhere Docker is installed.

### Pull the image

```
docker pull bhavanavuttunoori/wine-predictor:1.0
```

### Run the container

Mount a folder containing your test CSV and pass the CSV name as an argument:

```
docker run --rm \
  -v /path/to/your/data:/app/data \
  bhavanavuttunoori/wine-predictor:1.0 \
  /app/data/TestDataset.csv \
  /app/model
```

Replace /path/to/your/data with the absolute path to the directory holding your test CSV. The model is already inside the container at /app/model.

### Running on a fresh EC2 instance

Launch a single EC2 instance (Ubuntu 22.04, t2.medium), then:

```
sudo apt update
sudo apt install -y docker.io
sudo usermod -aG docker ubuntu
# log out and back in for the group change to apply

docker pull bhavanavuttunoori/wine-predictor:1.0
docker run --rm -v $HOME:/app/data \
  bhavanavuttunoori/wine-predictor:1.0 \
  /app/data/TestDataset.csv /app/model
```


## Building the Docker image from scratch

If you want to rebuild the image instead of pulling it from Docker Hub:

```
mvn clean package
docker build -t wine-predictor:1.0 .
```

To test it locally:

```
docker run --rm -v $(pwd)/data:/app/data wine-predictor:1.0 \
  /app/data/ValidationDataset.csv /app/model
```

The Dockerfile in this repo uses bitnamilegacy/spark:3.5.0 as the base image. Bitnami moved their free images to the bitnamilegacy namespace on Docker Hub in September 2025, but the image content is the same as the original bitnami/spark:3.5.0.


## Notes on the dataset

The provided CSV files use semicolons as the column separator, and the headers contain nested double quotes (for example "fixed acidity" appears as ""fixed acidity""). The loadWineCsv method in WineTrainer.java handles this by reading with sep=";", then renaming columns positionally to clean names like fixed_acidity and volatile_acidity, and casting all numeric columns to double.


## Results

The F1 score is the same in every environment, which confirms the pipeline is consistent across local, distributed, and containerized runs:

- Local training on a laptop: 0.5549
- Parallel training on 4 EC2 instances: 0.5549
- Single-machine prediction on EC2 without Docker: 0.5549
- Prediction inside the Docker container: 0.5549


## Troubleshooting

If Spark workers do not connect to the master, check that the security group allows all TCP traffic within itself, and that the master URL uses the master's private IP rather than its public IP.

On Windows, Spark needs winutils.exe and hadoop.dll for local model saves. Place them in C:\hadoop\bin and set HADOOP_HOME=C:\hadoop.

If the docker build fails with "bitnami/spark:3.5.0 not found", use bitnamilegacy/spark:3.5.0 instead.

If training runs out of memory, either reduce numTrees in WineTrainer.java or use a larger instance type (for example t2.large).
