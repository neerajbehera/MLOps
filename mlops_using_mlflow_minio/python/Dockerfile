# python/Dockerfile
FROM jupyter/base-notebook:latest

USER root

# Install OpenJDK 17
RUN apt-get update && apt-get install -y openjdk-17-jdk-headless && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

# Set JAVA_HOME
ENV JAVA_HOME=/usr/lib/jvm/java-17-openjdk-arm64
ENV PATH=$JAVA_HOME/bin:$PATH

# Install Spark
ENV SPARK_VERSION=3.5.0
ENV SPARK_HOME=/usr/local/spark
RUN wget -qO- https://archive.apache.org/dist/spark/spark-${SPARK_VERSION}/spark-${SPARK_VERSION}-bin-hadoop3.tgz | \
    tar xz -C /usr/local/ && \
    mv /usr/local/spark-${SPARK_VERSION}-bin-hadoop3 $SPARK_HOME

# Download Hadoop AWS JARs and place them in Spark's jars directory
RUN wget -P $SPARK_HOME/jars/ https://repo1.maven.org/maven2/org/apache/hadoop/hadoop-aws/3.3.4/hadoop-aws-3.3.4.jar && \
    wget -P $SPARK_HOME/jars/ https://repo1.maven.org/maven2/com/amazonaws/aws-java-sdk-bundle/1.12.262/aws-java-sdk-bundle-1.12.262.jar

# Install Python packages
COPY requirements.txt /tmp/
RUN pip install --no-cache-dir -r /tmp/requirements.txt

USER jovyan
WORKDIR /home/jovyan